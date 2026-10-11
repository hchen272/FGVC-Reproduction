"""Thin runtime wrapper around the pinned, otherwise unchanged P2P-Net sources.

Training uses the author model and losses. The only source compatibility change
is the removed NumPy ``np.int`` alias; original files remain untouched.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import threading
from types import SimpleNamespace

AUTHOR_COMMIT = "400ad83459634597ed1f37ec6b9dad1849c5fee6"
AUTHOR_WEIGHT_SHA256_PREFIX = "19c8e357"
_IMPORT_LOCK = threading.RLock()
_INT_ALIAS = re.compile(r"\bnp\.int\b")
_MODULE_ORDER = ("anchors", "clustering", "Resnet", "dataset", "model", "utils")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_author(source_root):
    """Load author modules in an isolated, hash-keyed package.

    The cache package is importable by spawned DataLoader workers. Temporary
    absolute-import aliases accommodate the author's sibling imports; preexisting
    modules with those names are restored before returning.
    """
    source_root = Path(source_root).resolve()
    source_files = sorted(source_root.glob("*.py"))
    if not source_files:
        raise FileNotFoundError(f"No P2P author sources in {source_root}")
    required = {f"{name}.py" for name in _MODULE_ORDER}
    if not required.issubset({p.name for p in source_files}):
        raise FileNotFoundError("Incomplete P2P author source checkout")
    changes = {}
    source_digest = hashlib.sha256()
    effective = {}
    for path in source_files:
        original = path.read_bytes()
        original_text = original.decode("utf-8")
        patched_text, replacements = _INT_ALIAS.subn("np.int64", original_text)
        patched = patched_text.encode("utf-8")
        effective[path.name] = patched
        source_digest.update(path.name.encode("utf-8") + b"\0" + original + b"\0")
        changes[path.name] = {
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "effective_sha256": hashlib.sha256(patched).hexdigest(),
            "np_int_replacements": replacements,
        }
    source_sha = source_digest.hexdigest()
    package_name = "_p2p_author_" + source_sha[:20]
    cache_parent = source_root.parent / "_p2p_runtime_cache"
    cache = cache_parent / package_name
    metadata = {
        "author_commit": AUTHOR_COMMIT,
        "source_root": Path(os.path.relpath(source_root, Path(__file__).resolve().parents[2])).as_posix(),
        "source_sha256": source_sha,
        "cache_root": Path(os.path.relpath(cache, Path(__file__).resolve().parents[2])).as_posix(),
        "package_name": package_name,
        "compatibility": "exact token np.int -> np.int64 in isolated source copies",
        "numpy_global_patch": False,
        "source_files": changes,
        "preserved_author_behaviors": [
            "graph_assign cumulative permutation behavior",
            "cosine epoch modulo nb_epoch, including final-epoch learning-rate reset",
            "CUB resize550/crop448 and mean0.5/std0.5 normalization",
            "all three regularization targets retain gradients",
            "author full inference computes part branch; predictions sum global heads1..4",
        ],
    }
    with _IMPORT_LOCK:
        cache.mkdir(parents=True, exist_ok=True)
        for name, content in effective.items():
            destination = cache / name
            if not destination.exists() or destination.read_bytes() != content:
                destination.write_bytes(content)
        init = cache / "__init__.py"
        if not init.exists():
            init.write_text("# Runtime namespace for unmodified author modules.\n", encoding="utf-8")
        (cache / "compatibility_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        # Spawned workers inherit sys.path and can import the dataset class.
        cache_parent_string = str(cache_parent)
        if cache_parent_string not in sys.path:
            sys.path.insert(0, cache_parent_string)
        missing = object()
        previous = {name: sys.modules.get(name, missing) for name in _MODULE_ORDER}
        loaded = {}
        try:
            importlib.import_module(package_name)
            for name in _MODULE_ORDER:
                module = importlib.import_module(package_name + "." + name)
                loaded[name] = module
                sys.modules[name] = module
        finally:
            for name, old in previous.items():
                if old is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old
    return SimpleNamespace(**loaded, metadata=metadata)


def make_model(author, weights_path, device):
    """Build the exact author RN50 P2P configuration using ImageNet-V1 weights."""
    import torch

    weights_path = Path(weights_path)
    weight_sha = _sha256(weights_path)
    if not weight_sha.startswith(AUTHOR_WEIGHT_SHA256_PREFIX):
        raise ValueError(
            "Expected the author's resnet50-19c8e357.pth ImageNet-V1 checkpoint; "
            f"received SHA256 {weight_sha}"
        )
    backbone = author.Resnet.resnet50(pretrained=False)
    # Official 19c8e357 file is a legacy tar checkpoint. Its hash was checked above.
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(state, strict=True)
    model = author.model.PMG(backbone, 512, 2048, 200, 4)
    author.metadata["initialization"] = {
        "weights_path": Path(os.path.relpath(weights_path, Path(__file__).resolve().parents[2])).as_posix(),
        "weights_sha256": weight_sha,
        "weights_url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        "backbone": "author Resnet.resnet50",
        "strict_load": True,
        "feature_size": 512,
        "num_ftrs": 2048,
        "classes_num": 200,
        "topn": 4,
    }
    return model.to(device)


def forward_loss(author, model, inputs, targets):
    """Reproduce train.py losses without detaching regularization targets.

    Returns scalar total loss, detached scalar diagnostics, and the sum of four
    global-head logits used by the author for classification.
    """
    import torch
    import torch.nn.functional as F

    y1, y2, y3, y4, yp1, yp2, yp3, yp4, part_probs, f1_m, f1, f2_m, f2, f3_m, f3 = model(inputs)
    raw = model.module if hasattr(model, "module") else model
    batch_size, topn = inputs.shape[0], raw.topn
    peaks = (0.7, 0.8, 0.9, 1.0)
    image_losses = [author.utils.smooth_CE(logit, targets, peak)
                    for logit, peak in zip((y1, y2, y3, y4), peaks)]
    targets_parts = targets.unsqueeze(1).repeat(1, topn).view(-1)
    part_losses = [author.utils.smooth_CE(logit, targets_parts, peak)
                   for logit, peak in zip((yp1, yp2, yp3, yp4), peaks)]
    ranking = author.utils.ranking_loss(
        part_probs, author.utils.list_loss(yp4, targets_parts).view(batch_size, topn)
    )
    regularizers = [torch.mean(-torch.sum(
        F.log_softmax(predicted, dim=-1) * F.softmax(original, dim=-1), dim=-1
    )) * 0.1 for predicted, original in ((f1_m, f1), (f2_m, f2), (f3_m, f3))]
    loss_img = image_losses[0] + image_losses[1] + image_losses[2] + image_losses[3]
    loss_parts = part_losses[0] + part_losses[1] + part_losses[2] + part_losses[3] + ranking
    loss_reg = regularizers[0] + regularizers[1] + regularizers[2]
    loss = loss_img + loss_parts + loss_reg
    components = {
        "loss_img": float(loss_img.detach().item()),
        "loss_parts": float(loss_parts.detach().item()),
        "loss_reg": float(loss_reg.detach().item()),
        "loss_rank": float(ranking.detach().item()),
    }
    for index, term in enumerate(image_losses, 1):
        components[f"loss_img_{index}"] = float(term.detach().item())
    for index, term in enumerate(part_losses, 1):
        components[f"loss_part_{index}"] = float(term.detach().item())
    for index, term in enumerate(regularizers, 1):
        components[f"loss_reg_{index}"] = float(term.detach().item())
    return loss, components, y1 + y2 + y3 + y4


def fast_logits(model, x):
    """Global-only inference with the author's sum-of-four-global-head logits.

    Requires eval mode: omitting part computations must not alter BatchNorm
    updates or PR state. This shortcut must be checked against full author
    inference before reporting timings/results.
    """
    import torch

    raw = model.module if hasattr(model, "module") else model
    if raw.training:
        raise ValueError("fast_logits requires model.eval(); training must use full author forward")
    _, _, f1, f2, f3 = raw.backbone(x)
    batch = x.shape[0]
    f1 = raw.conv_block1(f1).view(batch, -1)
    f2 = raw.conv_block2(f2).view(batch, -1)
    f3 = raw.conv_block3(f3).view(batch, -1)
    y1 = raw.classifier1(f1)
    y2 = raw.classifier2(f2)
    y3 = raw.classifier3(f3)
    y4 = raw.classifier_concat(torch.cat((f1, f2, f3), -1))
    return y1 + y2 + y3 + y4


def pose_state(model):
    """Return checkpoint-safe PR state (PR is not an nn.Module)."""
    raw = model.module if hasattr(model, "module") else model
    pr = raw.PR
    return {
        "centers": pr.centers.tolist(),
        "centers_dtype": str(pr.centers.dtype),
        "count": int(pr.count),
        "num_center": int(pr.num_center),
        "feature_dim": int(pr.feature_dim),
    }


def load_pose_state(model, state):
    """Restore PR centers/count without changing author clustering behavior."""
    import numpy as np

    raw = model.module if hasattr(model, "module") else model
    pr = raw.PR
    centers = np.asarray(state["centers"], dtype=state.get("centers_dtype", "float64"))
    if centers.shape != pr.centers.shape:
        raise ValueError(f"PR center shape mismatch: {centers.shape} != {pr.centers.shape}")
    if int(state.get("num_center", pr.num_center)) != pr.num_center:
        raise ValueError("PR num_center mismatch")
    if int(state.get("feature_dim", pr.feature_dim)) != pr.feature_dim:
        raise ValueError("PR feature_dim mismatch")
    count = int(state["count"])
    if count < 0 or not np.isfinite(centers).all():
        raise ValueError("Invalid saved PR state")
    pr.centers = centers.copy()
    pr.count = count


def _read_pairs(path):
    rows = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        identifier, value = line.split(maxsplit=1)
        identifier = int(identifier)
        if identifier in rows:
            raise ValueError(f"Duplicate ID in {path}:{line_number}: {identifier}")
        rows[identifier] = value
    return rows


def raw_metadata(data_root):
    """Parse official CUB metadata by IDs, preserving images.txt row order."""
    root = Path(data_root)
    images = _read_pairs(root / "images.txt")
    labels = _read_pairs(root / "image_class_labels.txt")
    splits = _read_pairs(root / "train_test_split.txt")
    class_names = _read_pairs(root / "classes.txt")
    if set(images) != set(labels) or set(images) != set(splits):
        raise ValueError("CUB metadata image IDs do not align")
    if set(class_names) != set(range(1, 201)):
        raise ValueError("Expected the 200 official CUB classes")
    classes = [class_names[index] for index in range(1, 201)]
    train_rows, test_rows = [], []
    for image_id, image in images.items():
        official_split = int(splits[image_id])
        class_id = int(labels[image_id])
        if official_split not in (0, 1) or class_id not in class_names:
            raise ValueError(f"Invalid CUB split/class for image {image_id}")
        relative = PurePosixPath(image)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid CUB relative image path {image}")
        if relative.parts[0] != class_names[class_id]:
            raise ValueError(f"CUB path and species label disagree for image {image_id}")
        split = "train" if official_split == 1 else "test"
        row = {"cub_image_id": image_id, "image": "images/" + relative.as_posix(),
               "label": class_id - 1, "split": split, "sample_id": f"cub_{image_id:05d}"}
        (train_rows if split == "train" else test_rows).append(row)
    if len(train_rows) != 5994 or len(test_rows) != 5794 or len(classes) != 200:
        raise ValueError(f"Unexpected official CUB sizes: {len(train_rows)}/{len(test_rows)}/{len(classes)}")
    return classes, train_rows, test_rows
