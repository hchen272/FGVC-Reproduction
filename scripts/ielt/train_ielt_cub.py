#!/usr/bin/env python3
"""Offline, auditable Elusive Images IELT CUB recipe with frozen ordinary ViT training helpers.

Official CUB train (5994) is used in full; official test (5794) is monitored
each epoch, as in the author's val loader, but NEVER selects a checkpoint.
The fixed last epoch is evaluated and benchmarked. No best.pt is produced.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

AUTHOR_COMMIT = "37639fd61f3d8b0b1a23a5d9922a48cd2b0f8fb0"
AUTHOR_REPO = "https://github.com/catalys1/elusive-images-fgvc"
R50_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
RANKING_POLICY = "logits_float32_descending_stable_class_index_tiebreak_v1"


def load_runtime():
    # Keep --help, parsing and source audits usable without importing local torch.
    global np, torch, nn, Dataset, DataLoader, T, Image, timm, tqdm, torchvision
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    import torchvision.transforms as T
    from PIL import Image
    import timm
    from tqdm.auto import tqdm


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


class Status:
    def __init__(self, output, args):
        self.path, self.started, self.last_write = output / "status.json", time.monotonic(), 0.0
        self.data = {"run_id": output.name, "model": args.model, "seed": args.seed,
                     "dataset_name": "CUB-200-2011", "input_mode": "full", "state": "running",
                     "stage": "initializing", "epoch": 0, "epochs": args.epochs,
                     "batch": 0, "batches": 0, "completed": 0, "total": 0,
                     "progress": 0.0, "metrics": {}, "history": [], "error": None,
                     "smoke_run": args.smoke, "selection_policy": "fixed_last_epoch"}
        self.update(force=True)

    def update(self, force=False, **values):
        self.data.update(values)
        self.data["updated_at"] = utc_now()
        self.data["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        if self.data["total"]:
            self.data["progress"] = min(1.0, self.data["completed"] / self.data["total"])
        if force or time.monotonic() - self.last_write >= 2.0:
            atomic_json(self.path, self.data)
            self.last_write = time.monotonic()


def parse_pairs(path):
    result = {}
    with Path(path).open(encoding="utf8") as stream:
        for line_number, line in enumerate(stream, 1):
            key, value = line.strip().split(maxsplit=1)
            key = int(key)
            if key in result:
                raise ValueError(f"Duplicate id {key} in {path}:{line_number}")
            result[key] = value
    return result


def load_official_cub(root):
    names = parse_pairs(root / "classes.txt")
    images = parse_pairs(root / "images.txt")
    labels = parse_pairs(root / "image_class_labels.txt")
    splits = parse_pairs(root / "train_test_split.txt")
    if set(names) != set(range(1, 201)) or len(images) != 11788:
        raise ValueError("Expected unmodified official CUB-200-2011: 200 classes, 11788 images")
    if not set(images) == set(labels) == set(splits):
        raise ValueError("Official image ids differ across annotations")
    classes = [names[index] for index in range(1, 201)]
    records = {"train": [], "test": []}
    seen_paths = set()
    for image_id in sorted(images):
        value = images[image_id].replace("\\", "/")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or ":" in value or value in seen_paths:
            raise ValueError(f"Invalid or duplicate image path: {value}")
        seen_paths.add(value)
        label, is_train = int(labels[image_id]), int(splits[image_id])
        if not 1 <= label <= 200 or is_train not in (0, 1):
            raise ValueError(f"Invalid official annotation for image {image_id}")
        image = root / "images" / relative
        if not image.is_file():
            raise FileNotFoundError(image)
        split = "train" if is_train else "test"
        records[split].append({"sample_id": f"cub_{image_id:05d}", "cub_image_id": image_id,
                               "image": "images/" + value, "label": label - 1,
                               "species": classes[label - 1], "split": split,
                               "official_train": bool(is_train)})
    if {key: len(value) for key, value in records.items()} != {"train": 5994, "test": 5794}:
        raise ValueError("Official CUB train/test population mismatch")
    if any(len({r["label"] for r in rows}) != 200 for rows in records.values()):
        raise ValueError("Official splits must cover all 200 classes")
    return classes, records


def write_manifest(output, records):
    destination = output / "dataset_manifest.jsonl"
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf8") as stream:
        for split in ("train", "test"):
            for record in records[split]:
                stream.write(canonical(record) + "\n")
    os.replace(temporary, destination)
    return sha256_file(destination)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Lightning's default deterministic=None does not request deterministic algorithms.
    # benchmark=False is the default; do not add TF32 or cudnn benchmark optimizations.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def get_rng(generators):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "loaders": {key: value.get_state() for key, value in generators.items()}}


def restore_rng(state, generators):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    for key, generator in generators.items():
        generator.set_state(state["loaders"][key].cpu())


def make_datasets(root, records):
    train_transform = T.Compose([
        T.RandomResizedCrop((448, 448), scale=(0.08, 1.0), ratio=(0.75, 1 / 0.75),
                            interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ColorJitter(0.1, 0.1, 0.1), T.RandomHorizontalFlip(0.5), T.PILToTensor(),
    ])
    test_transform = T.Compose([
        T.Resize([512, 512], interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop((448, 448)), T.PILToTensor(),
    ])

    class OfficialImages(Dataset):
        def __init__(self, rows, transform):
            self.records, self.transform = rows, transform

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            row = self.records[index]
            try:
                with Image.open(root / row["image"]) as source:
                    # Match author's ordinary image loader: no GT crop, no EXIF rotation.
                    image = source.convert("RGB")
                return self.transform(image), row["label"], index
            except Exception as error:
                raise RuntimeError(f"CUB image {row['cub_image_id']} {row['image']}: {error}") from error

    return {"train": OfficialImages(records["train"], train_transform),
            "test": OfficialImages(records["test"], test_transform)}


def create_model(args):
    from ielt_elusive_core import IELTAdapter
    expected = '69c9a2822a47838f57e522350b10ccc0d89d67fdcee07603e4ae3259b3350336'
    if not args.weights.is_file() or sha256_file(args.weights) != expected:
        raise ValueError('IELT requires the exact existing Google augreg ImageNet-1K NPZ')
    model = IELTAdapter(args.weights, expected_sha256=expected, image_size=448, num_classes=200)
    provenance = {'pretraining': 'ImageNet-1K Google augreg',
        'resolved_model': 'vit_base_patch16_224.augreg_in1k + frozen Elusive IELT',
        'path': str(args.weights), 'sha256': expected, 'file_bytes': args.weights.stat().st_size,
        'offline_loader': 'timm NPZ load before IELT block surgery; no downloaded/random-backbone fallback',
        'img_size': 448, 'global_pool': '', 'proj_drop_rate': 0.1, 'head_drop_rate': 0.7,
        'reinit_final_blocks': True, 'skipped_pretrained_classifier': ['head.weight', 'head.bias'],
        'position_embedding_resize': 'timm NPZ bilinear interpolation, antialias=False, 14x14 to28x28'}
    return model, provenance


def parameter_groups(model, weight_decay):
    return model.make_parameter_groups(weight_decay=weight_decay)


class Metrics:
    def __init__(self, classes=200):
        self.confusion = np.zeros((classes, classes), dtype=np.int64)
        self.samples, self.loss_sum, self.top3, self.top5 = 0, 0.0, 0, 0

    def add(self, logits, labels, loss):
        # Full float32 logits, deterministic tie policy shared with earlier tests.
        values, truth = logits.detach().float().cpu().numpy(), labels.detach().cpu().numpy()
        order = np.argsort(-values, axis=1, kind="stable")
        predictions = order[:, 0]
        np.add.at(self.confusion, (truth, predictions), 1)
        self.samples += len(truth)
        self.loss_sum += float(loss) * len(truth)
        self.top3 += int((order[:, :3] == truth[:, None]).any(1).sum())
        self.top5 += int((order[:, :5] == truth[:, None]).any(1).sum())
        return values, truth, order

    def result(self):
        support, called = self.confusion.sum(1), self.confusion.sum(0)
        tp = np.diag(self.confusion)
        present = support > 0
        accuracy = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=present)
        f1 = np.divide(2 * tp, support + called, out=np.zeros_like(tp, dtype=float), where=(support + called) > 0)
        return {"loss": self.loss_sum / self.samples, "top1": float(tp.sum() / self.samples),
                "top3": self.top3 / self.samples, "top5": self.top5 / self.samples,
                "macro_f1": float(f1[present].mean()), "mean_per_class_accuracy": float(accuracy[present].mean()),
                "samples": self.samples, "classes_evaluated": int(present.sum()),
                "macro_average_policy": "classes_with_ground_truth_support"}

    def report(self, classes):
        rows = []
        for label, species in enumerate(classes):
            actual, predicted, correct = int(self.confusion[label].sum()), int(self.confusion[:, label].sum()), int(self.confusion[label, label])
            rows.append({"label": label, "species": species, "support": actual, "predicted": predicted,
                         "correct": correct, "precision": correct / predicted if predicted else 0.0,
                         "recall": correct / actual if actual else 0.0,
                         "f1": 2 * correct / (actual + predicted) if actual + predicted else 0.0})
        return {"aggregate": self.result(), "per_class": rows}


def amp_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def normalize_on_device(images, device, mean, std):
    # Match author's normalize_on_gpu=true rather than changing normalization order.
    return (images.to(device, non_blocking=True).float().div_(255.0) - mean) / std


def run_epoch(model, loader, optimizer, scheduler, scaler, device, mean, std, args, status,
              epoch, completed, training, collect=False):
    stage = "train" if training else ("final_test" if collect else "test")
    batches = min(len(loader), 3) if args.smoke else len(loader)
    accumulator, saved = Metrics(), {"logits": [], "labels": [], "indices": []}
    model.train(training)
    model.set_epoch(epoch - 1)
    model.updating_layer_selections = bool(training and epoch - 1 >= model.warmup_epochs)
    status.update(force=True, stage=stage, epoch=epoch, batch=0, batches=batches, completed=completed)
    started, skipped = time.monotonic(), 0
    with (torch.enable_grad() if training else torch.inference_mode()), tqdm(total=batches, desc=f"{stage} {epoch}/{args.epochs}", unit="batch") as bar:
        for batch_index, (images, labels, indices) in enumerate(loader):
            if batch_index >= batches:
                break
            images = normalize_on_device(images, device, mean, std)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with amp_context():
                logits, complement_logits = model(images)
                loss = model.loss(logits, complement_logits, labels)
            if (tuple(logits.shape) != (len(labels), 200)
                    or not torch.isfinite(logits).all() or not torch.isfinite(complement_logits).all()):
                raise FloatingPointError(f'Invalid {stage} IELT scores at epoch {epoch}, batch {batch_index + 1}')
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {stage} loss at epoch {epoch}, batch {batch_index + 1}")
            if training:
                scaler.scale(loss).backward()
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Lightning's scheduler interval=step advances once per batch even
                # if FP16's scaler skipped the actual optimizer update on overflow.
                # Record such skips so a run never hides mixed-precision instability.
                skipped += int(scaler.get_scale() < old_scale)
                scheduler.step()
            values, truth, _ = accumulator.add(logits, labels, loss.detach().item())
            if collect:
                saved["logits"].append(values)
                saved["labels"].append(truth)
                saved["indices"].append(indices.numpy())
            completed += 1
            result = accumulator.result()
            existing = dict(status.data.get("metrics", {}))
            existing["train" if training else "test"] = result
            existing["learning_rate"] = optimizer.param_groups[0]["lr"]
            bar.update(1)
            bar.set_postfix(loss=f"{result['loss']:.4f}", top1=f"{result['top1']:.4f}")
            status.update(batch=batch_index + 1, completed=completed, metrics=existing)
            if batch_index + 1 == batches:
                break
    result = accumulator.result()
    result.update({"elapsed_seconds": time.monotonic() - started, "batches": batches,
                   "truncated_smoke_run": args.smoke, "fp16_optimizer_steps_skipped": skipped,
                   "layer_selection_quotas": model.num_selects.detach().cpu().tolist(),
                   "selection_updates_enabled": bool(model.updating_layer_selections)})
    status.update(force=True, completed=completed)
    if collect:
        saved = {key: np.concatenate(value, axis=0) for key, value in saved.items()}
    return result, accumulator, saved, completed


def timing_summary(values):
    values = np.asarray(values, dtype=np.float64) * 1000.0
    return {"iterations": len(values), "mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p90_ms": float(np.quantile(values, 0.9)), "p95_ms": float(np.quantile(values, 0.95)),
            "min_ms": float(values.min()), "max_ms": float(values.max()),
            "std_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def benchmark(model, dataset, device, mean, std, args, status, completed):
    model.eval()
    model.updating_layer_selections = False
    warmup, iterations = (2, 5) if args.smoke else (20, 100)
    samples = min(len(dataset), 20)
    tensor = normalize_on_device(dataset[0][0].unsqueeze(0), device, mean, std)
    scopes = {}
    with torch.inference_mode():
        for scope in ("classifier_only", "preprocess_plus_classifier"):
            for index in range(warmup):
                images = tensor if scope == "classifier_only" else normalize_on_device(dataset[index % samples][0].unsqueeze(0), device, mean, std)
                with amp_context():
                    model(images)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            status.update(force=True, stage="benchmark_" + scope, batch=0, batches=iterations)
            values = []
            with tqdm(total=iterations, desc=scope + " batch=1", unit="prediction") as bar:
                for index in range(iterations):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    images = tensor if scope == "classifier_only" else normalize_on_device(dataset[index % samples][0].unsqueeze(0), device, mean, std)
                    with amp_context():
                        logits = model(images)[0]
                    if tuple(logits.shape) != (1, 200):
                        raise ValueError("Unexpected benchmark output shape")
                    torch.cuda.synchronize(device)
                    values.append(time.perf_counter() - started)
                    completed += 1
                    bar.update(1)
                    status.update(batch=index + 1, completed=completed)
            scopes[scope] = {**timing_summary(values), "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
                             "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    scopes["classifier_only"]["scope"] = "full online backbone + 200-way head; preprocessed tensor already on device; synchronized forward"
    scopes["preprocess_plus_classifier"].update({"unique_samples": samples,
        "scope": "warm-filesystem original-image decode, resize512/center448, ImageNet normalize, transfer + full forward"})
    count = sum(parameter.numel() for parameter in model.parameters())
    result = {"model": args.model, "batch_size": 1, "input_size": [3, 448, 448], "input_mode": "full",
              "precision": "float16", "device": str(device), "gpu": torch.cuda.get_device_name(device),
              "warmup_iterations": warmup, "parameters": count, "total_parameters": count,
              "trainable_parameters": count, "feature_cache_used": False, "smoke_run": args.smoke,
              "benchmark_state": "completed", "valid_for_comparison": not args.smoke,
              **scopes, "excluded": ["YOLO detection", "segmentation", "video decoding", "network transfer", "rendering"]}
    status.update(force=True, completed=completed)
    return result, completed


def source_fingerprints(args):
    root = Path(__file__).resolve().parents[2]
    def record(path):
        path = Path(path)
        relative = Path(os.path.relpath(path.resolve(), root)).as_posix()
        return {"path": relative, "sha256": sha256_file(path)}
    sources = {"trainer": record(__file__)}
    for name in ("classes.txt", "images.txt", "image_class_labels.txt", "train_test_split.txt"):
        sources[name] = record(args.data_root / name)
    author_root = root / "third_party" / "elusive_images"
    for name in ("src/models/base.py", "src/models/ielt.py"):
        sources[name] = record(author_root / name)
    sources["ielt_core"] = record(Path(__file__).with_name("ielt_elusive_core.py"))
    return {"author_repo": AUTHOR_REPO, "author_commit": AUTHOR_COMMIT, "files": sources,
            "source_sha256": {"train_ielt_cub.py": sources["trainer"]["sha256"],
                              "ielt_elusive_core.py": sources["ielt_core"]["sha256"]},
            "author_sources": {"files_sha256": {k: v["sha256"] for k, v in sources.items() if k.startswith("src/")}}}


def save_predictions(output, saved, records, classes):
    logits, labels, indices = saved["logits"], saved["labels"], saved["indices"]
    if len(np.unique(indices)) != len(indices) or not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("Final test loader changed official image order")
    order = np.argsort(-logits, axis=1, kind="stable")
    stable_logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(stable_logits.astype(np.float64))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    path = output / "predictions.csv"
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "cub_image_id", "true_label", "true_species", "predicted_label",
                         "predicted_species", "top1_probability", "correct", "top3_labels", "top3_species", "image"])
        for index, true_label, candidates, scores in zip(indices, labels, order, probabilities):
            record = records[int(index)]
            predicted = int(candidates[0])
            writer.writerow([record["sample_id"], record["cub_image_id"], int(true_label), classes[int(true_label)],
                             predicted, classes[predicted], float(scores[predicted]), int(predicted == true_label),
                             json.dumps(candidates[:3].tolist()), json.dumps([classes[int(c)] for c in candidates[:3]]), record["image"]])
    os.replace(temporary, path)
    atomic_npz(output / "test_logits.npz", logits=logits.astype(np.float32), labels=labels.astype(np.int64),
               cub_image_ids=np.array([records[int(index)]["cub_image_id"] for index in indices], dtype=np.int64),
               sample_ids=np.array([records[int(index)]["sample_id"] for index in indices]),
               classes=np.array(classes))


def train(args, status):
    load_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("Author 16-mixed recipe requires CUDA; no CPU/random-weight fallback")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This matched recipe only accepts CUDA")
    torch.cuda.set_device(device)
    output = args.output_dir
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("Use a fresh run directory or --resume auto; existing checkpoints are never overwritten")
    if (output / "completed.json").exists() and not args.resume:
        raise FileExistsError("Output has a completed marker; choose a new run or use --resume auto")
    seed_everything(args.seed)
    classes, records = load_official_cub(args.data_root)
    manifest_sha = write_manifest(output, records)
    atomic_json(output / "classes.json", classes)
    fingerprints = source_fingerprints(args)
    atomic_json(output / "source_fingerprints.json", fingerprints)
    status.update(force=True, stage="loading_local_pretrained_weights", dataset={"train": 5994, "test": 5794})
    model, provenance = create_model(args)
    groups, assignments = parameter_groups(model, 5e-4 if args.model == "resnet50" else 1e-5)
    atomic_json(output / "parameter_groups.json", assignments)
    model.to(device)
    datasets = make_datasets(args.data_root, records)
    generators = {key: torch.Generator().manual_seed(args.seed + (0 if key == "train" else 100000))
                  for key in ("train", "test")}
    loaders = {key: DataLoader(dataset, batch_size=args.batch_size if key == "train" else args.eval_batch_size,
                              shuffle=key == "train", num_workers=args.num_workers, pin_memory=True,
                              drop_last=key == "train", worker_init_fn=seed_worker, generator=generators[key],
                              persistent_workers=False)
               for key, dataset in datasets.items()}
    batches = {key: min(len(loader), 3) if args.smoke else len(loader) for key, loader in loaders.items()}
    optimizer = torch.optim.SGD(groups, lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[group["lr"] for group in groups],
                                                   total_steps=args.epochs * batches["train"],
                                                   pct_start=1e-4, cycle_momentum=False)
    scaler = torch.amp.GradScaler("cuda")
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    settings = {"model": args.model, "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size, "num_workers": args.num_workers,
                "smoke_run": args.smoke, "manifest_sha256": manifest_sha,
                "classes_sha256": sha256_file(output / "classes.json"), "weights_sha256": provenance["sha256"],
                "author_commit": AUTHOR_COMMIT, "torch_version": str(torch.__version__),
                "torchvision_version": str(torchvision.__version__), "timm_version": str(timm.__version__),
                "script_sha256": fingerprints["files"]["trainer"]["sha256"],
                "core_sha256": fingerprints['files']['ielt_core']['sha256'],
                "author_sources_sha256": {k: v['sha256'] for k, v in fingerprints['files'].items() if k.startswith('src/')},
                "defer_benchmark": args.defer_benchmark, "selection_policy": "fixed_last_epoch"}
    config = {**settings, "dataset_name": "CUB-200-2011", "data_root": str(args.data_root),
              "output_dir": str(output), "recipe": "Elusive Images IELT, matched ordinary ViT data/weights/training recipe",
              "ielt": {"votes_per_head": 24, "total_selections": 126, "warmup_epochs": 10,
                       "alpha": 0.4, "loss": ".6CE(part)+.4CE(complement)",
                       "prediction": "part_logits including assist_logits; no two-head averaging",
                       "reinit_final_blocks": True, "proj_drop_rate": .1, "head_drop_rate": .7,
                       "resume_adaptation": "Rebind per-block quota references after device move and checkpoint loading",
                       "evaluation": "Disable dynamic quota updates; use complete5794 official test"},
              "author_repo": AUTHOR_REPO, "author_commit": AUTHOR_COMMIT, "pretrained_weights": provenance,
              "input_mode": "full", "image_size": 448, "sample_counts": {"train": 5994, "test": 5794},
              "training_population": "all official train; no separate val split",
              "test_usage": "official test monitored each epoch; fixed last epoch only; never best-selection/early-stop",
              "early_stopping": False, "gradient_accumulation": 1, "gradient_clipping": None,
              "precision": "16-mixed", "normalization_on_gpu": True, "normalization_mean": MEAN,
              "normalization_std": STD, "train_transform": "RandomResizedCrop448(scale=.08..1,ratio=.75..1/.75,bilinear,antialias),ColorJitter(.1,.1,.1),HorizontalFlip(.5)",
              "test_transform": "Resize([512,512],bilinear,antialias),CenterCrop448",
              "drop_last_train": True, "train_samples_per_epoch": batches["train"] * args.batch_size,
              "batches_per_epoch": batches, "optimizer": "SGD", "momentum": 0.9, "nesterov": False,
              "max_learning_rates": [0.01, 0.01, 0.001, 0.001], "weight_decay": 5e-4 if args.model == "resnet50" else 1e-5,
              "scheduler": "OneCycleLR", "pct_start": 1e-4, "cycle_momentum": False,
              "anneal_strategy": "cos", "div_factor": 25.0, "final_div_factor": 10000.0,
              "total_optimizer_steps": args.epochs * batches["train"],
              "onecycle_on_fp16_overflow": "advance per batch, matching Lightning interval=step; skips logged",
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
              "gpu": torch.cuda.get_device_name(device), "ranking_policy": RANKING_POLICY,
              "seed_policy": "three independent runs seed42/43/44; no ensemble; fixed official split",
              "resume_settings": settings, "created_at": utc_now()}
    start_epoch, history = 0, []
    if args.resume and (args.resume != "auto" or (output / "last.pt").is_file()):
        path = output / "last.pt" if args.resume == "auto" else Path(args.resume)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["resume_settings"] != settings or checkpoint["classes"] != classes:
            raise ValueError("Resume recipe, data, weights, runtime or class ordering changed")
        start_epoch, history = checkpoint["epoch_completed"], checkpoint["history"]
        if start_epoch != len(history) or start_epoch > args.epochs:
            raise ValueError("Resume completed epoch does not match history")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], generators)
        config["resumed_from"] = str(path)
        del checkpoint
    atomic_json(output / "config.json", config)
    benchmark_iterations = 5 if args.smoke else 100
    total = args.epochs * (batches["train"] + batches["test"]) + batches["test"] + (0 if args.defer_benchmark and not args.smoke else 2 * benchmark_iterations)
    completed = start_epoch * (batches["train"] + batches["test"])
    status.update(force=True, total=total, completed=completed, history=history, precision="float16",
                  image_size=448, sample_counts=config["sample_counts"], selected_epoch=args.epochs,
                  metrics={"train": history[-1]["train"], "test": history[-1]["test"]} if history else {})
    training_started = time.monotonic()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics, _, _, completed = run_epoch(model, loaders["train"], optimizer, scheduler, scaler,
            device, mean, std, args, status, epoch, completed, True)
        test_metrics, _, _, completed = run_epoch(model, loaders["test"], optimizer, scheduler, scaler,
            device, mean, std, args, status, epoch, completed, False)
        history.append({"epoch": epoch, "train": train_metrics, "test": test_metrics,
                        "learning_rate": optimizer.param_groups[0]["lr"], "updated_at": utc_now()})
        status.update(force=True, stage="saving_checkpoint", history=history,
                      metrics={"train": train_metrics, "test": test_metrics,
                               "learning_rate": optimizer.param_groups[0]["lr"]})
        checkpoint = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch_completed": epoch,
                      "selected_epoch": args.epochs, "selection_policy": "fixed_last_epoch", "history": history,
                      "classes": classes, "config": config, "rng": get_rng(generators), "resume_settings": settings}
        # Atomic epoch boundary; interruptions replay the unfinished epoch, never
        # resume a partly consumed iterator or silently reset optimizer/RNG state.
        atomic_torch_save(output / "last.pt", checkpoint)
        atomic_json(output / "history.json", history)
        del checkpoint
        print(f"Epoch {epoch}/{args.epochs} train_top1={train_metrics['top1']:.6f} test_monitor_top1={test_metrics['top1']:.6f}; fixed final epoch={args.epochs}", flush=True)
    # Last checkpoint exists also when resumed after the final epoch completed.
    selected = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    if selected["epoch_completed"] != args.epochs or selected["resume_settings"] != settings:
        raise ValueError("Fixed-final checkpoint is not the configured last epoch")
    model.load_state_dict(selected["model"], strict=True)
    atomic_torch_save(output / "final.pt", selected)
    del selected
    final_metrics, accumulator, saved, completed = run_epoch(model, loaders["test"], optimizer, scheduler,
        scaler, device, mean, std, args, status, args.epochs, completed, False, collect=True)
    if not args.smoke and final_metrics["samples"] != 5794:
        raise ValueError("Final official test incomplete")
    save_predictions(output, saved, records["test"], classes)
    report = {**accumulator.report(classes), "model": args.model, "seed": args.seed,
              "dataset_name": "CUB-200-2011", "input_mode": "full", "image_size": 448,
              "selected_epoch": args.epochs, "selection_policy": "fixed_last_epoch",
              "test_selection_policy": "fixed last epoch; official test monitor never selects checkpoint",
              "manifest_sha256": manifest_sha, "smoke_run": args.smoke, "ranking_policy": RANKING_POLICY,
              "pretraining": provenance["pretraining"], "precision": "float16", "output_classes": 200,
              "feature_cache_used": False, "trained_this_invocation_seconds": time.monotonic() - training_started,
              "final_test": final_metrics, "author_commit": AUTHOR_COMMIT}
    atomic_json(output / "classification_report.json", report)
    atomic_npz(output / "test_confusion.npz", confusion=accumulator.confusion)
    # Inference memory includes model/inputs only, not SGD momentum or training
    # gradients. Last/final checkpoints already contain all resume state.
    del optimizer, scheduler, scaler
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    if args.defer_benchmark and not args.smoke:
        timing = {'model': args.model, 'benchmark_state': 'pending', 'valid_for_comparison': False,
                  'reason': 'Deferred until all three training runs finish and the GPU is idle; no latency result yet.'}
    else:
        timing, completed = benchmark(model, datasets["test"], device, mean, std, args, status, completed)
    atomic_json(output / "benchmark.json", timing)
    files = ["last.pt", "final.pt", "classification_report.json", "predictions.csv", "test_logits.npz",
             "test_confusion.npz", "benchmark.json", "config.json", "dataset_manifest.jsonl", "history.json",
             "classes.json", "parameter_groups.json", "source_fingerprints.json"]
    artifacts = {name: {"path": name, "sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
                 for name in files}
    marker = {"schema_version": 1, "state": "completed", "run_id": output.name, "model": args.model,
              "seed": args.seed, "dataset_name": "CUB-200-2011", "epochs": args.epochs,
              "epoch_completed": args.epochs, "selected_epoch": args.epochs, "selection_policy": "fixed_last_epoch",
              "sample_counts": {"train": 5994, "test": 5794}, "image_size": 448, "input_mode": "full",
              "smoke_run": args.smoke, "aggregate": report["aggregate"], "artifacts": artifacts,
              "benchmark_state": timing['benchmark_state'], "completed_at": utc_now()}
    atomic_json(output / "completed.json", marker)
    status.update(force=True, state="completed", stage="done", completed=total, total=total,
                  epoch=args.epochs, history=history, metrics={"train": history[-1]["train"], "test": final_metrics},
                  selected_epoch=args.epochs, benchmark=timing, classification_report=str(output / "classification_report.json"))
    print(canonical({"state": "completed", "output": str(output), "seed": args.seed,
                     "model": args.model, "smoke_run": args.smoke, "aggregate": report["aggregate"]}), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", choices=("ielt",), required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--weights", type=Path, required=True)
    result.add_argument("--epochs", type=int, default=50)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--eval-batch-size", type=int, default=32)
    result.add_argument("--num-workers", type=int, default=8)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--resume", nargs="?", const="auto")
    result.add_argument("--smoke", action="store_true", help="Isolated one-epoch three-batch training/evaluation plus small benchmark")
    result.add_argument('--defer-benchmark', action='store_true')
    result.add_argument('--benchmark-only', action='store_true')
    return result



def other_gpu_processes():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return sorted({int(line.strip()) for line in raw.splitlines() if line.strip().isdigit() and int(line.strip()) != os.getpid()})


def benchmark_only(args):
    # A busy GPU is an explicit retry state, never a measured latency.
    others = other_gpu_processes()
    if others:
        print(canonical({'benchmark_state': 'waiting_for_gpu', 'other_gpu_pids': others}), flush=True)
        return 75
    load_runtime()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    output = args.output_dir
    marker = json.loads((output / 'completed.json').read_text(encoding='utf-8'))
    if marker['seed'] != args.seed or marker['model'] != 'ielt' or marker['selected_epoch'] != 50:
        raise ValueError('Benchmark requires the matching formal fixed-final IELT checkpoint')
    for artifact in marker['artifacts'].values():
        path = output / artifact['path']
        if path.resolve().parent != output.resolve() or sha256_file(path) != artifact['sha256']:
            raise ValueError('Training artifact hash mismatch before benchmark')
    classes, records = load_official_cub(args.data_root)
    checkpoint = torch.load(output / 'final.pt', map_location='cpu', weights_only=False)
    if checkpoint['classes'] != classes or checkpoint['epoch_completed'] != 50:
        raise ValueError('Benchmark checkpoint class order or epoch differs')
    if checkpoint['resume_settings']['script_sha256'] != sha256_file(__file__):
        raise ValueError('Trainer changed since training')
    if checkpoint['resume_settings']['core_sha256'] != sha256_file(Path(__file__).with_name('ielt_elusive_core.py')):
        raise ValueError('IELT core changed since training')
    model, provenance = create_model(args)
    if provenance['sha256'] != checkpoint['resume_settings']['weights_sha256']:
        raise ValueError('Pretraining source changed')
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.to(device).eval()
    model.updating_layer_selections = False
    datasets = make_datasets(args.data_root, records)
    previous = json.loads((output / 'status.json').read_text(encoding='utf-8'))
    status = Status(output, args)
    status.data.update(previous)
    status.update(force=True, state='running', stage='benchmark', error=None)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    timing, completed = benchmark(model, datasets['test'], device, mean, std, args, status, previous['completed'])
    if other_gpu_processes():
        atomic_json(output / 'benchmark.json', {'model': 'ielt', 'benchmark_state': 'pending',
                    'valid_for_comparison': False, 'reason': 'Another GPU job started during measurement; retry required.'})
        status.update(force=True, state='completed', stage='done', benchmark_state='pending')
        # Keep the previous pending artifact marker valid until a successful retry.
        marker['artifacts']['benchmark.json']['sha256'] = sha256_file(output / 'benchmark.json')
        marker['artifacts']['benchmark.json']['bytes'] = (output / 'benchmark.json').stat().st_size
        atomic_json(output / 'completed.json', marker)
        return 75
    atomic_json(output / 'benchmark.json', timing)
    marker['artifacts']['benchmark.json']['sha256'] = sha256_file(output / 'benchmark.json')
    marker['artifacts']['benchmark.json']['bytes'] = (output / 'benchmark.json').stat().st_size
    marker['benchmark_state'] = 'completed'
    marker['benchmark_completed_at'] = utc_now()
    atomic_json(output / 'completed.json', marker)
    status.update(force=True, state='completed', stage='done', completed=previous['total'],
                  total=previous['total'], benchmark=timing, benchmark_state='completed')
    print(canonical({'benchmark_state': 'completed', 'run_id': output.name}), flush=True)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if args.smoke:
        args.epochs = 1
        if not any("smoke" in part.lower() for part in args.output_dir.parts):
            raise ValueError("Smoke outputs must use a path containing a dedicated 'smoke' directory")
    elif args.epochs != 50 or args.batch_size != 16 or args.eval_batch_size != 32:
        raise ValueError("Formal author calibration fixes 50 epochs, train batch16, test batch32")
    if args.num_workers < 0 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("Invalid loader arguments")
    if args.benchmark_only:
        try:
            return benchmark_only(args)
        except Exception:
            traceback.print_exc()
            return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = Status(args.output_dir, args)
    try:
        train(args, status)
    except BaseException as error:
        status.update(force=True, state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
