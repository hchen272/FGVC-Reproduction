#!/usr/bin/env python3
"""Run the CUB reproduction recipes locally, one model and seed at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import sys
from pathlib import Path, PureWindowsPath


REPO = Path(__file__).resolve().parents[1]
MODELS = ("resnet50", "vit", "wsdan", "ielt", "p2p")
METRICS = ("top1", "top3", "top5", "inference_ms", "with_preprocess_ms")
CUB_METADATA = ("classes.txt", "images.txt", "image_class_labels.txt", "train_test_split.txt")


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def relative_path(value: str, label: str) -> Path:
    """Keep configurable paths inside the repository, independent of the cwd."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty repository-relative path")
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{label} must be relative to the repository root: {value}")
    resolved = (REPO / path).resolve()
    try:
        return resolved.relative_to(REPO)
    except ValueError as exc:
        raise ValueError(f"{label} cannot leave the repository: {value}") from exc


def positive_int(config: dict, field: str) -> int:
    value = config.get(field)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Config field {field} must be a positive integer")
    return value


def load_config(model: str) -> dict:
    config = read_json(REPO / "configs" / f"{model}.json")
    if not isinstance(config, dict) or config.get("model") != model:
        raise ValueError(f"configs/{model}.json must identify model {model}")
    for field in ("epochs", "batch_size", "eval_batch_size"):
        positive_int(config, field)
    if type(config.get("num_workers")) is not int or config["num_workers"] < 0:
        raise ValueError(f"configs/{model}.json has an invalid num_workers value")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError(f"configs/{model}.json must list nonnegative integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"configs/{model}.json contains duplicate seeds")
    for field in ("entrypoint", "weights", "data_root"):
        config[field] = relative_path(config.get(field), field).as_posix()
    if Path(config["entrypoint"]).suffix != ".py":
        raise ValueError(f"configs/{model}.json entrypoint must be a Python script")
    expected = config.get("weights_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError(f"configs/{model}.json must provide a complete weights_sha256")
    return config


def command_for(config: dict, seed: int, data: Path, output: Path, workers: int, resume: bool) -> list[str]:
    model = config["model"]
    command = [sys.executable, "-u", config["entrypoint"]]
    if model == "p2p":
        command += ["--root", ".", "--source-root", "third_party/P2P-Net", "--out", output.as_posix(),
                    "--workers", str(workers)]
    else:
        command += ["--model", model, "--output-dir", output.as_posix(), "--num-workers", str(workers)]
    command += ["--seed", str(seed), "--data-root", data.as_posix(), "--weights", config["weights"],
                "--epochs", str(config["epochs"]), "--batch-size", str(config["batch_size"]),
                "--eval-batch-size", str(config["eval_batch_size"])]
    if resume:
        if model == "p2p":
            command += ["--resume-file", (output / "last.pt").as_posix()]
        else:
            command += ["--resume", "auto"]
    return command


def completed_marker(output: Path, seed: int, epochs: int) -> dict:
    marker = read_json(REPO / output / "completed.json")
    if not isinstance(marker, dict) or marker.get("state") != "completed":
        raise ValueError(f"{output}/completed.json is not a completed run")
    selected = marker.get("selected_epoch", marker.get("epochs", marker.get("epoch_completed")))
    if marker.get("seed") != seed or selected != epochs:
        raise ValueError(f"{output}/completed.json does not match the requested seed and epochs")
    return marker


def validate_assets(config: dict, data: Path) -> None:
    """Check required local files before any subprocess can start training."""
    required = [REPO / config["entrypoint"], REPO / config["weights"]]
    required += [REPO / data / filename for filename in CUB_METADATA]
    if config["model"] == "p2p":
        required += [REPO / "third_party" / "P2P-Net" / filename
                     for filename in ("model.py", "Resnet.py", "anchors.py", "dataset.py", "utils.py", "clustering.py")]
    missing = [path.relative_to(REPO).as_posix() for path in required
               if not path.is_file() or path.stat().st_size == 0]
    if not (REPO / data / "images").is_dir():
        missing.append((data / "images").as_posix() + "/")
    if missing:
        raise FileNotFoundError("Required local assets are missing or empty:\n  " + "\n  ".join(missing))
    digest = hashlib.sha256()
    with (REPO / config["weights"]).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != config["weights_sha256"]:
        raise ValueError(f"Pretrained checkpoint identity differs from the recorded recipe: {config['weights']}")


def make_plan(args) -> list[dict]:
    configs = {model: load_config(model) for model in args.models}
    output_root = relative_path(args.output_root, "output-root")
    data_override = relative_path(args.data_root, "data-root") if args.data_root else None
    jobs = []
    for model, config in configs.items():
        unsupported = sorted(set(args.seeds) - set(config["seeds"]))
        if unsupported:
            raise ValueError(f"Seeds {unsupported} are not listed in configs/{model}.json")
        data = data_override if data_override is not None else Path(config["data_root"])
        for seed in args.seeds:
            output = relative_path((output_root / model / f"seed{seed}").as_posix(), "run output")
            jobs.append({"model": model, "seed": seed, "config": config, "data": data,
                         "output": output, "skip": False, "resume": False})
    if args.dry_run:
        for job in jobs:
            job["command"] = command_for(job["config"], job["seed"], job["data"], job["output"],
                                         args.num_workers, args.resume)
        return jobs
    checked_assets = set()
    for job in jobs:
        output = REPO / job["output"]
        if output.exists() and not output.is_dir():
            raise FileExistsError(f"Run output is not a directory: {job['output']}")
        nonempty = output.is_dir() and next(output.iterdir(), None) is not None
        if nonempty and not args.resume:
            raise FileExistsError(f"Run already contains files: {job['output']}. Use --resume or a new --output-root.")
        if (output / "completed.json").is_file():
            completed_marker(job["output"], job["seed"], job["config"]["epochs"])
            job["skip"] = True
        elif nonempty:
            if not (output / "last.pt").is_file():
                raise FileNotFoundError(f"Cannot resume {job['output']}: last.pt is missing")
            job["resume"] = True
        asset_key = (job["model"], job["data"])
        if not job["skip"] and asset_key not in checked_assets:
            validate_assets(job["config"], job["data"])
            checked_assets.add(asset_key)
        job["command"] = command_for(job["config"], job["seed"], job["data"], job["output"],
                                     args.num_workers, job["resume"])
    return jobs


def run_metrics(job: dict) -> dict:
    output = REPO / job["output"]
    marker = completed_marker(job["output"], job["seed"], job["config"]["epochs"])
    report = read_json(output / "classification_report.json")
    metrics = report.get("aggregate") if isinstance(report, dict) else None
    if not metrics:
        metrics = marker.get("aggregate") or marker.get("metrics")
    if not metrics and (output / "result.json").is_file():
        result = read_json(output / "result.json")
        metrics = result.get("aggregate") or result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"No aggregate classification metrics in {job['output']}")
    if metrics.get("samples") != 5794:
        raise ValueError(f"{job['output']} does not contain the complete official CUB test split")
    benchmark = read_json(output / "benchmark.json")
    if job["model"] == "p2p":
        timing = benchmark["global_only_equivalent"]
        model_ms = timing["classifier_only_mean_ms"]
        preprocessing_ms = timing["preprocess_plus_classifier_mean_ms"]
    else:
        model_ms = benchmark["classifier_only"]["mean_ms"]
        preprocessing_ms = benchmark["preprocess_plus_classifier"]["mean_ms"]
    record = {"seed": job["seed"], "epochs": job["config"]["epochs"], "samples": metrics["samples"],
              **{field: float(metrics[field]) for field in ("top1", "top3", "top5")},
              "inference_ms": float(model_ms), "with_preprocess_ms": float(preprocessing_ms),
              "precision": benchmark.get("precision"), "gpu": benchmark.get("gpu")}
    if any(not math.isfinite(record[field]) for field in METRICS):
        raise ValueError(f"Nonfinite accuracy or timing value in {job['output']}")
    if any(not 0 <= record[field] <= 1 for field in ("top1", "top3", "top5")):
        raise ValueError(f"Accuracy values must be fractions in {job['output']}")
    if job["model"] == "p2p":
        record["native_author_forward_ms"] = benchmark["author_full_forward"]["classifier_only_mean_ms"]
    return record


def write_summary(jobs: list[dict], output_root: str) -> Path:
    models = {}
    for job in jobs:
        models.setdefault(job["model"], []).append(run_metrics(job))
    for model, runs in list(models.items()):
        aggregate = {field: {"mean": statistics.mean(run[field] for run in runs),
                             "sample_std": statistics.stdev(run[field] for run in runs) if len(runs) > 1 else None}
                     for field in METRICS}
        models[model] = {"seeds": [run["seed"] for run in runs], "runs": runs, "aggregate": aggregate}
    summary = {"dataset": "CUB-200-2011", "split": "official train/test", "input_size": 448,
               "selection_policy": "fixed_last_epoch", "accuracy_unit": "fraction",
               "latency_unit": "ms/image", "aggregation": "Independent seeds; mean and sample std (ddof=1); no ensemble.",
               "models": models}
    path = REPO / relative_path(output_root, "output-root") / "summary.json"
    temporary = path.with_name(".summary.json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path.relative_to(REPO)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--models", nargs="+", choices=MODELS, required=True)
    result.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    result.add_argument("--data-root", help="Repository-relative CUB_200_2011 directory")
    result.add_argument("--output-root", default="runs", help="Repository-relative output directory")
    result.add_argument("--num-workers", type=int, default=8)
    result.add_argument("--resume", action="store_true", help="Skip completed runs and resume incomplete runs from last.pt")
    result.add_argument("--dry-run", action="store_true", help="Validate configuration and print commands without checking assets or writing files")
    return result


def main() -> int:
    argument_parser = parser()
    args = argument_parser.parse_args()
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        argument_parser.error("Models and seeds must not contain duplicates")
    if not args.seeds or any(seed < 0 for seed in args.seeds) or args.num_workers < 0:
        argument_parser.error("Seeds and num-workers must be nonnegative")
    try:
        jobs = make_plan(args)
        print(f"Working directory: {REPO}", flush=True)
        for job in jobs:
            label = f"{job['model']} seed {job['seed']}"
            if job["skip"]:
                print(f"Skipping completed run: {label}", flush=True)
                continue
            print(f"{label}: {shlex.join(job['command'])}", flush=True)
            if not args.dry_run:
                subprocess.run(job["command"], cwd=REPO, check=True)
                completed_marker(job["output"], job["seed"], job["config"]["epochs"])
        if not args.dry_run:
            print(f"Results written to {write_summary(jobs, args.output_root).as_posix()}", flush=True)
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Training stopped with exit code {exc.returncode}; no later run was started.", file=sys.stderr)
        return exc.returncode if exc.returncode > 0 else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
