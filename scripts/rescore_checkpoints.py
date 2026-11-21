#!/usr/bin/env python3
"""
Rescore saved checkpoints with the current ColorCLIPTrainer evaluation pipeline.

Computes per-part retrieval metrics (including test Log_Loss), post-training
centroid silhouette (for K <= 797), and tolerance-8 crowd comparison (for K <= 96,
unless --no-crowd is passed), writing <exp>_final_metrics.json and <exp>_per_name.csv
under experiments/<round>/metrics/.

Each checkpoint is first verified against its training epoch row in metrics.csv
on CHECK_KEYS so a mismatched split, vocabulary, or weight file is caught cleanly.
"""
from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import resource
import shutil
import sys
import tarfile
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ML.data_prep import DataLoader
from ML.trainers.color_clip_trainer import ColorCLIPTrainer

CHECK_KEYS = ("r_at_1", "r_at_5", "mrr", "median_rank")


class _Logger:
    """Minimal logger providing log_dir for ColorCLIPTrainer."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir


def checkpoint_epoch(rows: list[dict], config: dict) -> int:
    """Replay _check_early_stopping to recover the saved best checkpoint's epoch."""
    es = config.get("training", {}).get("early_stopping", {})
    if es.get("patience", 0) <= 0:
        return len(rows)
    metric = es.get("metric", "r_at_1")
    min_delta = es.get("min_delta", 0.0)
    best, epoch = -1.0, 0
    for idx, row in enumerate(rows, 1):
        val = float(row[metric])
        if val > best + min_delta:
            best, epoch = val, idx
    return epoch


def _already_current(out_path: str, per_name_path: str, k: int, no_crowd: bool) -> bool:
    if not (os.path.exists(out_path) and os.path.exists(per_name_path)):
        return False
    with open(out_path) as f:
        prev = json.load(f)
    clean_part = prev.get("parts", {}).get("clean", {})
    if "log_loss" not in clean_part:
        return False
    if k <= 797 and "silhouette" not in prev:
        return False
    if (not no_crowd) and k <= 96 and "colour_to_name_diff" not in prev:
        return False
    return True


def _rescore_one(
    round_name: str,
    exp: str,
    force: bool,
    tol: float,
    no_crowd: bool,
    data_cache: dict,
) -> tuple[str, str]:
    models_dir = os.path.join("experiments", round_name, "models", exp)
    metrics_root = os.path.join("experiments", round_name, "metrics")
    os.makedirs(metrics_root, exist_ok=True)
    out_path = os.path.join(metrics_root, f"{exp}_final_metrics.json")
    per_name_path = os.path.join(metrics_root, f"{exp}_per_name.csv")
    csv_path = os.path.join(metrics_root, f"{exp}_metrics.csv")

    if not os.path.exists(os.path.join(models_dir, "color_clip_model.pth")):
        return "skipped", "no final checkpoint"
    if not os.path.exists(csv_path):
        return "skipped", "no metrics.csv"

    best_path = os.path.join(models_dir, "best_color_clip_model.pth")
    ckpt_path = best_path if os.path.exists(best_path) else os.path.join(models_dir, "color_clip_model.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    k = int(config["data"]["top_n_colors"])

    if not force and _already_current(out_path, per_name_path, k, no_crowd):
        return "skipped", "metrics already current"

    prev_json = None
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev_json = json.load(f)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    epoch = checkpoint_epoch(rows, config)

    try:
        data_key = json.dumps([config["data"], config["seed"]], sort_keys=True)
        if data_key not in data_cache:
            data_cache.clear()
            gc.collect()
            bundle = DataLoader(config).load()
            bundle["train_loader"] = None
            bundle["X_train"] = None
            gc.collect()
            data_cache[data_key] = bundle
        with tempfile.TemporaryDirectory() as tmp:
            trainer = ColorCLIPTrainer(config, data_cache[data_key], _Logger(tmp))
            trainer.optimizer = None
            trainer.model.load_state_dict(ckpt["model_state_dict"])
            trainer.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])

            check = trainer.evaluate()
            logged = rows[epoch - 1]
            diffs = {key: abs(float(check[key]) - float(logged[key])) for key in CHECK_KEYS}
            if max(diffs.values()) > tol:
                msg = ", ".join(f"{key} {float(check[key]):.4f} vs {float(logged[key]):.4f}" for key in CHECK_KEYS)
                return "failed", f"epoch {epoch} not reproduced: {msg}"

            parts = trainer.evaluate_parts(
                compute_crowd=False if no_crowd else None,
                include_whole=False,
            )
            trainer.best_epoch = epoch
            trainer._save_final_metrics(parts)
            with open(os.path.join(tmp, "final_metrics.json")) as f:
                out = json.load(f)
            shutil.copy2(os.path.join(tmp, "per_name.csv"), per_name_path)
    except Exception as exc:
        return "failed", f"{type(exc).__name__}: {exc}"

    if prev_json is not None:
        if "last" in prev_json:
            out["last"] = prev_json["last"]
        if "last_beats_best" in prev_json:
            out["last_beats_best"] = prev_json["last_beats_best"]
    out["source"] = "rescored"
    out["check_max_abs_diff"] = max(diffs.values())
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    sil = out.get("silhouette")
    sil_str = f", sil {sil:+.4f}" if sil is not None else ""
    crowd = out.get("colour_to_name_diff")
    crowd_str = f", c2n {crowd:.4f}" if crowd is not None else ""
    return "written", (
        f"epoch {epoch}, R@1 clean {out['r_at_1']:.4f}, "
        f"log_loss {out['log_loss']:.4f}{sil_str}{crowd_str}"
    )


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _target_data_key(rnd: str, exp: str) -> str:
    cfg_path = os.path.join("experiments", rnd, "configs", f"{exp}.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        ckpt_path = os.path.join("experiments", rnd, "models", exp, "best_color_clip_model.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join("experiments", rnd, "models", exp, "color_clip_model.pth")
        cfg = torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]
    return json.dumps([cfg["data"], cfg["seed"]], sort_keys=True)


def _pack_shards(
    targets: list[tuple[str, str]],
    pack_dir: str,
    shard_size: int,
    no_crowd: bool,
    mem_mb: int,
    mem_797_mb: int,
):
    """Pack ML/, script, and sharded checkpoint tarballs for HTCondor submission."""
    os.makedirs(pack_dir, exist_ok=True)
    os.makedirs(os.path.join(pack_dir, "out"), exist_ok=True)

    def _code_filter(info: tarfile.TarInfo):
        name = os.path.basename(info.name)
        if name == "__pycache__" or name == ".DS_Store" or name.startswith("._"):
            return None
        return info

    code_tar = os.path.join(pack_dir, "code.tar.gz")
    with tarfile.open(code_tar, "w:gz") as tar:
        tar.add("ML", filter=_code_filter)
        tar.add(os.path.join("scripts", "rescore_checkpoints.py"), filter=_code_filter)

    by_key = {}
    skipped = 0
    for rnd, exp in targets:
        models_dir = os.path.join("experiments", rnd, "models", exp)
        metrics_csv = os.path.join("experiments", rnd, "metrics", f"{exp}_metrics.csv")
        if not (os.path.exists(os.path.join(models_dir, "color_clip_model.pth")) and os.path.exists(metrics_csv)):
            skipped += 1
            continue
        key = _target_data_key(rnd, exp)
        by_key.setdefault(key, []).append((rnd, exp))

    shards = []
    for key in sorted(by_key):
        items = by_key[key]
        k = json.loads(key)[0].get("top_n_colors") or 0
        req_mem = mem_797_mb if int(k) >= 797 else mem_mb
        for start in range(0, len(items), shard_size):
            shards.append((items[start:start + shard_size], req_mem))

    mode = "nocrowd" if no_crowd else "crowd"
    tasks = []
    for idx, (chunk, req_mem) in enumerate(shards):
        job_id = f"job_{idx:04d}"
        job_tar = os.path.join(pack_dir, f"rescore_{job_id}.tar.gz")
        manifest_bytes = "".join(f"{rnd}, {exp}\n" for rnd, exp in chunk).encode()
        with tarfile.open(job_tar, "w:gz") as tar:
            info = tarfile.TarInfo(name=f"manifest_{job_id}.txt")
            info.size = len(manifest_bytes)
            tar.addfile(info, io.BytesIO(manifest_bytes))
            for rnd, exp in chunk:
                mdir = os.path.join("experiments", rnd, "models", exp)
                best = os.path.join(mdir, "best_color_clip_model.pth")
                final = os.path.join(mdir, "color_clip_model.pth")
                if os.path.exists(best):
                    tar.add(best)
                tar.add(final)
                tar.add(os.path.join("experiments", rnd, "metrics", f"{exp}_metrics.csv"))
                prev_fm = os.path.join("experiments", rnd, "metrics", f"{exp}_final_metrics.json")
                if os.path.exists(prev_fm):
                    tar.add(prev_fm)
        tasks.append(f"{job_id}, {mode}, {req_mem}\n")

    with open(os.path.join(pack_dir, "tasks.txt"), "w") as f:
        f.writelines(tasks)
    packed_runs = sum(len(chunk) for chunk, _ in shards)
    print(f"Packed {packed_runs} runs ({skipped} incomplete skipped) into {len(shards)} jobs at {pack_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Rescore saved checkpoints into final_metrics.json and per_name.csv.")
    parser.add_argument("rounds", nargs="*", help="Rounds to rescore (e.g. 14th 16th)")
    parser.add_argument("--exp", help="Single experiment name inside the round")
    parser.add_argument("--batch", help="File with '<round>, <exp>' lines to process")
    parser.add_argument("--force", action="store_true", help="Recompute even when new metrics already exist")
    parser.add_argument("--no-crowd", action="store_true", help="Skip tolerance-8 crowd scoring")
    parser.add_argument("--tol", type=float, default=1e-3, help="Epoch parity tolerance (default 1e-3)")
    parser.add_argument("--pack-dir", help="Pack code + sharded checkpoints for HTCondor submission")
    parser.add_argument("--shard-size", type=int, default=15, help="Checkpoints per HTCondor job when packing (default 15)")
    parser.add_argument("--mem", type=int, default=1900, help="request_memory MB for K <= 96 when packing")
    parser.add_argument("--mem-large", type=int, default=2400, help="request_memory MB for K >= 797 when packing")
    args = parser.parse_args()

    os.chdir(ROOT)
    targets = []
    if args.batch:
        with open(args.batch) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rnd, exp = [part.strip() for part in line.split(",", 1)]
                targets.append((rnd, exp))
    elif args.exp:
        if len(args.rounds) != 1:
            parser.error("--exp requires exactly one round argument")
        targets.append((args.rounds[0], args.exp))
    elif args.rounds:
        for rnd in args.rounds:
            models_root = os.path.join("experiments", rnd, "models")
            if os.path.isdir(models_root):
                for exp in sorted(os.listdir(models_root)):
                    if os.path.isdir(os.path.join(models_root, exp)):
                        targets.append((rnd, exp))
    else:
        parser.error("provide rounds or --batch")

    if args.pack_dir:
        _pack_shards(
            targets, args.pack_dir, args.shard_size, args.no_crowd, args.mem, args.mem_large,
        )
        return

    counts = {"written": 0, "skipped": 0, "failed": 0}
    data_cache = {}
    for rnd, exp in targets:
        status, msg = _rescore_one(rnd, exp, args.force, args.tol, args.no_crowd, data_cache)
        gc.collect()
        counts[status] += 1
        tag = f"[{rnd}/{exp}]"
        if status == "failed":
            print(f"  FAILED {tag}: {msg}", flush=True)
        elif status == "written":
            print(f"  {tag}: {msg}", flush=True)
        else:
            print(f"  skipped {tag}: {msg}", flush=True)

    print(
        f"Summary: {counts['written']} written, {counts['skipped']} skipped, "
        f"{counts['failed']} failed | peak RSS {_peak_rss_mb():.1f} MB",
        flush=True,
    )
    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
