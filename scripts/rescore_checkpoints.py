#!/usr/bin/env python3
"""
Write final_metrics.json for runs that finished before the trainer wrote the current version.

The trainer ends by scoring the restored best checkpoint on each part of its test set
(data_prep.test_parts, research_plan.md note 10): the first max_eval_samples rows that
chose the epoch, halves A and B of the rest, and A + B. It saves that to
final_metrics.json, which collect_results.py reports. This script gives
older runs the same numbers: it rebuilds each run's data with the run's own config (same
split, same seed), loads its best checkpoint, and scores it with the trainer's own
evaluate_parts().

Each run is checked before anything is written. The checkpoint is first scored on the
first max_eval_samples test samples, exactly as during training, and must reproduce the
metrics.csv row of the epoch it was saved at (the last epoch whose early-stopping metric
beat the running best by more than min_delta). A match proves the split, the vocabulary
and the weights are the ones the run had; a run that fails is reported and skipped.

Output: experiments/<round>/metrics/<exp>_final_metrics.json, with "source": "rescored".
collect_results.py then reports it with Metrics_Source = rescored. Rebuild the CSV with
    python3 cluster/collect_results.py <round> --local --type clip --sort Colors,Loss_Type,Embed_Dim

Skipped: runs with no final checkpoint (crashed or still running; collect_results marks
them incomplete), and runs whose final_metrics.json already has the per-part scores
(unless --force). Older final_metrics.json files are replaced.

Usage:
    .venv/bin/python scripts/rescore_checkpoints.py 10th 12th [--force] [--tol 1e-3]
"""
import argparse
import csv
import json
import os
import sys
import tempfile

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML.data_prep import DataLoader
from ML.trainers.color_clip_trainer import ColorCLIPTrainer

# Metrics compared against the metrics.csv row of the checkpoint's epoch.
CHECK_KEYS = ("r_at_1", "r_at_5", "mrr", "median_rank")


class _Logger:
    """The trainer only needs log_dir (for its models/ dir and final_metrics.json)."""

    def __init__(self, log_dir):
        self.log_dir = log_dir


def checkpoint_epoch(rows, config):
    """Epoch the run's best checkpoint was saved at, replaying _check_early_stopping.

    With early stopping off there is no best checkpoint and the final one is the last
    epoch.
    """
    es = config["training"].get("early_stopping", {})
    if es.get("patience", 0) <= 0:
        return len(rows)
    metric, min_delta = es.get("metric", "r_at_1"), es.get("min_delta", 0.0)
    best, epoch = -1.0, 0
    for i, row in enumerate(rows, 1):
        val = float(row[metric])
        if val > best + min_delta:
            best, epoch = val, i
    return epoch


def rescore_round(round_name, force, tol, data_cache):
    models_root = os.path.join("experiments", round_name, "models")
    metrics_root = os.path.join("experiments", round_name, "metrics")
    results = {"written": [], "skipped": [], "failed": []}

    for exp in sorted(os.listdir(models_root)):
        models_dir = os.path.join(models_root, exp)
        out_path = os.path.join(metrics_root, f"{exp}_final_metrics.json")
        csv_path = os.path.join(metrics_root, f"{exp}_metrics.csv")
        if not os.path.exists(os.path.join(models_dir, "color_clip_model.pth")):
            results["skipped"].append((exp, "no final checkpoint"))
            continue
        if os.path.exists(out_path) and not force:
            with open(out_path) as f:
                if "parts" in json.load(f):
                    results["skipped"].append((exp, "final_metrics.json is current"))
                    continue
        if not os.path.exists(csv_path):
            results["skipped"].append((exp, "no metrics.csv"))
            continue

        best_path = os.path.join(models_dir, "best_color_clip_model.pth")
        ckpt_path = best_path if os.path.exists(best_path) else os.path.join(models_dir, "color_clip_model.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        epoch = checkpoint_epoch(rows, config)

        try:
            data_key = json.dumps([config["data"], config["seed"]], sort_keys=True)
            if data_key not in data_cache:
                data_cache.clear()  # one split in memory at a time
                data_cache[data_key] = DataLoader(config).load()
            with tempfile.TemporaryDirectory() as tmp:
                trainer = ColorCLIPTrainer(config, data_cache[data_key], _Logger(tmp))
                trainer.model.load_state_dict(ckpt["model_state_dict"])
                trainer.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])

                check = trainer.evaluate()
                logged = rows[epoch - 1]
                diffs = {k: abs(float(check[k]) - float(logged[k])) for k in CHECK_KEYS}
                if max(diffs.values()) > tol:
                    results["failed"].append((exp, f"epoch {epoch} not reproduced: " + ", ".join(
                        f"{k} {float(check[k]):.4f} vs {float(logged[k]):.4f}" for k in CHECK_KEYS)))
                    continue

                parts = trainer.evaluate_parts()
                trainer.best_epoch = epoch
                trainer._save_final_metrics(parts)
                with open(os.path.join(tmp, "final_metrics.json")) as f:
                    out = json.load(f)
        except Exception as e:  # report and move on; one bad run should not stop a round
            results["failed"].append((exp, f"{type(e).__name__}: {e}"))
            continue

        out["source"] = "rescored"
        out["check_max_abs_diff"] = max(diffs.values())
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        sel = out["parts"]["select"]["r_at_1"]
        results["written"].append((exp, f"epoch {epoch}, R@1 clean {out['r_at_1']:.4f} "
                                        f"(A {out['parts']['a']['r_at_1']:.4f}, "
                                        f"B {out['parts']['b']['r_at_1']:.4f}, select {sel:.4f})"))
        print(f"  {exp}: {results['written'][-1][1]}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("rounds", nargs="+")
    parser.add_argument("--force", action="store_true", help="overwrite existing final_metrics.json")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="largest accepted |difference| from the logged epoch (default 1e-3)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_cache = {}
    for round_name in args.rounds:
        print(f"=== {round_name}", flush=True)
        results = rescore_round(round_name, args.force, args.tol, data_cache)
        for exp, why in results["failed"]:
            print(f"  FAILED {exp}: {why}")
        for exp, why in results["skipped"]:
            print(f"  skipped {exp}: {why}")
        print(f"  {round_name}: {len(results['written'])} written, "
              f"{len(results['skipped'])} skipped, {len(results['failed'])} failed", flush=True)


if __name__ == "__main__":
    main()
