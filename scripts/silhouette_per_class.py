#!/usr/bin/env python3
"""
Per-class silhouette scores in three spaces: raw RGB, OKLCH, and learned.

Produces grouped bar charts (3 PNGs of ~32 classes each) and a CSV table
showing, for each class, how well-separated it is in each representation.

Usage:
    python scripts/silhouette_per_class.py --checkpoint <path/to/model.pth>
    python scripts/silhouette_per_class.py --checkpoint <...> --max-samples 30000
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import csv
import math
import sys
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import train_test_split

from ML.data_prep import DataLoader
from ML.embeddings import rgb_to_oklch
from ML.models.clip_models import ColorCLIPModel
from ML.utils import get_device

BATCH = 8192
CLASSES_PER_CHART = 32


def encode_in_batches(model, inputs, device):
    """Encode inputs through the color encoder in chunks."""
    chunks = []
    with torch.no_grad():
        for i in range(0, len(inputs), BATCH):
            chunks.append(model.encode_color(inputs[i:i + BATCH].to(device)).cpu())
    return torch.cat(chunks).numpy()


def plot_chunk(order_chunk, class_names, mean_rgb, mean_oklch, mean_learned,
               delta, n_samples, k, part, n_parts, out_path):
    """Plot one chunk of ~32 classes as a triple-bar chart."""
    n = len(order_chunk)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.35), 5.5))
    x = np.arange(n)
    w = 0.27

    ax.bar(x - w, mean_rgb[order_chunk], w, label="RGB", color="#8899aa", zorder=2)
    ax.bar(x,     mean_oklch[order_chunk], w, label="OKLCH", color="#5b9bd5", zorder=2)
    colors_l = ["#c0392b" if delta[i] < 0 else "#27ae60" for i in order_chunk]
    ax.bar(x + w, mean_learned[order_chunk], w, color=colors_l, zorder=2)

    # legend with both learned colors explained
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor="#27ae60"), Patch(facecolor="#c0392b")]
    labels  += ["Learned (improved)", "Learned (worsened)"]
    ax.legend(handles, labels, loc="upper left", fontsize=7)

    # annotate worsened classes
    for j, i in enumerate(order_chunk):
        if delta[i] < 0:
            ax.annotate(class_names[i], (j + w, mean_learned[i]),
                        textcoords="offset points", xytext=(0, -12), ha="center",
                        fontsize=6, color="#c0392b", fontweight="bold", rotation=90)

    ax.axhline(0, color="black", linewidth=0.5, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([class_names[i] for i in order_chunk], rotation=90, fontsize=7)
    ax.set_ylabel("Mean silhouette score")
    ax.set_title(f"Per-class silhouette ({n_samples:,} samples, {k} classes) — "
                 f"part {part}/{n_parts}", fontsize=10)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Per-class silhouette: RGB vs OKLCH vs learned")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved *_color_clip_model.pth")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Stratified subsample cap (silhouette is O(N²); default 50k)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for charts and CSV")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    print("Loading data bundle...")
    bundle = DataLoader(config).load()
    label_encoder = bundle["label_encoder"]
    class_names = list(label_encoder.classes_)
    k = len(class_names)

    # --- rebuild model ---
    model_cfg = config["model"]
    model = ColorCLIPModel(
        vocab_size=bundle["vocab_size"],
        embed_dim=model_cfg["embed_dim"],
        color_hidden_dims=model_cfg.get("color_hidden_dims"),
        text_hidden_dims=model_cfg.get("text_hidden_dims"),
        text_encoder_type=model_cfg.get("text_encoder", "bow"),
        num_classes=k,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- prepare test data ---
    X_test = bundle["X_test"]      # (N, 3) normalized RGB [0, 1]
    y_test = bundle["y_test"]      # (N,)

    # stratified subsample if needed
    if len(X_test) > args.max_samples:
        print(f"Subsampling {len(X_test):,} → {args.max_samples:,} (stratified)...")
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=args.max_samples,
            stratify=y_test, random_state=42)

    # three representations
    X_rgb = X_test.astype(np.float64)
    X_oklch = rgb_to_oklch(X_test * 255.0, normalize=True).astype(np.float64)

    model_input = torch.FloatTensor(X_oklch.astype(np.float32))
    print(f"Encoding {len(X_test):,} samples...")
    X_embed = encode_in_batches(model, model_input, device).astype(np.float64)

    # --- silhouette ---
    print("Computing silhouette scores (RGB)...")
    sil_rgb = silhouette_samples(X_rgb, y_test)
    print("Computing silhouette scores (OKLCH)...")
    sil_oklch = silhouette_samples(X_oklch, y_test)
    print("Computing silhouette scores (learned)...")
    sil_learned = silhouette_samples(X_embed, y_test)

    # per-class means
    classes = np.arange(k)
    counts = np.bincount(y_test, minlength=k)
    mean_rgb = np.array([sil_rgb[y_test == c].mean() if counts[c] > 0 else 0.0 for c in classes])
    mean_oklch = np.array([sil_oklch[y_test == c].mean() if counts[c] > 0 else 0.0 for c in classes])
    mean_learned = np.array([sil_learned[y_test == c].mean() if counts[c] > 0 else 0.0 for c in classes])
    delta = mean_learned - mean_rgb

    # sort by RGB silhouette (worst first)
    order = np.argsort(mean_rgb)

    # --- output ---
    os.makedirs(args.out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(args.out_dir, "silhouette_per_class.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "count", "silhouette_rgb", "silhouette_oklch",
                     "silhouette_learned", "delta_rgb_learned"])
        for i in order:
            w.writerow([class_names[i], int(counts[i]),
                        f"{mean_rgb[i]:.4f}", f"{mean_oklch[i]:.4f}",
                        f"{mean_learned[i]:.4f}", f"{delta[i]:.4f}"])
    print(f"Wrote {csv_path}")

    # Bar charts — split into chunks of CLASSES_PER_CHART
    n_parts = math.ceil(k / CLASSES_PER_CHART)
    for p in range(n_parts):
        chunk = order[p * CLASSES_PER_CHART : (p + 1) * CLASSES_PER_CHART]
        png_path = os.path.join(args.out_dir, f"silhouette_per_class_{p + 1}.png")
        plot_chunk(chunk, class_names, mean_rgb, mean_oklch, mean_learned,
                   delta, len(X_test), k, p + 1, n_parts, png_path)
        print(f"Wrote {png_path}")

    # summary
    n_improved = (delta > 0).sum()
    n_worsened = (delta < 0).sum()
    print(f"\nSummary: {n_improved}/{k} improved, {n_worsened}/{k} worsened, "
          f"mean delta = {delta.mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
