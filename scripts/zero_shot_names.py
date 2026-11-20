#!/usr/bin/env python3
"""
Task 10 — can the two-tower model name colors it never trained on?

A plain classifier cannot: its output layer has one slot per training class, so an
unseen name scores nothing by construction. A two-tower model can, in principle,
embed any name whose words are in its vocabulary and score colors against it. That
compositional generalization is the one capability the CLIP framing buys us, and it
has never been measured.

Protocol
    Train set  : the top-96 names (an existing checkpoint, no retraining here).
    Unseen set : names ranked 97..797 that are spelled entirely with the 96-name
                 vocabulary, so the BoW tower can embed them. Their samples appear
                 in neither the checkpoint's train nor its test split.
    pure ZSL        : rank the unseen names only  (chance = 1/n_unseen)
    generalized ZSL : rank seen + unseen together (chance = 1/(96+n_unseen);
                      harder, and the realistic setting)

Usage:
    python scripts/zero_shot_names.py --checkpoint <path/to/best_color_clip_model.pth>
"""
import argparse
import os
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from ML.embeddings import BagOfWordsEmbedder, rgb_to_oklch
from ML.models.clip_models import ColorCLIPModel

BATCH = 8192


def recall_at_k(sims, target_col, ks=(1, 5)):
    """sims: (N, C) scores; target_col: index of the true name. Returns {k: R@k}."""
    gt = sims[:, target_col]
    ranks = (sims >= gt[:, None]).sum(1)
    return {k: float((ranks <= k).float().mean()) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seen", type=int, default=96, help="how many names the model was trained on")
    ap.add_argument("--pool", type=int, default=797, help="draw unseen names from the top-<pool> list")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    mcfg = cfg["model"]

    df = pd.read_csv(cfg["data"]["csv_path"], usecols=["r", "g", "b", "colorname"])
    vc = df["colorname"].value_counts()
    seen_names = list(vc.nlargest(args.seen).index)
    pool_names = list(vc.nlargest(args.pool).index)

    # Rebuild the checkpoint's vocabulary exactly as training did: fit on seen names only.
    bow = BagOfWordsEmbedder(top_n_words=(None if cfg["data"].get("vocab_size") == "auto"
                                          else cfg["data"].get("vocab_size")))
    bow.fit(seen_names)

    # An unseen name is usable only if every one of its tokens is in that vocabulary;
    # otherwise its BoW vector drops information and the comparison would be unfair.
    unseen_names = []
    for n in pool_names[args.seen:]:
        toks = bow._tokenize(n)
        if toks and all(t in bow.word_to_idx for t in toks):
            unseen_names.append(n)
    print(f"vocabulary from top-{args.seen} names : {bow.vocab_size} words")
    print(f"unseen names ranked {args.seen+1}-{args.pool} : {len(unseen_names)} of "
          f"{len(pool_names)-args.seen} are fully spellable -> testable zero-shot")

    model = ColorCLIPModel(
        vocab_size=bow.vocab_size,
        embed_dim=mcfg["embed_dim"],
        color_hidden_dims=mcfg.get("color_hidden_dims"),
        text_hidden_dims=mcfg.get("text_hidden_dims"),
        text_encoder_type=mcfg.get("text_encoder", "bow"),
        num_classes=args.seen,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Every response whose label is an unseen name: never used in training or testing.
    d = df[df["colorname"].isin(set(unseen_names))]
    print(f"zero-shot evaluation samples          : {len(d):,}")

    rgb = d[["r", "g", "b"]].values.astype(np.float32)
    colors = (rgb_to_oklch(rgb, normalize=True).astype(np.float32)
              if cfg["data"].get("color_space", "rgb") == "oklch" else rgb / 255.0)
    name_to_col = {n: i for i, n in enumerate(unseen_names)}
    target = torch.tensor([name_to_col[n] for n in d["colorname"]])

    with torch.no_grad():
        proto_unseen = model.encode_text(torch.FloatTensor(bow.transform(unseen_names)))
        proto_seen = model.encode_text(torch.FloatTensor(bow.transform(seen_names)))
        embeds = torch.cat([model.encode_color(torch.from_numpy(colors[i:i + BATCH]))
                            for i in range(0, len(colors), BATCH)])

    # pure ZSL: candidates are the unseen names only
    sims_pure = embeds @ proto_unseen.t()
    gt = sims_pure[torch.arange(len(target)), target]
    ranks_pure = (sims_pure >= gt[:, None]).sum(1)
    # generalized ZSL: seen names compete too (they sit first in the candidate list)
    sims_gen = embeds @ torch.cat([proto_seen, proto_unseen]).t()
    gt_g = sims_gen[torch.arange(len(target)), target + len(seen_names)]
    ranks_gen = (sims_gen >= gt_g[:, None]).sum(1)

    def row(label, ranks, n_cand):
        r1 = float((ranks == 1).float().mean())
        r5 = float((ranks <= 5).float().mean())
        med = float(ranks.float().median())
        print(f"  {label:<22}{n_cand:>10,}{r1:>10.4f}{r5:>10.4f}{med:>12.1f}{1/n_cand:>12.5f}")

    print(f"\n{'protocol':<24}{'candidates':>10}{'R@1':>10}{'R@5':>10}{'med rank':>12}{'chance R@1':>12}")
    row("pure zero-shot", ranks_pure, len(unseen_names))
    row("generalized zero-shot", ranks_gen, len(seen_names) + len(unseen_names))
    print("\n  a plain classifier scores exactly 0.0000 on both, by construction")


if __name__ == "__main__":
    sys.exit(main())
