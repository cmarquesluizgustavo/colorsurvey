#!/usr/bin/env python3
"""
Build script for the Plurality Oracle annex.

The oracle is a lookup table with no learning: cut the RGB cube into equal boxes, and in
each box answer with the name people gave most often there (its plurality). It is the
best possible classifier *for that box size* — no rule that only sees which box a colour
falls in can be more accurate on average. It is not a ceiling in general: a different
box size, or a model that draws smoother boundaries, can do better.

Built on the TRAIN split, scored on the TEST split (test_size 0.2, seed 42). Boxes with
no training sample answer with the overall most common name.

Silhouette uses the same definition as annexes/geometry: centroids from the full test
set, scored on a stratified 60k sample. The oracle's "embedding" of a sample is the
one-hot vector of the name it answers, so its silhouette is computed exactly in closed
form (distance from one-hot e_j to centroid P_c is sqrt(1 - 2 P_c[j] + |P_c|^2)).

Outputs:
  - oracle.csv            accuracy, names that own territory, silhouette — per box size
  - name_shares_96c.csv   per name: share of samples vs share of oracle answers (96c)
  - name_shares_96c.png
  - README.md
"""
import os
import sys
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ML.embeddings import BagOfWordsEmbedder
from ML.models.clip_models import ColorCLIPModel

COLOURS = (14, 96, 797, 5363)
BOXES = (8, 16, 32, 64)            # boxes per RGB axis
SUB, CHUNK = 60_000, 15_000
MODEL = "experiments/10th/models/clip_{k}c_proto_dim64_ch16_th256/best_color_clip_model.pth"
SURFACE, BAR, INK, MUTED, GRID = "#fcfcfb", "#2a78d6", "#0b0b0b", "#6b6a66", "#e6e5e0"


def load(k):
    df = pd.read_csv(ROOT / "mainsurvey.csv", usecols=["r", "g", "b", "colorname"])
    d = df[df.colorname.isin(df.colorname.value_counts().nlargest(k).index)]
    X = d[["r", "g", "b"]].values.astype(np.int64)
    le = LabelEncoder()
    y = le.fit_transform(d.colorname.values)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    idx, _ = train_test_split(np.arange(len(Xte)), train_size=SUB, stratify=yte, random_state=42)
    return Xtr, Xte, ytr, yte, idx, list(le.classes_)


def box_id(X, boxes):
    q = np.minimum(X * boxes // 256, boxes - 1)
    return (q[:, 0] * boxes + q[:, 1]) * boxes + q[:, 2]


def oracle(Xtr, ytr, Xte, K, boxes):
    """Answer for every test sample, and how many names win at least one box."""
    counts = pd.DataFrame({"box": box_id(Xtr, boxes), "name": ytr}).value_counts().reset_index(name="n")
    winners = counts.sort_values(["box", "n", "name"], ascending=[True, False, True]).drop_duplicates("box")
    answer = np.full(boxes ** 3, np.bincount(ytr, minlength=K).argmax())   # empty boxes
    answer[winners.box.values] = winners.name.values
    return answer[box_id(Xte, boxes)], winners.name.nunique()


def per_class_mean(v, y, K):
    w = np.bincount(y, minlength=K).astype(float)
    pc = np.bincount(y, weights=v, minlength=K) / np.maximum(w, 1)
    return float((pc * w).sum() / w.sum()), float(pc[w > 0].mean())


def onehot_silhouette(pred, y, idx, K):
    """Exact silhouette of one-hot answers, centroids from all of `y`, scored on `idx`."""
    P = np.zeros((K, K), dtype=np.float32)
    np.add.at(P, (y, pred), np.float32(1.0))
    P /= np.maximum(P.sum(1, keepdims=True), 1)
    S = (P ** 2).sum(1)
    used = np.unique(pred[idx])
    D = np.sqrt(np.maximum(1 - 2 * P[:, used] + S[:, None], 0))       # (K classes, used answers)
    col = np.searchsorted(used, pred[idx])
    truth = y[idx]
    a = D[truth, col]
    order = np.argsort(D, axis=0)[:2]                                  # nearest two centroids per answer
    first, second = order[0, col], order[1, col]
    b = np.where(first == truth, D[second, col], D[first, col])
    return (b - a) / np.maximum(np.maximum(a, b), 1e-12)


def euclid_silhouette(E, y, idx, K):
    E = torch.as_tensor(np.ascontiguousarray(E), dtype=torch.float32)
    yt = torch.as_tensor(y)
    C = torch.zeros(K, E.shape[1]).index_add_(0, yt, E) / \
        torch.bincount(yt, minlength=K).clamp(min=1).unsqueeze(1).float()
    T, ys = E[idx], yt[idx]
    out = torch.empty(len(T))
    for i in range(0, len(T), CHUNK):
        d = torch.cdist(T[i:i + CHUNK], C)
        lb = ys[i:i + CHUNK]
        r = torch.arange(len(lb))
        a = d[r, lb].clone()
        d[r, lb] = float("inf")
        b = d.min(1).values
        out[i:i + CHUNK] = (b - a) / torch.maximum(a, b)
    return out.numpy()


def model_scores(k, Xte, yte, idx, names):
    ck = torch.load(ROOT / MODEL.format(k=k), map_location="cpu", weights_only=False)
    m = ck["config"]["model"]
    bow = BagOfWordsEmbedder(top_n_words=None)
    bow.fit(names)
    model = ColorCLIPModel(vocab_size=bow.vocab_size, embed_dim=m["embed_dim"],
                           color_hidden_dims=m.get("color_hidden_dims"),
                           text_hidden_dims=m.get("text_hidden_dims"), num_classes=len(names))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    X = torch.from_numpy((Xte / 255.0).astype(np.float32))
    with torch.no_grad():
        E = torch.cat([model.encode_color(X[i:i + 32768]) for i in range(0, len(X), 32768)])
        protos = model.encode_text(torch.FloatTensor(bow.transform(names)))
    pred = torch.cat([(E[i:i + CHUNK] @ protos.t()).argmax(1) for i in range(0, len(E), CHUNK)]).numpy()
    return pred, E.numpy()


def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.astype(object).to_dict("records"):
        out.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    os.chdir(ROOT)
    rows, refs, parts, share = [], [], [], None
    for k in COLOURS:
        print(f"{k} names...", flush=True)
        Xtr, Xte, ytr, yte, idx, names = load(k)
        K = len(names)
        s_rgb = euclid_silhouette(Xte / 255.0, yte, idx, K)
        rgb_ovr, rgb_cls = per_class_mean(s_rgb, yte[idx], K)
        m_pred, m_emb = model_scores(k, Xte, yte, idx, names)
        s_mod = euclid_silhouette(m_emb, yte, idx, K)
        m_ovr, m_cls = per_class_mean(s_mod, yte[idx], K)
        refs.append(dict(names=k, rgb_sil=rgb_ovr, rgb_sil_cls=rgb_cls,
                         model_r1=(m_pred == yte).mean(), model_r1_cls=per_class_mean((m_pred == yte).astype(float), yte, K)[1],
                         model_named=len(np.unique(m_pred)), model_sil=m_ovr, model_sil_cls=m_cls))
        for boxes in BOXES:
            pred, owners = oracle(Xtr, ytr, Xte, K, boxes)
            hit = (pred == yte).astype(float)
            s_ovr, s_cls = per_class_mean(onehot_silhouette(pred, yte, idx, K), yte[idx], K)
            rows.append(dict(names=k, boxes=boxes, box_width=round(256 / boxes, 1),
                             r1=hit.mean(), r1_cls=per_class_mean(hit, yte, K)[1],
                             own_territory=owners, ever_answered=len(np.unique(pred)),
                             sil=s_ovr, sil_cls=s_cls))
        kb = max((r for r in rows if r["names"] == k), key=lambda r: r["r1"])["boxes"]
        pred, _ = oracle(Xtr, ytr, Xte, K, kb)
        s_or = onehot_silhouette(pred, yte, idx, K)
        y, p = yte[idx], pred[idx]
        said = np.isin(y, np.unique(pred))
        for group, m in (("right answer", said & (p == y)), ("wrong answer", said & (p != y)),
                         ("name never answered", ~said)):
            parts.append(dict(names=k, group=group, share=m.mean(), oracle=s_or[m].mean(),
                              model=s_mod[m].mean(), rgb=s_rgb[m].mean()))
        if k == 96:
            best_boxes = kb
            share = pd.DataFrame(dict(name=names,
                                      sample_share=np.bincount(yte, minlength=K) / len(yte),
                                      oracle_share=np.bincount(pred, minlength=K) / len(pred),
                                      model_share=np.bincount(m_pred, minlength=K) / len(m_pred)))
            share = share.sort_values("sample_share", ascending=False)
            share.to_csv(HERE / "name_shares_96c.csv", index=False, float_format="%.5f")

    res = pd.DataFrame(rows)
    ref = pd.DataFrame(refs)
    res.to_csv(HERE / "oracle.csv", index=False, float_format="%.4f")

    # chart: how often each name is answered vs how often it is the true label (96 names)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    floor = 0.25 * share.oracle_share[share.oracle_share > 0].min()   # band for exact zeros
    x = share.sample_share.values
    y = np.where(share.oracle_share.values > 0, share.oracle_share.values, floor)
    ax.axhspan(floor / 1.6, floor * 1.6, color=GRID, zorder=0)
    ax.plot([x.min() / 1.2, x.max() * 1.5], [x.min() / 1.2, x.max() * 1.5], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(x, y, s=22, color=BAR, edgecolor=SURFACE, linewidth=1, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(floor / 2.5, 1)
    never = int((share.oracle_share == 0).sum())
    ax.text(x.min(), floor * 2.1, f"never answered: {never} of 96 names", fontsize=8, color=INK)
    top = share.iloc[0]
    ax.annotate(f'{top["name"]} (most common)', (top.sample_share, top.oracle_share), xytext=(-8, 6),
                textcoords="offset points", fontsize=7, color=INK, ha="right")
    ax.set_xlabel("share of test samples with this name", fontsize=8, color=MUTED)
    ax.set_ylabel("share of oracle answers that are this name", fontsize=8, color=MUTED)
    ax.set_title(f"96 names, {best_boxes} boxes per axis", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "name_shares_96c.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # README
    t = res.merge(ref[["names", "model_r1", "model_r1_cls"]], on="names")
    acc = t[["names", "boxes", "box_width", "r1", "r1_cls", "own_territory", "ever_answered"]].rename(columns={
        "box_width": "box width (RGB units)", "r1": "R@1", "r1_cls": "R@1 per name",
        "own_territory": "names that win a box", "ever_answered": "names ever answered"})
    model_acc = ref[["names", "model_r1", "model_r1_cls", "model_named"]].rename(columns={
        "model_r1": "R@1", "model_r1_cls": "R@1 per name", "model_named": "names ever answered"})
    best = res.loc[res.groupby("names").r1.idxmax()]
    sil = best[["names", "boxes", "sil", "sil_cls"]].merge(
        ref[["names", "rgb_sil", "rgb_sil_cls", "model_sil", "model_sil_cls"]], on="names")
    sil = sil[["names", "boxes", "rgb_sil", "model_sil", "sil", "rgb_sil_cls", "model_sil_cls", "sil_cls"]].rename(
        columns={"boxes": "oracle boxes", "rgb_sil": "raw RGB", "model_sil": "model", "sil": "oracle",
                 "rgb_sil_cls": "raw RGB (per name)", "model_sil_cls": "model (per name)",
                 "sil_cls": "oracle (per name)"})
    f = {c: "{:.4f}" for c in ("R@1", "R@1 per name")}
    f.update({c: "{:+.3f}" for c in ("raw RGB", "model", "oracle", "raw RGB (per name)",
                                     "model (per name)", "oracle (per name)")})
    f["box width (RGB units)"] = "{:g}"
    b96 = best.set_index("names").loc[96]
    r96 = ref.set_index("names").loc[96]

    readme = f"""# Plurality Oracle: the best lookup table, and what its silhouette means

A classifier with no learning. Cut the RGB cube into equal boxes; in each box, answer with
the name people gave most often there. Built on the training split, scored on the test
split. Regenerate with `python annexes/plurality_oracle/build.py`.

**The short answer.** For a given box size this is the most accurate answer any rule
based on that box can give, and our model matches it. Above 14 names it only ever says a
minority of the names — the rest are outnumbered in every box. Its silhouette is not a
geometry benchmark: placing each sample exactly on its answer makes right answers look
perfectly tight and wrong answers maximally misplaced, so the score mostly restates
accuracy. At 14 names that makes it look better than any model; above that, worse.

## Parameters

| | |
|---|---|
| split | test_size 0.2, stratified, seed 42 — the same as every experiment |
| box sizes | {", ".join(f"{b}³" for b in BOXES)} boxes ({", ".join(f"{256 / b:g}" for b in BOXES)} RGB units wide) |
| empty boxes | answer with the overall most common name |
| reference model | prototype, dim64, colour tower [16], text tower [256] (10th round) |
| silhouette | centroids from the full test set, scored on a stratified 60k sample |

## 1. Accuracy

{md(acc, f)}

The model, for comparison (full test set):

{md(model_acc, f)}

At its best box size (16 per axis) the lookup table is within 0.005 of the model at every
colour count — the "we are at the ceiling" result of research_plan.md note 1. Smaller boxes let
more names win somewhere but have fewer training samples per box, and accuracy falls.
"Best for a box size" is not a ceiling in general.

## 2. Most names are never answered

![](name_shares_96c.png)

Each dot is one of the 96 names: how often it is the true label (x) against how often the
oracle answers it (y). Common names sit above the dashed line — the oracle says them more
often than they occur — and {int((share.oracle_share == 0).sum())} names sit on the band at the
bottom: the oracle never says them, because in every box some other name is more common.
That is note 3 ("the tail is unreachable by argmax"). Full list: `name_shares_96c.csv`.

## 3. Silhouette: it mostly restates accuracy

{md(sil, f)}

The oracle's silhouette is better than the model's at 14 names and worse above that.
The split by sample type (best box size) shows why:

{md(pd.DataFrame(parts).rename(columns={"share": "share of samples", "oracle": "oracle",
                                         "model": "model", "rgb": "raw RGB"}),
    {"share of samples": "{:.1%}", "oracle": "{:+.3f}", "model": "{:+.3f}", "raw RGB": "{:+.3f}"})}

Each sample is placed exactly on its answer. A right answer sits on top of its own group
(very high silhouette); a wrong answer sits on top of another group (very low). The model
keeps each colour's position, so it is less extreme in both directions. The oracle's
overall score is therefore set by how many answers are right: 76% at 14 names (it wins),
under half at 96 names with a quarter of samples belonging to names it never says (it
loses). The answer vectors are also all equally far apart, so the oracle has no notion
that "light blue" is close to "blue". Use this as a warning about the metric, not as a
geometry target: a hard classifier's silhouette tracks its accuracy.

## Notes

- The oracle's embedding is the one-hot vector of its answer; its silhouette is computed
  exactly, not sampled from a model.
- Ties within a box go to the lower name index (`argmax`).
- An earlier version of this analysis built the boxes on the test set itself; this one
  uses the training split, so it is a fair comparison with the model.
"""
    (HERE / "README.md").write_text(readme)
    print("Done.")


if __name__ == "__main__":
    main()
