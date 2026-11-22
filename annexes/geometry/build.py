#!/usr/bin/env python3
"""
Thin consumer build script for the Geometry annex: how well colour names separate
in raw colour spaces and in the learned embedding, and which settings change that.

Per-model Silhouette, Silhouette_Per_Class, R_at_1, and Class_Oriented_R_at_1 are
read directly from experiments/*/experiment_results.csv (and per-name silhouettes from
experiments/*/metrics/*_per_name.csv), paired with hyperparameters from
experiments/*/configs/*.json. Only raw colour spaces, the untrained network control,
and the 96c method checks (exact vs simplified silhouette, sample sizes) are computed
directly using ML.metrics.post_training.
"""
import functools
import json
import os
import re
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
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ML.data_prep import test_parts
from ML.embeddings import BagOfWordsEmbedder, rgb_to_oklch
from ML.metrics.post_training import (
    compute_centroids,
    compute_silhouette_metrics,
    simplified_silhouette_samples,
)
from ML.models.clip_models import ColorCLIPModel

COLOURS = (14, 96, 797)  # 5363 left out: too many names with one or no clean row
SUB, CHUNK = 60_000, 15_000
REF_ROUND = "10th"
REF_EXP = "clip_96c_proto_dim64_ch16_th256"
REF_MODEL = f"experiments/{REF_ROUND}/models/{REF_EXP}/best_color_clip_model.pth"
TOWER_ORDER = ["0", "4", "8", "16", "32", "8x32", "64", "128", "32x128"]
LOSS_INV = {"prototype": "proto", "masked": "mask", "original": "orig", "supcon": "supc"}
SURFACE, BAR, INK, MUTED, GRID = "#fcfcfb", "#2a78d6", "#0b0b0b", "#6b6a66", "#e6e5e0"


# ---------------------------------------------------------------- data and metric

@functools.lru_cache(maxsize=None)
def _survey():
    return pd.read_csv(ROOT / "mainsurvey.csv", usecols=["r", "g", "b", "colorname"])


@functools.lru_cache(maxsize=None)
def load_split(k, seed=42, n_select=5000):
    """The clean test rows (halves A + B, research_plan.md note 10) of a run's split."""
    df = _survey()
    d = df[df.colorname.isin(df.colorname.value_counts().nlargest(k).index)]
    X = d[["r", "g", "b"]].values.astype(np.float32) / 255.0
    le = LabelEncoder()
    y = le.fit_transform(d.colorname.values)
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    parts = test_parts(yte, n_select)
    clean = np.sort(np.concatenate([parts["a"], parts["b"]]))
    Xte, yte = Xte[clean], yte[clean]
    idx, _ = train_test_split(np.arange(len(Xte)), train_size=SUB, stratify=yte, random_state=42)
    return Xte, yte, idx, list(le.classes_)


def score_space(V, yte, idx, K):
    """Compute overall and per-class silhouette using shared post_training helpers."""
    V_clean = torch.as_tensor(np.ascontiguousarray(V), dtype=torch.float32)
    C = compute_centroids(V_clean, yte, K)
    s = simplified_silhouette_samples(V_clean[idx], yte[idx], C, chunk_size=CHUNK)
    y_sub = yte[idx]
    w = np.bincount(y_sub, minlength=K).astype(float)
    present = w > 0
    pc = np.bincount(y_sub, weights=s, minlength=K) / np.maximum(w, 1.0)
    return float((pc * w).sum() / w.sum()), float(pc[present].mean())


def load_model(path, names):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["config"]["model"]
    bow = BagOfWordsEmbedder(top_n_words=None)
    bow.fit(names)
    model = ColorCLIPModel(
        vocab_size=bow.vocab_size,
        embed_dim=m["embed_dim"],
        color_hidden_dims=m.get("color_hidden_dims"),
        text_hidden_dims=m.get("text_hidden_dims"),
        text_encoder_type=m.get("text_encoder", "bow"),
        num_classes=len(names),
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck["config"], torch.FloatTensor(bow.transform(names))


def encode(model, X):
    with torch.no_grad():
        return torch.cat([
            model.encode_color(torch.from_numpy(np.ascontiguousarray(X[i:i + 32768])))
            for i in range(0, len(X), 32768)
        ])


# ---------------------------------------------------------------- colour spaces

_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                 [0.0193339, 0.1191920, 0.9503041]])
_LMS = np.array([[0.8189330101, 0.3618667424, -0.1288597137], [0.0329845436, 0.9293118715, 0.0361456387],
                 [0.0482003018, 0.2643662691, 0.6338517070]])
_OKL = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                 [0.0259040371, 0.7827717662, -0.8086757660]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def _linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _xyz(x):
    return _linear(x) @ _XYZ.T


def _lab(x):
    t = _xyz(x) / _WHITE
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _ycbcr(x):
    Y = x @ np.array([0.299, 0.587, 0.114])
    return np.stack([Y, 0.564 * (x[..., 2] - Y), 0.713 * (x[..., 0] - Y)], -1)


SPACES = {
    "RGB": lambda x: x,
    "YCbCr": _ycbcr,
    "linear RGB": _linear,
    "XYZ": _xyz,
    "CIELAB": _lab,
    "OKLab": lambda x: np.cbrt(_xyz(x) @ _LMS.T) @ _OKL.T,
    "OKLCH (hue stored as 0–1)": lambda x: rgb_to_oklch(x * 255.0, normalize=True),
}


# ---------------------------------------------------------------- read results from main CSVs

def _round_sort_key(round_name: str):
    m = re.match(r"(\d+)", round_name)
    return (0, int(m.group(1)), round_name) if m else (1, 0, round_name)


def load_results_table() -> pd.DataFrame:
    """Assemble per-run geometry and R@1 table directly from experiment_results.csv."""
    rows = []
    exp_root = ROOT / "experiments"
    round_dirs = sorted(
        (p for p in exp_root.iterdir() if p.is_dir() and (p / "experiment_results.csv").exists()),
        key=lambda p: _round_sort_key(p.name),
    )
    for rdir in round_dirs:
        rnd = rdir.name
        m_num = re.match(r"(\d+)", rnd)
        if m_num and int(m_num.group(1)) < 7:
            continue
        df = pd.read_csv(rdir / "experiment_results.csv")
        if "Silhouette" not in df.columns or "Silhouette_Per_Class" not in df.columns:
            continue
        for _, r in df.iterrows():
            if str(r.get("Status", "completed")) != "completed":
                continue
            if pd.isna(r.get("Silhouette")) or pd.isna(r.get("Silhouette_Per_Class")):
                continue
            exp = str(r["Experiment"])
            if re.search(r"_no(3|14|96)_", exp):
                continue
            k = int(r["Colors"])
            if k not in COLOURS:
                continue
            cfg_path = rdir / "configs" / f"{exp}.json"
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                cfg = json.load(f)
            if cfg["data"].get("color_space", "rgb") != "rgb" or cfg["data"].get("exclude_top_n", 0) != 0:
                continue
            if int(cfg["data"].get("top_n_colors", k)) != k:
                continue

            m, t = cfg["model"], cfg.get("training", {})
            loss_short = LOSS_INV.get(str(r.get("Loss_Type", "")).lower())
            if not loss_short:
                lm = re.search(r"_(proto|mask|orig|supc)_", exp)
                if not lm:
                    continue
                loss_short = lm.group(1)
            ckpt = ("best_color_clip_model.pth"
                    if (ROOT / "experiments" / rnd / "models" / exp / "best_color_clip_model.pth").exists()
                    else "color_clip_model.pth")
            rows.append(dict(
                round=rnd,
                colors=k,
                exp=exp,
                path=f"experiments/{rnd}/models/{exp}/{ckpt}",
                loss=loss_short,
                dim=m["embed_dim"],
                ch="x".join(map(str, m.get("color_hidden_dims") or [])) or "0",
                th="x".join(map(str, m.get("text_hidden_dims") or [])) or "0",
                lr=t.get("lr", 3e-4),
                temp=t.get("temperature", 0.07),
                t2c=t.get("t2c_weight", 1.0),
                eps=t.get("t2c_eps", 0),
                bs=cfg["data"].get("batch_size"),
                epochs=t.get("epochs"),
                patience=(t.get("early_stopping") or {}).get("patience"),
                seed=cfg.get("seed"),
                sil_ovr=float(r["Silhouette"]),
                sil_cls=float(r["Silhouette_Per_Class"]),
                r1_ovr=float(r["R_at_1"]),
                r1_cls=float(r["Class_Oriented_R_at_1"]),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- analyses

HP = ["lr", "temp", "t2c", "eps", "bs", "epochs", "patience", "seed"]
VARIANT = r"_lr\d|_temp|_t2c|_c2t|_bs\d|balsamp|_cw|learnproto|_sgd"


def default_grid(cp):
    """Architecture grid at default hyperparameters, one value per architecture."""
    g = cp[(cp.lr == 3e-4) & (cp.temp == 0.07) & (cp.t2c == 1.0) & (cp.eps == 0)
           & (cp.seed == 42) & ~cp.exp.str.contains(VARIANT)]
    return g.groupby(["colors", "loss", "dim", "ch", "th"])[["sil_ovr", "r1_ovr"]].mean().reset_index()


def noise_floor(cp):
    keys = ["colors", "exp", "loss", "dim", "ch", "th"] + HP
    reps = cp.fillna(-1).groupby(keys).filter(lambda x: len(x) > 1).groupby(keys)
    st = reps.agg(n=("sil_ovr", "size"), sil_sd=("sil_ovr", "std"), r1_sd=("r1_ovr", "std")).reset_index()
    if st.empty:
        return st, float("nan"), float("nan")
    w = st.n - 1
    pool = lambda c: float(np.sqrt((w * st[c] ** 2).sum() / w.sum()))
    return st, pool("sil_sd"), pool("r1_sd")


def tower_pairs(grid):
    rows = []
    keys = ["colors", "loss", "dim", "th"]
    for a, b in (("16", "32x128"), ("16", "64"), ("64", "32x128"), ("128", "32x128")):
        A = grid[grid.ch == a].set_index(keys)
        B = grid[grid.ch == b].set_index(keys)
        for metric in ("sil_ovr", "r1_ovr"):
            d = (B[metric] - A[metric]).dropna()
            for k in COLOURS:
                x = d[d.index.get_level_values("colors") == k]
                if len(x) == 0:
                    continue
                rows.append(dict(
                    change=f"[{a}] -> [{b}]",
                    metric=metric,
                    colors=k,
                    pairs=len(x),
                    better=int((x > 0).sum()),
                    mean=x.mean(),
                    p_sign=binomtest(int((x > 0).sum()), len(x)).pvalue,
                    p_wilcoxon=wilcoxon(x).pvalue if len(x) >= 6 else np.nan,
                ))
    return pd.DataFrame(rows)


def tower_cost_by_loss(grid):
    keys = ["colors", "loss", "dim", "th"]
    A = grid[grid.ch == "16"].set_index(keys)
    B = grid[grid.ch == "32x128"].set_index(keys)
    d = (B[["sil_ovr", "r1_ovr"]] - A[["sil_ovr", "r1_ovr"]]).dropna().reset_index()
    d = d[d.colors.isin([96, 797])]
    return d.groupby(["loss", "colors"]).agg(
        pairs=("r1_ovr", "size"),
        silhouette=("sil_ovr", "mean"),
        r1=("r1_ovr", "mean"),
        r1_better=("r1_ovr", lambda v: int((v > 0).sum())),
    ).reset_index()


def tower_table(grid, base):
    rows = []
    for (loss, ch), g in grid.groupby(["loss", "ch"]):
        if loss not in ("mask", "proto"):
            continue
        for k, h in g.groupby("colors"):
            rows.append(dict(
                loss=loss,
                ch=ch,
                colors=k,
                archs=len(h),
                mean_gain=h.sil_ovr.mean() - base[k],
                best_gain=h.sil_ovr.max() - base[k],
            ))
    return pd.DataFrame(rows)


def proto_vs_mask(grid):
    keys = ["colors", "dim", "ch", "th"]
    P = grid[grid.loss == "proto"].set_index(keys)
    M = grid[grid.loss == "mask"].set_index(keys)
    rows = []
    for metric in ("sil_ovr", "r1_ovr"):
        d = (P[metric] - M[metric]).dropna()
        for k in COLOURS:
            x = d[d.index.get_level_values("colors") == k]
            if len(x):
                rows.append(dict(
                    metric=metric,
                    colors=k,
                    pairs=len(x),
                    proto_ahead=int((x > 0).sum()),
                    mean=x.mean(),
                    p_sign=binomtest(int((x > 0).sum()), len(x)).pvalue,
                ))
    return pd.DataFrame(rows)


def geometry_vs_accuracy(grid):
    """Spearman between silhouette and R@1 across architectures: pooled, then within each loss."""
    rows = []
    for k in COLOURS:
        g = grid[grid.colors == k]
        row = {"names": k, "all losses pooled": g.sil_ovr.corr(g.r1_ovr, method="spearman"), "n pooled": len(g)}
        for loss in ("proto", "mask", "orig", "supc"):
            h = g[g.loss == loss]
            row[loss] = h.sil_ovr.corr(h.r1_ovr, method="spearman") if len(h) >= 8 else np.nan
            row[f"n {loss}"] = len(h)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts

def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_towers(towers, base, path):
    t = towers[towers.loss == "mask"]
    fig, axes = plt.subplots(1, len(COLOURS), figsize=(10.5, 3.6), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, k in zip(axes, COLOURS):
        _style(ax)
        s = t[t.colors == k].set_index("ch").mean_gain.reindex(TOWER_ORDER)
        y = np.arange(len(TOWER_ORDER))
        ax.barh(y, s.fillna(0).values, height=0.62, color=BAR)
        for yi, v in zip(y, s.values):
            if np.isnan(v):
                ax.text(0.002, yi, "not run", va="center", fontsize=7, color=MUTED)
        ax.set_yticks(y, [f"[{c}]" if c != "0" else "[] linear" for c in TOWER_ORDER])
        ax.set_title(f"{k} names  (raw RGB {base[k]:+.3f})", fontsize=9, color=INK, loc="left")
        ax.invert_yaxis()
    axes[0].set_ylabel("colour tower (hidden layers)", fontsize=8, color=MUTED)
    fig.supxlabel("silhouette gain over raw RGB (mean over architectures, masked loss)", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_spaces(spaces, path):
    names = [n for n in SPACES if n != "RGB"]
    fig, axes = plt.subplots(1, len(COLOURS), figsize=(10.5, 3.0), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, k in zip(axes, COLOURS):
        _style(ax)
        s = spaces.set_index("space")[str(k)]
        d = [s[n] - s["RGB"] for n in names]
        y = np.arange(len(names))
        ax.barh(y, d, height=0.62, color=BAR)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_yticks(y, names)
        ax.set_title(f"{k} names", fontsize=9, color=INK, loc="left")
        ax.invert_yaxis()
    fig.supxlabel("silhouette relative to raw RGB (left of the line = worse than RGB)", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- README

def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.astype(object).to_dict("records"):
        out.append("| " + " | ".join(
            fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "—" for c in cols
        ) + " |")
    return "\n".join(out)


def main():
    os.chdir(ROOT)

    print("Loading splits for raw colour spaces and method checks...", flush=True)
    splits = {k: load_split(k) for k in COLOURS}

    print("Raw colour spaces...", flush=True)
    sp = {name: {} for name in SPACES}
    for k in COLOURS:
        Xte, yte, idx, names = splits[k]
        for name, fn in SPACES.items():
            V = np.nan_to_num(fn(Xte.astype(np.float64))).astype(np.float32)
            sp[name][str(k)] = score_space(V, yte, idx, len(names))[0]
    spaces = pd.DataFrame(sp).T.rename_axis("space").reset_index()
    spaces.to_csv(HERE / "colour_spaces.csv", index=False)
    base = {k: float(spaces.set_index("space").loc["RGB", str(k)]) for k in COLOURS}

    print("Reading geometry results from experiment_results.csv...", flush=True)
    cp = load_results_table()
    if cp.empty:
        raise RuntimeError("No Silhouette columns found in experiments/*/experiment_results.csv")
    cp.to_csv(HERE / "checkpoints.csv", index=False)

    grid = default_grid(cp)
    _, noise_sil, noise_r1 = noise_floor(cp)
    n_reps = len(noise_floor(cp)[0])
    towers = tower_table(grid, base)
    towers.to_csv(HERE / "towers.csv", index=False)
    pairs = tower_pairs(grid)
    pairs.to_csv(HERE / "tower_pairs.csv", index=False)
    cost = tower_cost_by_loss(grid)
    pvm = proto_vs_mask(grid)
    gva = geometry_vs_accuracy(grid)

    best = grid.loc[grid.groupby("colors").sil_ovr.idxmax()]

    # Reference model per-name silhouette read directly from main per_name.csv, plus RGB comparison.
    Xte, yte, idx, names = splits[96]
    K = len(names)
    rgb_sil = compute_silhouette_metrics(torch.from_numpy(Xte), yte, K, subsample=SUB, chunk_size=CHUNK, seed=42)
    pc_rgb = rgb_sil["per_class_silhouette"]

    ref_row = cp[(cp["round"] == REF_ROUND) & (cp["exp"] == REF_EXP)]
    if ref_row.empty:
        raise RuntimeError(f"Reference model {REF_ROUND}/{REF_EXP} missing from experiment_results.csv")
    ref_ovr = float(ref_row["sil_ovr"].iloc[0])

    ref_pn = pd.read_csv(ROOT / f"experiments/{REF_ROUND}/metrics/{REF_EXP}_per_name.csv").set_index("Name")
    pc_emb = np.array([float(ref_pn.loc[name, "Silhouette"]) for name in names])
    delta = pc_emb - pc_rgb
    per = pd.DataFrame(dict(
        name=names,
        test_samples=np.bincount(yte, minlength=K),
        silhouette_rgb=pc_rgb,
        silhouette_learned=pc_emb,
        change=delta,
    )).sort_values("change")
    per.to_csv(HERE / "per_class_96c.csv", index=False, float_format="%.4f")
    tangled = pc_rgb < np.median(pc_rgb)

    # One-off method checks: untrained network control, exact vs simplified silhouette, sample sizes.
    ref_model, _, bow96 = load_model(ROOT / REF_MODEL, names)
    E = encode(ref_model, Xte).numpy()
    control = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        rnd = ColorCLIPModel(
            vocab_size=bow96.shape[1], embed_dim=64, color_hidden_dims=[16],
            text_hidden_dims=[256], num_classes=K,
        ).eval()
        control.append(score_space(encode(rnd, Xte).numpy(), yte, idx, K)[0])

    small, _ = train_test_split(idx, train_size=10_000, stratify=yte[idx], random_state=42)
    exact = {
        lab: float(silhouette_samples(V[small].astype(np.float64), yte[small]).mean())
        for lab, V in (("rgb", Xte), ("learned", E))
    }
    simple = {
        lab: float(simplified_silhouette_samples(V[small], yte[small], compute_centroids(V, yte, K)).mean())
        for lab, V in (("rgb", Xte), ("learned", E))
    }
    sizes = {}
    for n in (15_000, 30_000, 60_000):
        sub, _ = (train_test_split(idx, train_size=n, stratify=yte[idx], random_state=7)
                  if n < len(idx) else (idx, None))
        sizes[n] = (score_space(Xte, yte, sub, K)[0], score_space(E, yte, sub, K)[0])

    # t->c x batch size (14th round), and other single-run knobs at 96c
    tb = cp[cp.exp.str.contains("_t2c") & cp.exp.str.contains("_bs")]
    t2c = tb.pivot_table(index="t2c", columns="bs", values="sil_ovr").round(3)
    t2c_r1 = tb.pivot_table(index="t2c", columns="bs", values="r1_ovr")
    temps = cp[
        (cp.colors == 96)
        & cp.exp.str.match(r"clip_96c_proto_dim64_ch16_th256(_temp[\d.]+)?$")
        & (cp.lr == 3e-4)
    ].groupby("temp")[["sil_ovr", "r1_ovr"]].mean()
    lr12 = cp[cp.exp.str.contains(r"_dim32_ch16_th0_lr")].dropna(subset=["sil_ovr"])
    lr_range = lr12.groupby("loss").agg(
        sil_lo=("sil_ovr", "min"), sil_hi=("sil_ovr", "max"),
        r1_lo=("r1_ovr", "min"), r1_hi=("r1_ovr", "max"),
    )

    print("Charts...", flush=True)
    chart_towers(towers, base, HERE / "towers.png")
    chart_spaces(spaces, HERE / "colour_spaces.png")

    # ------------------------------------------------------------ README
    sp_show = spaces.copy()
    for k in COLOURS:
        sp_show[str(k)] = sp_show[str(k)].map("{:+.3f}".format)
    sp_show.columns = ["space"] + [f"{k}c" for k in COLOURS]

    tw = towers[towers.loss == "mask"].pivot(index="ch", columns="colors", values="mean_gain").reindex(TOWER_ORDER)
    tw_show = pd.DataFrame({"colour tower": [f"[{c}]" if c != "0" else "[] (linear)" for c in tw.index]})
    for k in COLOURS:
        tw_show[f"{k}c"] = [f"{v:+.3f}" if pd.notna(v) else "—" for v in tw[k].values]

    pr = pairs[(pairs.metric == "sil_ovr")].copy()
    pr["better"] = pr.better.astype(str) + " / " + pr.pairs.astype(str)
    pr_show = pr[["change", "colors", "better", "mean", "p_sign"]].rename(
        columns={"change": "tower change", "colors": "names", "better": "wider better",
                 "mean": "mean gain", "p_sign": "p (sign test)"})

    gain_rows = pd.DataFrame([
        dict(
            names=k,
            **{"raw RGB": base[k], "best model": b.sil_ovr},
            gain=b.sil_ovr - base[k],
            model=f"{b.loss} dim{b.dim} ch[{b.ch}] th[{b.th}]",
        )
        for k, b in best.set_index("colors").iterrows()
    ])

    pvm_show = pvm.pivot(index="colors", columns="metric", values="mean").reset_index()
    pvm_show = pvm_show.merge(pvm[pvm.metric == "sil_ovr"][["colors", "pairs", "proto_ahead", "p_sign"]], on="colors")
    pvm_show.columns = ["names", "R@1 (proto − mask)", "silhouette (proto − mask)", "pairs",
                        "proto ahead on silhouette", "p (sign test)"]
    for c in ("names", "pairs", "proto ahead on silhouette"):
        pvm_show[c] = pvm_show[c].astype(int)

    gva_show = pd.DataFrame({"names": gva["names"], "all losses pooled": gva["all losses pooled"].map("{:+.2f}".format)})
    for loss in ("proto", "mask", "orig", "supc"):
        gva_show[loss] = [f"{v:+.2f} (n={n})" if pd.notna(v) else "—" for v, n in zip(gva[loss], gva[f"n {loss}"])]
    t2c_show = t2c.reset_index()
    for col in t2c.columns:
        t2c_show[col] = t2c_show[col].map("{:+.3f}".format)
    t2c_show.columns = ["t→c weight"] + [f"batch {int(c)}" for c in t2c.columns]
    t2c_show["t→c weight"] = t2c_show["t→c weight"].map(lambda v: f"{v:g}")

    worst = per.head(8)[["name", "silhouette_rgb", "silhouette_learned", "change"]]
    best_cls = per.tail(5)[["name", "silhouette_rgb", "silhouette_learned", "change"]]
    f4 = {c: "{:+.3f}" for c in ("silhouette_rgb", "silhouette_learned", "change", "raw RGB", "best model", "gain",
                                 "mean gain", "R@1 (proto − mask)", "silhouette (proto − mask)",
                                 "silhouette", "R@1")}
    f4.update({"p (sign test)": "{:.1g}"})

    rounds_covered = sorted(cp["round"].unique(), key=_round_sort_key)
    readme = f"""# Geometry: how well colour names separate

How cleanly the names form groups — in raw colour spaces with no model, and in the
learned colour embedding — and which settings change it. Learned silhouette and R@1
come directly from `experiments/*/experiment_results.csv` and `experiments/*/metrics/*_per_name.csv`.
Regenerate with `python annexes/geometry/build.py`.

**The short answer.** Names are barely groups at all in raw colour: above 14 names the
average sample sits closer to another name's centre than to its own. No hand-designed
colour space does better than plain RGB. Training improves separation at every colour
count, and the colour tower is the main lever — a second hidden layer beats a wider single
one in {int(pr.loc[pr.colors == 96, 'better'].iloc[0].split(' / ')[0])} of {int(pr.loc[pr.colors == 96, 'pairs'].iloc[0])} matched comparisons at 96 names and {int(pr.loc[pr.colors == 797, 'better'].iloc[0].split(' / ')[0])} of {int(pr.loc[pr.colors == 797, 'pairs'].iloc[0])} at 797. Accuracy and geometry are different
questions — some changes help one and cost the other.

## Metric

Silhouette per sample: `(b − a) / max(a, b)`, where `a` is the distance to its own name's
centre and `b` to the nearest other name's centre. +1 is cleanly separated, 0 is on the
border, negative is closer to another name. This is the simplified (centroid) form;
the textbook all-pairs form compares every sample with every other and is infeasible on
600k samples. Centres come from all clean test rows; scoring uses a stratified 60k sample of them.
"Overall" weights names by sample count.

## 1. Raw colour spaces (no model)

![](colour_spaces.png)

{md(sp_show)}

RGB and YCbCr (a linear re-mix of RGB) are at the top at every colour count. Every
perceptually-uniform space — CIELAB, OKLab — is worse: spaces built so that distance
matches perceived difference group *names* less well than device RGB. The OKLCH row is
the form `rgb_to_oklch` produces: hue stored as a plain number from 0 to 1, so 0.99 and
0.01 look far apart although they are neighbouring hues. That costs it against OKLab (the
same colours, without the wrap-around). Models only see this form with
`color_space: "oklch"`, which was used only in the 5th round — where it made no accuracy
difference (−0.0001 R@1 vs RGB over 432 paired runs). Every other round uses RGB.

## 2. Learned embedding vs raw RGB

{md(gain_rows, f4)}

The gain grows from 14 to 797 names. **Control:** a randomly
initialised network of the 96c reference shape (dim64, ch[16], th[256]) scores
{", ".join(f"{c:+.3f}" for c in control)} — no better than raw RGB ({base[96]:+.3f}) — so the
gain is learned, not a side effect of mapping 3 dimensions into 64.

## 3. The colour tower

![](towers.png)

Mean gain over raw RGB per colour tower (masked loss, the only loss run across the whole
grid; `—` = never run):

{md(tw_show)}

Matched pairs — same colour count, loss, embedding size and text tower; only the colour
tower changes:

{md(pr_show, f4)}

At 96 and 797 names this is not noise: a retrained identical config moves silhouette by
about {noise_sil:.3f} (sd), the gains are 10–25× that, and the pairs go the same way.
At 14 names the tower does not matter.
`[32,128]` beating `[128]` means the second layer matters, not only width.

**It is not free for every loss.** R@1 change for `[16] → [32,128]`:

{md(cost.rename(columns={"colors": "names", "r1": "R@1", "r1_better": "pairs where R@1 rose"}), f4)}

Masked pays ~0.5–0.9 R@1 points. Prototype has a single matched pair per colour count, so
its "free" reading is one observation, not a test.

## 4. Other settings (96 names)

**t→c weight × batch size** (14th round, prototype dim64 ch[64] th[256]) — silhouette:

{md(t2c_show)}

The ranking is identical at all three batch sizes: 0.1 best, off second, the default 1
third, 10 worst. Larger batches are better at every weight. R@1 across all twelve runs
spans {t2c_r1.values.min():.4f}–{t2c_r1.values.max():.4f}.

**Starting temperature** (prototype dim64 ch[16] th[256], one run each):
{", ".join(f"{t:g} → {r.sil_ovr:+.3f}" for t, r in temps.iterrows())}; R@1
{temps.r1_ovr.min():.4f}–{temps.r1_ovr.max():.4f}. Higher is better so far; nothing above 0.2 was run.

**Learning rate** (12th round, 1e-5 … 3e-3, dim32 ch[16] th[]): silhouette range per loss
{"; ".join(f"{l} {r.sil_lo:+.3f} … {r.sil_hi:+.3f}" for l, r in lr_range.iterrows())},
Original and supcon improve at every step up in learning rate (about +0.02 over the range);
prototype and masked show no direction. The 17th round (lr 0.01–0.5) is in report_8th_to_18th.md.

## 5. Prototype vs masked, same architecture

{md(pvm_show, f4)}

Prototype is more accurate at every colour count, by more as names grow. Its geometry is
worse at 96 names and level at 797.

## 6. Are better-separated models more accurate?

Spearman correlation between silhouette and R@1 across architectures (default
hyperparameters; `—` = fewer than 8 architectures):

{md(gva_show)}

Pooled over all losses the two look unrelated at 96 names. That is an artefact of mixing
losses: prototype models are accurate with average separation, masked models the reverse,
so pooling cancels the relationship inside each. Within prototype — the loss we use — the
architecture that separates best is almost exactly the one that is most accurate.

## 7. Who gains and who pays (96 names, reference model)

The model improves {int((delta > 0).sum())} of {K} names, overall {base[96]:+.3f} → {ref_ovr:+.3f}.
It changes the numbers more than the order: per-name silhouette in RGB vs learned,
Spearman {spearmanr(pc_rgb, pc_emb).statistic:.3f}. It spends its separation where names are
tangled: names in the more tangled half gain {delta[tangled].mean():+.3f} on average, the
better-separated half {delta[~tangled].mean():+.3f} (Spearman of change vs starting
silhouette {spearmanr(delta, pc_rgb).statistic:+.3f}). The loss only asks each sample to be on
the right side of a boundary; nothing rewards a tight cluster once it is correct.

Largest losses:

{md(worst, f4)}

Largest gains:

{md(best_cls, f4)}

Full list: `per_class_96c.csv`. Per-name scores for individual runs are stored in
`experiments/<round>/metrics/<exp>_per_name.csv`.

## Method notes

- **Noise floor.** {n_reps} configs were retrained with identical settings and seed; pooled sd
  {noise_sil:.4f} silhouette, {noise_r1:.4f} R@1. Across the 16th round's three seeds (which
  also change the train/test split), silhouette moves by about 0.002 (96 names) to 0.004
  (797 names), sd per setting.
- **Simplified vs exact silhouette** (96c, 10k sample): raw RGB {simple["rgb"]:+.3f} vs
  {exact["rgb"]:+.3f}, learned {simple["learned"]:+.3f} vs {exact["learned"]:+.3f}. The simplified form
  reads more tangled and more improved; it ranks models the same way. Do not quote it as
  the textbook number.
- **Sample size does not matter.** Overall silhouette at 15k / 30k / 60k:
  RGB {", ".join(f"{v[0]:+.3f}" for v in sizes.values())}; learned
  {", ".join(f"{v[1]:+.3f}" for v in sizes.values())}.
- **Centres come from all clean test rows, not the sample.** An earlier version computed
  them from the 60k sample, which left many rare names with no sample or only a few rows.
- **Clean rows only.** Every model is scored on its own split's test rows after the first
  5,000, which chose its epoch (research_plan.md note 10). 5,363 names are left out: after
  removing those rows, some names have one clean row or none.
- **Coverage.** {len(cp)} runs across rounds {rounds_covered[0]} through {rounds_covered[-1]}
  at 14, 96 and 797 names. Gaps that remain show as `—` in the tower table.
"""
    (HERE / "README.md").write_text(readme)
    print("Done.")


if __name__ == "__main__":
    main()
