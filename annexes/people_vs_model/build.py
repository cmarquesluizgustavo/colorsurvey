#!/usr/bin/env python3
"""
People vs model, in both directions (thin consumer of main crowd metrics).

Core tolerance-8 crowd metrics and per-name values (Colour_To_Name_Diff,
Colour_To_Name_Halves, Top_Name_Match, Name_To_Colour_Diff, Name_To_Colour_Halves,
Centre_Distance, Width_Ratio) are read directly from experiments/*/experiment_results.csv
and experiments/*/metrics/*_per_name.csv on clean test halves A + B.

This annex script retains only the supplementary checks that do not live in the main
pipeline: tolerances 4 and 16, top-answer-only colour-to-name difference (including at
tolerance 8), people-halves top-name match, and name-to-colour AUC — all evaluated strictly
on the clean held-out rows A + B (skipping the first max_eval_samples early-stopping rows
via data_prep.test_parts).
"""
import glob
import json
import os
import sys
import tarfile
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from ML.metrics.post_training import CUBE_SIDE, _box_sum, _to_cube, _width_90

TOLS = (4, 8, 16)             # per channel, on the 0-255 grid
EXTRA_TOLS = (4, 16)          # tolerances scored only inside this annex
PLOT_TOL = 8                  # tolerance in per_name.png and per_name.csv
MIN_ANSWERS = 20              # answers nearby for a colour to count in colour to name
COMMON = 200                  # test answers for a name to count as common in name to colour
CHUNK = 8                     # names scored at a time
N = CUBE_SIDE
STEP = 2                      # colour to name is scored on 1 colour in STEP^3
_g = np.arange(0, N, STEP)
GRID = ((_g[:, None, None] * N + _g[None, :, None]) * N + _g[None, None, :]).ravel()

RUNS = [("14th", f"clip_96c_proto_dim64_ch64_th256_t2c{w}_bs{b}")
        for w in ("0", "01", "1", "10") for b in (96, 512, 2048)] + [
    # best R@1 per loss with a checkpoint, 96 and 14 names
    ("16th", "clip_96c_proto_dim64_ch32x128x128_th256_temp0.07_t2c01_s7"),
    ("8th", "clip_96c_mask_dim128_ch128_th128x256"),
    ("11th", "clip_96c_orig_dim4_ch8_th0"),
    ("11th", "clip_96c_supc_dim4_ch8_th0"),
    ("15th", "clip_14c_proto_dim128_ch64_th256"),
    ("8th", "clip_14c_mask_dim64_ch128_th256x512"),
    ("8th", "clip_14c_orig_dim64_ch32_th128x256"),
    ("8th", "clip_14c_supc_dim64_ch32_th128x256"),
] + [("20th", f"clip_96c_proto_dim64_ch64_th256_t2c{w}_sgd") for w in ("0", "01", "1", "10")] + [
    ("21st", f"clip_797c_proto_dim64_ch64_th256_t2c{w}_{o}")
    for o in ("adamw", "sgd") for w in ("0", "01", "1", "10")]


def checkpoint(rnd, run):
    return os.path.join("experiments", rnd, "models", run, "best_color_clip_model.pth")


def tv(p, q):
    return np.abs(p - q).sum() / 2


def score(rnd, run):
    """Score supplementary checks (tolerances 4 & 16, top-answer-only, AUC) on clean A + B."""
    import torch
    from sklearn.metrics import roc_auc_score
    from ML.data_prep import DataLoader, test_parts
    from ML.trainers.color_clip_trainer import ColorCLIPTrainer

    class Logger:
        log_dir = tempfile.mkdtemp()

    ckpt = torch.load(checkpoint(rnd, run), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg["data"].get("color_space", "rgb") != "rgb":
        raise ValueError("only RGB input is handled")
    bundle = DataLoader(cfg).load()
    tr = ColorCLIPTrainer(cfg, bundle, Logger())
    tr.model.load_state_dict(ckpt["model_state_dict"])
    tr.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
    text = tr._compute_class_prototypes()
    temp = tr.loss_fn.temperature
    names = list(bundle["label_encoder"].classes_)
    K = len(names)

    y_all = np.asarray(bundle["y_test"])
    n_select = cfg["training"].get("max_eval_samples", 5000)
    parts = test_parts(y_all, n_select)
    clean = np.sort(np.concatenate([parts["a"], parts["b"]]))
    half = np.isin(clean, parts["a"])
    X = np.rint(np.asarray(bundle["X_test"])[clean] * (N - 1)).astype(np.int64)
    y = y_all[clean]

    del bundle
    tr.train_loader = tr.test_loader = tr.data_bundle = None
    uniq, inv = np.unique((X[:, 0] * N + X[:, 1]) * N + X[:, 2], return_inverse=True)
    U = len(uniq)
    rgb = np.stack([uniq // (N * N), uniq // N % N, uniq % N], 1)
    cnt = np.bincount(inv, minlength=U).astype(np.float32)
    cnt_a = np.bincount(inv[half], minlength=U).astype(np.float32)
    cnt_b = np.bincount(inv[~half], minlength=U).astype(np.float32)

    def cube(v):
        return _to_cube(uniq, v)

    with torch.no_grad():
        uniq_embeds = tr.model.encode_color(torch.from_numpy(rgb / (N - 1)).float().to(tr.device)).cpu()

    def logits(ks):
        """(colours, len(ks)) scores divided by the temperature."""
        with torch.no_grad():
            return (uniq_embeds @ text[ks].T).numpy() / temp

    chunks = [list(range(s, min(s + CHUNK, K))) for s in range(0, K, CHUNK)]
    out = {"answers": np.bincount(y, minlength=K), "auc": np.zeros(K)}

    if K > 96:
        for ks in chunks:
            lg = logits(ks)
            for j, k in enumerate(ks):
                out["auc"][k] = roc_auc_score(y == k, lg[inv, j])
            print(f"  AUC: names {ks[0]}-{ks[-1]} done", flush=True)
    else:
        lse = np.full(U, -np.inf, np.float32)
        top_v, top_k = np.full(U, -np.inf, np.float32), np.zeros(U, np.int64)
        for ks in chunks:
            lg = logits(ks)
            lse = np.logaddexp(lse, np.logaddexp.reduce(lg, axis=1))
            j = lg.argmax(1)
            win = lg[np.arange(U), j] > top_v
            top_v[win], top_k[win] = lg[np.arange(U), j][win], np.array(ks)[j[win]]

        for tol in TOLS:
            tot = _box_sum(cube(cnt), tol)
            tot_a = _box_sum(cube(cnt_a), tol)
            tot_b = _box_sum(cube(cnt_b), tol)
            near = GRID[tot[GRID] >= MIN_ANSWERS]
            t, ta, tb = tot[near], tot_a[near], tot_b[near]
            del tot, tot_a, tot_b
            both = (ta > 0) & (tb > 0)

            diff_top = np.zeros(len(near), np.float32)
            best_a = np.full(len(near), -1.0, np.float32)
            best_b = np.full(len(near), -1.0, np.float32)
            top_a = np.zeros(len(near), np.int16)
            top_b = np.zeros(len(near), np.int16)

            is_extra = tol in EXTRA_TOLS
            if is_extra:
                diff_model = np.zeros(len(near), np.float32)
                diff_halves = np.zeros(len(near), np.float32)
                best_m = np.full(len(near), -1.0, np.float32)
                best_p = np.full(len(near), -1.0, np.float32)
                top_m = np.zeros(len(near), np.int16)
                top_p = np.zeros(len(near), np.int16)
                per = {f: np.zeros(K) for f in ("name_model", "name_halves", "width_model", "width_people")}
                if tol == TOLS[0]:
                    out["centre_model"] = np.zeros((K, 3))
                    out["centre_people"] = np.zeros((K, 3))

            for ks in chunks:
                lg = logits(ks)
                for j, k in enumerate(ks):
                    rows = y == k
                    if tol == TOLS[0]:
                        out["auc"][k] = roc_auc_score(rows, lg[inv, j])
                    people = _box_sum(cube(np.bincount(inv[rows], minlength=U).astype(np.float32)), tol)
                    a = _box_sum(cube(np.bincount(inv[rows & half], minlength=U).astype(np.float32)), tol)
                    b_cube = np.maximum(people - a, 0.0)
                    shares_people = people[near] / t
                    shares_a = a[near] / np.maximum(ta, 1)
                    shares_b = b_cube[near] / np.maximum(tb, 1)

                    diff_top += np.abs(_box_sum(cube(cnt * (top_k == k)), tol)[near] / t - shares_people) / 2
                    win_a = shares_a > best_a
                    best_a[win_a], top_a[win_a] = shares_a[win_a], k
                    win_b = shares_b > best_b
                    best_b[win_b], top_b[win_b] = shares_b[win_b], k

                    if is_extra:
                        shares_model = _box_sum(cube(cnt * np.exp(lg[:, j] - lse)), tol)[near] / t
                        diff_model += np.abs(shares_model - shares_people) / 2
                        diff_halves += np.abs(shares_a - shares_b) / 2
                        win_m = shares_model > best_m
                        best_m[win_m], top_m[win_m] = shares_model[win_m], k
                        win_p = shares_people > best_p
                        best_p[win_p], top_p[win_p] = shares_people[win_p], k

                        q = np.log(cnt) + lg[:, j]
                        q = np.exp(q - q.max())
                        q /= q.sum()
                        model_dist = _box_sum(cube(q), tol)
                        model_dist /= model_dist.sum()
                        b_norm = b_cube / b_cube.sum()
                        a_norm = a / a.sum()
                        per["name_halves"][k] = tv(a_norm, b_norm)
                        people_norm = people / people.sum()
                        per["name_model"][k] = tv(model_dist, people_norm)
                        per["width_model"][k], per["width_people"][k] = _width_90(model_dist), _width_90(people_norm)
                        if tol == TOLS[0]:
                            out["centre_model"][k] = q @ rgb
                            out["centre_people"][k] = X[rows].mean(0)
                print(f"  tolerance {tol}: names {ks[0]}-{ks[-1]} done", flush=True)

            out[f"colour_model_top_{tol}"] = diff_top.mean()
            out[f"top_match_halves_{tol}"] = (top_a == top_b)[both].mean()
            out[f"colours_scored_{tol}"] = len(near)
            if is_extra:
                out[f"colour_model_{tol}"] = diff_model.mean()
                out[f"colour_halves_{tol}"] = diff_halves[both].mean()
                out[f"top_match_{tol}"] = (top_m == top_p).mean()
                for f, v in per.items():
                    out[f"{f}_{tol}"] = v

        if "centre_model" in out:
            out["centre"] = np.linalg.norm(out["centre_model"] - out["centre_people"], axis=1)
    tc = cfg["training"]
    np.savez_compressed(
        f"{run}.npz", round=rnd, run=run, names=np.array(names), loss=tc["loss_type"],
        t2c_weight=tc.get("t2c_weight", 1.0), batch_size=cfg["data"]["batch_size"],
        temperature=temp, **out)
    print(f"{run}: AUC {out['auc'].mean():.3f}", flush=True)


def label(d):
    s = f"{len(d['names'])} names, {d['loss']}"
    if str(d["round"]) == "14th":
        s += f", weight {float(d['t2c_weight']):g}, batch {int(d['batch_size'])}"
    elif str(d["round"]) in ("20th", "21st"):
        s += f", weight {float(d['t2c_weight']):g}, {str(d['run']).rsplit('_', 1)[1].upper()}"
    return s


def report():
    """Assemble annex CSVs and charts from main experiment results + supplementary npz files."""
    import matplotlib.pyplot as plt
    import pandas as pd

    npz_runs = {
        (str(d["round"]), str(d["run"])): d
        for d in (dict(np.load(f)) for f in sorted(glob.glob(os.path.join(HERE, "out", "*.npz"))))
    }
    res_tables = {}
    for f in glob.glob(os.path.join(ROOT, "experiments", "*", "experiment_results.csv")):
        rnd = f.split(os.sep)[-2]
        res_tables[rnd] = pd.read_csv(f).set_index("Experiment")

    rows = []
    per_name_rows = []
    plot_runs = []

    for rnd, run in RUNS:
        tbl = res_tables.get(rnd)
        pn_path = os.path.join(ROOT, "experiments", rnd, "metrics", f"{run}_per_name.csv")
        d = npz_runs.get((rnd, run))
        if tbl is None or run not in tbl.index or not os.path.exists(pn_path) or d is None:
            continue
        main_row = tbl.loc[run]
        has_crowd = pd.notna(main_row.get("Colour_To_Name_Diff"))
        if len(d["names"]) <= 96 and not has_crowd:
            continue

        pn = pd.read_csv(pn_path).set_index("Name")
        names = [str(n) for n in d["names"]]
        answers = np.array([int(pn.loc[n, "Count"]) for n in names])
        tol8_name_model = np.array([float(pn.loc[n, "Name_To_Colour_Diff"]) for n in names])
        tol8_name_halves = np.array([float(pn.loc[n, "Name_To_Colour_Halves"]) for n in names])
        tol8_centre = np.array([float(pn.loc[n, "Centre_Distance"]) for n in names])
        tol8_width = np.array([float(pn.loc[n, "Width_Ratio"]) for n in names])
        common = answers >= COMMON

        plot_runs.append({
            "round": rnd,
            "run": run,
            "names": np.array(names),
            "answers": answers,
            "loss": str(d["loss"]),
            "t2c_weight": float(d["t2c_weight"]),
            "batch_size": int(d["batch_size"]),
            "auc": d["auc"],
            "name_model_8": tol8_name_model,
            "name_halves_8": tol8_name_halves,
            "centre": tol8_centre,
            "width_ratio_8": tol8_width,
        })

        def _opt_round(val, ndigits=4):
            if val is None:
                return None
            v = float(val)
            return None if np.isnan(v) else round(v, ndigits)

        order = np.argsort(-answers)
        for rank, k in enumerate(order, 1):
            per_name_rows.append({
                "names": len(names),
                "run": run,
                "rank": rank,
                "name": names[k],
                "answers": int(answers[k]),
                "name_to_colour_difference": _opt_round(tol8_name_model[k]),
                "people_halves": _opt_round(tol8_name_halves[k]),
                "centre_distance": _opt_round(tol8_centre[k]),
                "width_model_over_people": _opt_round(tol8_width[k]),
                "auc": round(float(d["auc"][k]), 4),
            })

        for tol in (TOLS if has_crowd else (None,)):
            if tol is None:
                nm = nh = None
                c2n_diff = c2n_halves = top_match = n2c_diff = n2c_halves = width_ratio = centre_dist = None
            elif tol == PLOT_TOL:
                nm, nh = tol8_name_model, tol8_name_halves
                c2n_diff = float(main_row["Colour_To_Name_Diff"])
                c2n_halves = float(main_row["Colour_To_Name_Halves"])
                top_match = float(main_row["Top_Name_Match"])
                n2c_diff = float(main_row["Name_To_Colour_Diff"])
                n2c_halves = float(main_row["Name_To_Colour_Halves"])
                width_ratio = float(main_row["Width_Ratio"])
                centre_dist = float(main_row["Centre_Distance"])
            else:
                nm, nh = d[f"name_model_{tol}"], d[f"name_halves_{tol}"]
                c2n_diff = float(d[f"colour_model_{tol}"])
                c2n_halves = float(d[f"colour_halves_{tol}"])
                top_match = float(d[f"top_match_{tol}"])
                n2c_diff = float(nm.mean())
                n2c_halves = float(nh.mean())
                width_ratio = float(np.median(d[f"width_model_{tol}"] / d[f"width_people_{tol}"]))
                centre_dist = float(tol8_centre.mean())

            rows.append({
                "round": rnd,
                "run": run,
                "names": len(names),
                "loss": str(d["loss"]),
                "t2c_weight": float(d["t2c_weight"]),
                "batch_size": int(d["batch_size"]),
                "R_at_1": float(main_row["R_at_1"]),
                "temperature": round(float(main_row["Temperature"]), 4),
                "tolerance": tol,
                "colour_to_name_difference": _opt_round(c2n_diff),
                "colour_to_name_difference_top_answer": _opt_round(d[f"colour_model_top_{tol}"]) if tol is not None else None,
                "colour_to_name_people_halves": _opt_round(c2n_halves),
                "top_name_match": _opt_round(top_match),
                "top_name_match_people_halves": _opt_round(d[f"top_match_halves_{tol}"]) if tol is not None else None,
                "name_to_colour_difference": _opt_round(n2c_diff),
                "name_to_colour_people_halves": _opt_round(n2c_halves),
                "common_names": int(common.sum()),
                "name_to_colour_difference_common": _opt_round(nm[common].mean()) if nm is not None else None,
                "name_to_colour_people_halves_common": _opt_round(nh[common].mean()) if nh is not None else None,
                "name_to_colour_difference_rare": _opt_round(nm[~common].mean()) if nm is not None and (~common).any() else None,
                "name_to_colour_people_halves_rare": _opt_round(nh[~common].mean()) if nh is not None and (~common).any() else None,
                "width_model_over_people": _opt_round(width_ratio, 3),
                "centre_distance": _opt_round(centre_dist, 2),
                "auc": round(float(d["auc"].mean()), 4),
            })

    if not rows:
        raise RuntimeError("No runs had matching main experiment results and supplementary .npz outputs.")

    summary_df = pd.DataFrame(rows).sort_values(
        ["names", "round", "loss", "t2c_weight", "batch_size", "tolerance"],
        ascending=[False, True, True, True, True, True],
    )
    summary_df["tolerance"] = summary_df["tolerance"].astype("Int64")
    summary_df.to_csv(os.path.join(HERE, "summary.csv"), index=False)

    pd.DataFrame(per_name_rows).sort_values(
        ["names", "run", "rank"], ascending=[False, True, True],
    ).to_csv(os.path.join(HERE, "per_name.csv"), index=False)

    shown = [d for d in plot_runs if d["round"] not in ("14th", "20th", "21st")
             or (d["batch_size"] == 2048 and d["t2c_weight"] in (0, 10))]
    shown.sort(key=lambda d: (-len(d["names"]), label(d)))
    sizes = sorted({len(d["names"]) for d in shown}, reverse=True)

    def _per_name_panel(d):
        return {
            "name_to_colour_difference": d["name_model_8"],
            "people_halves": d["name_halves_8"],
            "centre_distance": d["centre"],
            "width_model_over_people": d["width_ratio_8"],
            "auc": d["auc"],
        }

    panels = [
        ("name_to_colour_difference", "name to colour difference\n(black: people halves)"),
        ("centre_distance", "centre off by (RGB units)"),
        ("width_model_over_people", "width, model / people"),
        ("auc", "AUC"),
    ]
    bar_sizes = [k for k in sizes if k <= 96]
    if bar_sizes:
        fig, axes = plt.subplots(len(panels) * len(bar_sizes), 1,
                                 figsize=(22, 3 * len(panels) * len(bar_sizes)))
        for s, k in enumerate(bar_sizes):
            group = [d for d in shown if len(d["names"]) == k]
            order = np.argsort(-group[0]["answers"])
            x = np.arange(k)
            w = 0.8 / len(group)
            for i, d in enumerate(group):
                v = _per_name_panel(d)
                for j, (f, _) in enumerate(panels):
                    axes[s * len(panels) + j].bar(x + (i - (len(group) - 1) / 2) * w, v[f][order], w, label=label(d))
            axes[s * len(panels)].scatter(x, _per_name_panel(group[0])["people_halves"][order], color="k",
                                          marker="_", s=80, zorder=3, label="people halves")
            axes[s * len(panels) + 2].axhline(1, color="k", lw=0.8)
            aucs = np.concatenate([d["auc"] for d in group])
            axes[s * len(panels) + 3].set_ylim(aucs.min() - 0.01, 1)
            for j, (_, title) in enumerate(panels):
                ax = axes[s * len(panels) + j]
                ax.set_ylabel(title, fontsize=8)
                ax.set_xlim(-0.6, k - 0.4)
                ax.set_xticks(x, group[0]["names"][order], rotation=90, fontsize=6 if k > 20 else 8)
            axes[s * len(panels)].set_title(
                f"{k} names, tolerance {PLOT_TOL}: names from most answered (left) to least answered (right)",
                fontsize=10)
            axes[s * len(panels)].legend(fontsize=7, ncol=4)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "per_name.png"), dpi=110)
        plt.close(fig)

    if sizes:
        fig, axes = plt.subplots(len(sizes), 1, figsize=(11, 4 * len(sizes)), squeeze=False)
        for ax, k in zip(axes[:, 0], sizes):
            group = [d for d in shown if len(d["names"]) == k]
            order = np.argsort(-group[0]["answers"])
            for d in group:
                ax.plot(d["auc"][order], marker=".", lw=1, label=label(d))
            ax.set_ylabel("AUC")
            ax.set_title(f"{k} names: name to colour AUC, names from most to least answered", fontsize=10)
            if k <= 20:
                ax.set_xticks(range(k), group[0]["names"][order], rotation=45, ha="right", fontsize=8)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "auc.png"), dpi=120)
        plt.close(fig)


def pack(out):
    os.chdir(ROOT)
    os.makedirs(out, exist_ok=True)
    active_runs = [(rnd, run) for rnd, run in RUNS if os.path.exists(checkpoint(rnd, run))]
    with tarfile.open(os.path.join(out, "in.tar.gz"), "w:gz") as tar:
        tar.add("ML", filter=lambda t: None if "__pycache__" in t.name else t)
        tar.add("annexes/people_vs_model/build.py")
        for rnd, run in active_runs:
            tar.add(checkpoint(rnd, run))
    with open(os.path.join(out, "runs.txt"), "w") as f:
        f.writelines(f"{rnd}, {run}\n" for rnd, run in active_runs)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pack":
        pack(os.path.abspath(sys.argv[2]))
    elif cmd == "score":
        score(sys.argv[2], sys.argv[3])
    elif cmd == "report":
        report()
