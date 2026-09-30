"""
Generate ColorCLIP configs for the 15th round — filling in the geometry grid.

LOWER PRIORITY THAN EVERY OTHER OPEN ROUND. Nothing here is expected to move R@1,
and none of it answers a question about accuracy. The single purpose is to complete
the architecture x silhouette grid so the *geometry* (embedding-shape) analysis can be
stated without holes. Run it when the cluster is otherwise idle.

Why the grid has holes at all: the silhouette-vs-architecture result was assembled
post hoc from three rounds that were never designed to be compared. The 7th round
covered 14c/5363c with th32/64/128; the 8th covered 96c/797c with th128-512; the 11th
covered 96c only, with the bare/tiny color towers. So the cells that exist differ by
colour count, and two of the conclusions rest on absent cells rather than measured ones.

What we know going in (all measured, see research_plan.md notes 7-8):
  - The color tower is worth 3-5x more than embed_dim for silhouette. Gain over raw
    RGB, best cell per tower, masked: ch[32,128] = +0.108 (96c) / +0.184 (797c),
    ch[16] = +0.081 / +0.130, bare Linear(3,dim) = +0.014 (96c).
  - Capacity keeps buying geometry well past where it stops buying R@1 — a 457-param
    model is 1.4 R@1 points behind a 34,208-param one, but is *worse than raw RGB* as
    a space at dim4 (-0.421 vs -0.341).
  - The architecture ranking is nearly loss-independent (Spearman 0.84-0.99 between
    losses on shared archs), which is what makes filling holes with `masked` useful
    for predicting the others.

Three groups, in descending order of value.

C - prototype with a wide color tower (14 runs) -- the only group that could improve
    the model rather than only measure it, and the one with a confirmed first step.

    The premise is NOT untested. Two 9th-round-era runs sit in `experiments/runs/`
    (outside the `*/models/*` layout, which is why they were missed in the first pass
    of this analysis) and they already ran prototype at ch[64]:

      96c  prototype dim64 ch[16] th256 : silhouette -0.2729   R@1 0.4834   <- canonical
      96c  prototype dim64 ch[64] th256 : silhouette -0.2578   R@1 0.4858
      96c  prototype dim64 ch[64] th256 + t2c_weight 0.1
                                        : silhouette -0.2528   R@1 0.4860
      14c  prototype dim64 ch[16] th256 : silhouette +0.1586   R@1 0.7690
      14c  prototype dim64 ch[64] th256 : silhouette +0.1685   R@1 0.7691

    Widening the color tower one step improved BOTH metrics at 96c (+0.015 silhouette,
    +0.0024 R@1) and improved geometry at 14c with accuracy flat. That is exactly what
    the rho=+0.950 within-prototype rank correlation predicted (vs +0.510 masked, -0.148
    original, -0.180 supcon), so the correlation is behaving causally rather than
    incidentally.

    What is still open is whether it keeps going. ch[64] is the *second*-best tower in
    the masked table; ch[32,128] is the best (+0.108 vs +0.102 at 96c, +0.184 vs +0.168
    at 797c). And nothing wide has been tried at 797c or 5363c at all, where the masked
    gains are 1.7x larger. Hence: ch[64] as the replication anchor, ch[32,128] as the
    step beyond, dim 64 and 128, all four colour counts.

    Counter-evidence to respect: on identical architectures prototype's mean silhouette
    is -0.322 vs masked's -0.302, i.e. prototype still trades geometry for accuracy
    overall. The two runs above are a single architecture step, not a trend. Both gains
    are also small enough to sit near the +-0.006 R@1 early-stopping noise of note 10.
    Do not over-read them; that is what this group is for.

    Two cells are SKIPPED because they already exist (see SKIP below): 96c and 14c at
    dim64/ch[64]. Rerunning them under a new name would create a duplicate-name hazard
    of exactly the kind that has already bitten this project once.

A - 5363c wide color towers (6 runs) -- closes the one hole that currently reads as a
    finding. The 5363c column only ever had ch[8]/[16]/[32]/[8,32] and dim<=64, so its
    apparent collapse (best gain +0.045 vs 797c's +0.184) is *absence of the wide
    towers*, not a measurement. Since "geometry gain grows with K" is the headline of
    note 8's second half, the largest K should not be the extrapolated one.

B - 14c bare and tiny color towers (6 runs) -- lowest value of the three, included for
    grid completeness only. 14c is the only colour count where the labels are actually
    clusters in raw RGB (silhouette +0.124, positive), and nobody wants a tiny model
    there, so this fills table cells rather than answering anything. It does let the
    "bare linear map is a fine classifier and a useless color space" claim be stated at
    two colour counts instead of one.

Design decisions worth knowing:
  - `save_ranking`/`save_scores` are both False. Silhouette is computed from the
    checkpoint by re-encoding the test set, so no dumps are needed - and a score matrix
    at 5363c would be 601k x 5363 floats (~12.9 GB per run).
  - epochs=100 / patience=20 match the 11th round rather than being re-tuned, so these
    runs are directly comparable to the ~900 existing checkpoints the grid is built
    from. This deliberately inherits the test-set early-stopping leak of note 10;
    fixing it here would make the new cells incomparable to the old ones, which defeats
    the purpose. Every number from this round carries the same ~+0.006 R@1 inflation as
    everything it is being compared against.
  - Group B uses `th0` because the 96c ch0/ch4 cells it is matching come from the 11th
    round, which used a minimal text tower. Groups A and C use `th256` to match the
    8th/10th-round cells they extend. The text tower is known irrelevant (note 4), and
    the grid tables max over th, so this only matters for like-for-like reading.
  - `"color_hidden_dims": []` needs no library change: ColorEncoder only raises on
    None, and an empty list just skips the hidden-layer loop. Names use ch0/th0 for the
    empty case so cluster/generate_experiments_txt.py's `_ch([\\dx]+)` regex still parses.
"""
import json
import os

BASE = {
    "trainer_type": "color_clip",
    "seed": 42,
    "data": {
        "csv_path": "mainsurvey.csv",
        "test_size": 0.2,
        "balance_strategy": "none",
        "color_space": "rgb",
        "vocab_size": "auto",
        "batch_size": 2048,
    },
    "training": {
        "epochs": 100,
        "temperature": 0.07,
        "lr": 0.0003,
        "weight_decay": 0.0,
        "early_stopping": {"metric": "mrr", "patience": 20, "min_delta": 0.001},
        "save_ranking": False,
        "save_scores": False,
    },
}

LOSS_TAG = {"prototype": "proto", "masked": "mask"}

tag = lambda dims: "x".join(str(d) for d in dims) if dims else "0"

# (group, loss, colours, embed_dim, color_hidden, text_hidden)
GRID = []

# Already trained in the 9th-round era under this exact architecture; kept out so the
# round does not produce two different checkpoints sharing one name.
SKIP = {
    ("prototype", 96, 64, "64", "256"),   # experiments/runs/clip_96c_proto_dim64_ch64_th256
    ("prototype", 14, 64, "64", "256"),   # experiments/runs/clip_14c_proto_dim64_ch64_th256
}

# --- C: prototype x wide color tower, every colour count -------------------
# ch[64] is the replication anchor (already a confirmed win at 96c/14c), ch[32,128] the
# untested step beyond it. 797c and 5363c have never seen a wide prototype tower at all.
for colors in (14, 96, 797, 5363):
    for dim in (64, 128):
        for ch in ([64], [32, 128]):
            GRID.append(("C", "prototype", colors, dim, ch, [256]))

# --- A: 5363c wide color towers, masked (extends the 797c column) ----------
for dim in (64, 128):
    for ch in ([64], [128], [32, 128]):
        GRID.append(("A", "masked", 5363, dim, ch, [256]))

# --- B: 14c bare / tiny color towers, masked (matches the 96c ch0/ch4 cells)
for dim in (4, 8, 64):
    for ch in ([], [4]):
        GRID.append(("B", "masked", 14, dim, ch, []))

configs = []
skipped = []
for group, loss, colors, dim, ch, th in GRID:
    if (loss, colors, dim, tag(ch), tag(th)) in SKIP:
        skipped.append(f"clip_{colors}c_{LOSS_TAG[loss]}_dim{dim}_ch{tag(ch)}_th{tag(th)}")
        continue
    cfg = json.loads(json.dumps(BASE))
    cfg["data"]["top_n_colors"] = colors
    cfg["training"]["loss_type"] = loss
    cfg["model"] = {"embed_dim": dim, "color_hidden_dims": ch, "text_hidden_dims": th}
    name = (f"clip_{colors}c_{LOSS_TAG[loss]}_dim{dim}"
            f"_ch{tag(ch)}_th{tag(th)}")
    cfg["experiment_name"] = name
    configs.append((group, name, cfg))

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
for _, name, cfg in configs:
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)

names = [n for _, n, _ in configs]
assert len(set(names)) == len(names), "duplicate config name"
for g in ("C", "A", "B"):
    sel = [n for gg, n, _ in configs if gg == g]
    print(f"  {g}: {len(sel):>2} configs")
    for n in sel:
        print(f"       {n}")
for n in skipped:
    print(f"  skipped (already trained): {n}")
print(f"{len(configs)} configs written to {out_dir}")
