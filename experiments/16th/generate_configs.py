"""
Generate ColorCLIP configs for the 16th round — can geometry be pushed further, and how
big is the noise really?

Background (annexes/geometry/): a wider colour tower reliably improves silhouette.
ch[16] -> ch[32,128] wins in 49/49 matched pairs at 96c and 31/31 at 797c, and the
two-layer [32,128] beats the one-layer [128] in 40/45 and 30/30 — depth helps, not just
width. Single runs also hint that a higher starting temperature (0.02 -> 0.07 -> 0.2 gave
-0.283 -> -0.276 -> -0.270, R@1 unchanged) and a small t->c weight (0.1 beat 0, 1 and 10
at all three batch sizes in the 14th round) help geometry.

Two questions:
  1. Does geometry keep improving past what we tried? Three layers ([32,128,128]) and a
     wider two-layer tower ([64,256]) were never run; starting temperature was never set
     above 0.2.
  2. How large is run-to-run noise? Every run so far uses seed 42, and only 5 configs were
     ever retrained, so the noise floor (silhouette sd ~0.002) rests on 10 runs. Here
     every cell is run with 3 seeds.

Grid: prototype, 96c and 797c, dim64, th256, t2c_weight 0.1,
      colour tower {[32,128], [32,128,128], [64,256]} x starting temperature {0.07, 0.2, 0.5}
      x seed {42, 7, 123} = 54 runs.
[32,128] at temperature 0.07 is the in-round reference, so every comparison is internal.

Things to know before analysing:
  - `seed` sets BOTH the weight init AND the train/test split (ML/data_prep.py, the
    train_test_split uses config["seed"]). A seed-7 model must be scored on the seed-7
    test set; scoring it on the seed-42 split would test it on its own training data.
    Compare cells within a seed (paired), and read the seed-to-seed spread as noise that
    includes split variation — a conservative estimate.
  - Temperature is learnable (log-parametrised, clamped to [0.01, 100]); the config sets
    its starting value.
  - epochs/patience match the 15th round so cells are comparable to it; this inherits the
    test-set early stopping of research_plan note 10.
  - save_ranking/save_scores off: silhouette is recomputed from the checkpoints.
"""
import json
import os

BASE = {
    "trainer_type": "color_clip",
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
        "lr": 0.0003,
        "weight_decay": 0.0,
        "loss_type": "prototype",
        "t2c_weight": 0.1,
        "early_stopping": {"metric": "mrr", "patience": 20, "min_delta": 0.001},
        "save_ranking": False,
        "save_scores": False,
    },
}

COLORS = [96, 797]
TOWERS = [[32, 128], [32, 128, 128], [64, 256]]
TEMPERATURES = [0.07, 0.2, 0.5]
SEEDS = [42, 7, 123]

tag = lambda dims: "x".join(str(d) for d in dims)

configs = []
for colors in COLORS:
    for ch in TOWERS:
        for temp in TEMPERATURES:
            for seed in SEEDS:
                cfg = json.loads(json.dumps(BASE))
                cfg["seed"] = seed
                cfg["data"]["top_n_colors"] = colors
                cfg["training"]["temperature"] = temp
                cfg["model"] = {"embed_dim": 64, "color_hidden_dims": ch, "text_hidden_dims": [256]}
                name = f"clip_{colors}c_proto_dim64_ch{tag(ch)}_th256_temp{temp}_t2c01_s{seed}"
                cfg["experiment_name"] = name
                configs.append((name, cfg))

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
for name, cfg in configs:
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
assert len({n for n, _ in configs}) == len(configs)
print(f"{len(configs)} configs written to {out_dir}")
