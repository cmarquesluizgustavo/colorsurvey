"""
Generate ColorCLIP experiment configs for the 5th experiment round.

Grid:
    color_space   : [rgb, oklch]
    top_n_colors  : [15, 129]
    embed_dim     : [16, 32, 64, 128]
    temperature   : [0.03, 0.07, 0.15]
    lr            : [0.0003, 0.001, 0.003]
    weight_decay  : [0.0, 0.01, 0.1]

Fixed:
    vocab_size    = 100
    batch_size    = 512
    epochs        = 30
    balance       = none (original distribution)
"""
import json
import os

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
COLOR_SPACES  = ["rgb", "oklch"]
TOP_N_COLORS  = [15, 129]
EMBED_DIMS    = [16, 32, 64, 128]
TEMPERATURES  = [0.03, 0.07, 0.15]
LRS           = [0.0003, 0.001, 0.003]
WEIGHT_DECAYS = [0.0, 0.01, 0.1]

BASE = {
    "trainer_type": "color_clip",
    "seed": 42,
    "data": {
        "csv_path": "mainsurvey_data.csv",
        "test_size": 0.2,
        "balance_strategy": "none",
        "vocab_size": 100,
        "batch_size": 512,
    },
    "training": {
        "epochs": 30,
    },
}

configs = []
for cs in COLOR_SPACES:
    for top_n in TOP_N_COLORS:
        for dim in EMBED_DIMS:
            for temp in TEMPERATURES:
                for lr in LRS:
                    for wd in WEIGHT_DECAYS:
                        name = (
                            f"clip_{cs}_{top_n}c"
                            f"_dim{dim}"
                            f"_t{temp}"
                            f"_lr{lr}"
                            f"_wd{wd}"
                        )
                        cfg = json.loads(json.dumps(BASE))
                        cfg["experiment_name"] = name
                        cfg["data"]["color_space"] = cs
                        cfg["data"]["top_n_colors"] = top_n
                        cfg["model"] = {"embed_dim": dim}
                        cfg["training"]["temperature"] = temp
                        cfg["training"]["lr"] = lr
                        cfg["training"]["weight_decay"] = wd
                        configs.append((name, cfg))

# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
paths = []
for name, cfg in configs:
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    paths.append(path)

print(f"Generated {len(configs)} config files in {out_dir}")
for p in paths:
    print(f"  {os.path.basename(p)}")
