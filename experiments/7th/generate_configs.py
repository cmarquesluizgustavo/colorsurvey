"""
Generate ColorCLIP experiment configs for the 7th experiment round.

Grid:
    top_n_colors       : [14, 96, 797, 5363]
    loss_type          : [original, masked, supcon]
    embed_dim          : [8, 16, 32]
    color_hidden_dims  : [[8], [16], [32], [8,32]]
    text_hidden_dims   : [[32], [64], [128], [32,64]]

Fixed:
    color_space    = rgb
    lr             = 0.0003
    weight_decay   = 0.0
    temperature    = 0.07
    balance        = none
    batch_size     = 512
    epochs         = 100
    vocab_size     = auto (uses all unique tokens in the dataset)
    early_stopping : metric=mrr, patience=5, min_delta=0.001

Total: 4 × 3 × 3 × 4 × 4 = 576 configs
"""
import json
import os

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
TOP_N_COLORS       = [14, 96, 797, 5363]
LOSS_TYPES         = ["original", "masked", "supcon"]
EMBED_DIMS         = [8, 16, 32]
COLOR_HIDDEN_DIMS  = [[8], [16], [32], [8, 32]]
TEXT_HIDDEN_DIMS   = [[32], [64], [128], [32, 64]]

BASE = {
    "trainer_type": "color_clip",
    "seed": 42,
    "data": {
        "csv_path": "mainsurvey.csv",
        "test_size": 0.2,
        "balance_strategy": "none",
        "color_space": "rgb",
        "vocab_size": "auto",
        "batch_size": 512,
    },
    "training": {
        "epochs": 100,
        "temperature": 0.07,
        "lr": 0.0003,
        "weight_decay": 0.0,
        "early_stopping": {
            "metric": "mrr",
            "patience": 5,
            "min_delta": 0.001,
        },
    },
}


def dims_tag(dims):
    """[4,32] -> '4x32', [8] -> '8'."""
    return "x".join(str(d) for d in dims)


def loss_tag(loss_type):
    """original -> 'orig', masked -> 'mask', supcon -> 'supc'."""
    return {"original": "orig", "masked": "mask", "supcon": "supc"}[loss_type]


configs = []
for top_n in TOP_N_COLORS:
    for lt in LOSS_TYPES:
        for dim in EMBED_DIMS:
            for chd in COLOR_HIDDEN_DIMS:
                for thd in TEXT_HIDDEN_DIMS:
                    name = (
                        f"clip_{top_n}c"
                        f"_{loss_tag(lt)}"
                        f"_dim{dim}"
                        f"_ch{dims_tag(chd)}"
                        f"_th{dims_tag(thd)}"
                    )
                    cfg = json.loads(json.dumps(BASE))
                    cfg["experiment_name"] = name
                    cfg["data"]["top_n_colors"] = top_n
                    cfg["training"]["loss_type"] = lt
                    cfg["model"] = {
                        "embed_dim": dim,
                        "color_hidden_dims": chd,
                        "text_hidden_dims": thd,
                    }
                    configs.append((name, cfg))

# ---------------------------------------------------------------------------
# Write files into configs/ subfolder
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
paths = []
for name, cfg in configs:
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    paths.append(path)

print(f"Generated {len(configs)} config files in {out_dir}")
for p in paths[:3]:
    print(f"  {os.path.basename(p)}")
print(f"  ... ({len(configs) - 6} more)")
for p in paths[-3:]:
    print(f"  {os.path.basename(p)}")
