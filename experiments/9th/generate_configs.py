"""
Generate ColorCLIP experiment configs for the 9th experiment round.

This round introduces the new prototype loss (loss_type="prototype"): the batch
logit matrix is (N, K) — rows are batch colors, columns are the K fixed class
prototypes. color->text is plain K-way classification; text->color pulls each
present prototype toward its samples (SupCon soft targets). All K classes are
present every batch, which directly targets the long-tail problem.

Grid (deliberately small — a few entries per color count to kick the tires):
    top_n_colors       : [14, 96]
    loss_type          : [prototype]
    embed_dim          : [64, 128]
    color_hidden_dims  : [[64], [32, 128]]
    text_hidden_dims   : [[256]]

Fixed (same as 8th round):
    color_space    = rgb
    lr             = 0.0003
    weight_decay   = 0.0
    temperature    = 0.07
    balance        = none
    batch_size     = 2048
    epochs         = 100
    vocab_size     = auto (uses all unique tokens in the dataset)
    early_stopping : metric=mrr, patience=50, min_delta=0.001

Total: 2 × 1 × 2 × 2 × 1 = 8 configs
"""
import json
import os

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
TOP_N_COLORS       = [14, 96]
LOSS_TYPES         = ["prototype"]
EMBED_DIMS         = [64, 128]
COLOR_HIDDEN_DIMS  = [[64], [32, 128]]
TEXT_HIDDEN_DIMS   = [[256]]

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
        "early_stopping": {
            "metric": "mrr",
            "patience": 50,
            "min_delta": 0.001,
        },
    },
}


def dims_tag(dims):
    """[4,32] -> '4x32', [8] -> '8'."""
    return "x".join(str(d) for d in dims)


def loss_tag(loss_type):
    """original -> 'orig', masked -> 'mask', supcon -> 'supc', prototype -> 'proto'."""
    return {
        "original": "orig",
        "masked": "mask",
        "supcon": "supc",
        "prototype": "proto",
    }[loss_type]


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
for p in paths:
    print(f"  {os.path.basename(p)}")
