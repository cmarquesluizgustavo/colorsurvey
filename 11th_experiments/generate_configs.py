"""
Generate ColorCLIP configs for the 11th round — task 9: how small can the model be?

Accuracy is capped by the data (see research_plan.md notes 1-3), so the question is
cost, not accuracy. Two things were never tested: a color tower with no hidden layer
at all, and dim below 8. Note that 97% of the current parameters live in the text
tower, which we proved does nothing — so the default here is the minimal text tower.

Grid (96 colors, prototype loss, everything else fixed):
    embed_dim         : 4, 8, 16, 32, 64
    color_hidden_dims : [] (bare Linear(3,dim)), [4], [8], [16]
    text_hidden_dims  : [] (bare Linear(vocab,dim))
  + controls with text_hidden_dims=[256] at each dim, to confirm the text tower
    stays irrelevant even when the color tower is tiny.

`"color_hidden_dims": []` needs no library change: ColorEncoder only raises on None,
and an empty list just skips the hidden-layer loop. Names use ch0/th0 for the empty
case so cluster/generate_experiments_txt.py's `_ch([\\dx]+)` regex still parses.
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
        "top_n_colors": 96,
    },
    "training": {
        "epochs": 100,
        "temperature": 0.07,
        "lr": 0.0003,
        "weight_decay": 0.0,
        "loss_type": "prototype",
        "early_stopping": {"metric": "mrr", "patience": 20, "min_delta": 0.001},
        "save_ranking": False,
        "save_scores": True,   # needed for calibration / prior sweeps (task 17)
    },
}

DIMS = [4, 8, 16, 32, 64]
COLOR_HIDDEN = [[], [4], [8], [16]]
TEXT_HIDDEN = [[]]            # minimal text tower for the main grid
TEXT_CONTROL = [256]          # one control per dim, with ch=[16]
# The same sweep is run for the original CLIP loss, so the size/accuracy curve can be
# compared between the (N,K) prototype formulation and the classic (N,N) one.
LOSSES = {"proto": "prototype", "orig": "original", "mask": "masked", "supc": "supcon"}

tag = lambda dims: "x".join(str(d) for d in dims) if dims else "0"

configs = []
for loss_tag, loss_type in LOSSES.items():
    for dim in DIMS:
        for ch in COLOR_HIDDEN:
            for th in TEXT_HIDDEN:
                cfg = json.loads(json.dumps(BASE))
                cfg["training"]["loss_type"] = loss_type
                cfg["model"] = {"embed_dim": dim, "color_hidden_dims": ch,
                                "text_hidden_dims": th}
                name = f"clip_96c_{loss_tag}_dim{dim}_ch{tag(ch)}_th{tag(th)}"
                cfg["experiment_name"] = name
                configs.append((name, cfg))
        # control: same color tower size, fat text tower
        cfg = json.loads(json.dumps(BASE))
        cfg["training"]["loss_type"] = loss_type
        cfg["model"] = {"embed_dim": dim, "color_hidden_dims": [16],
                        "text_hidden_dims": TEXT_CONTROL}
        name = f"clip_96c_{loss_tag}_dim{dim}_ch16_th{tag(TEXT_CONTROL)}"
        cfg["experiment_name"] = name
        configs.append((name, cfg))

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
for name, cfg in configs:
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
print(f"{len(configs)} configs written to {out_dir}")
