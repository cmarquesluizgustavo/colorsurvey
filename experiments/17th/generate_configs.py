"""
Generate ColorCLIP configs for the 17th round — the learning rate above 3e-3.

The 12th round swept lr 1e-5 ... 3e-3 for all four losses (96c, dim32_ch16_th0) and found
nothing: prototype R@1 moved by at most ~0.007 and silhouette by ~0.007 with no direction.
Nothing above 3e-3 has ever been run, so "lr does not matter" is only known below it.

Why it stopped there: the optimizer is AdamW, where the learning rate is roughly the size
of each weight's step regardless of gradient size. At 0.1 every weight moves ~0.1 per
step — the scale of the weights themselves — so divergence is the expected outcome. That
is convention, not a measurement, and the project already has one divergence at the
default (masked, 3e-4, 12th round). This round measures where training breaks.

Grid: lr {1e-2, 3e-2, 1e-1, 5e-1} x 4 losses = 16 runs, otherwise identical to the 12th
round (96c, dim32_ch16_th0, epochs 300, patience 50) so the two rounds form one curve.

Expect several runs to diverge. collect_results.py marks those `Status: diverged`; that
is a result here, not a failure — record at which lr each loss breaks.
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
        "epochs": 300,
        "temperature": 0.07,
        "weight_decay": 0.0,
        "early_stopping": {"metric": "mrr", "patience": 50, "min_delta": 0.001},
        "save_ranking": False,
        "save_scores": False,
    },
    "model": {"embed_dim": 32, "color_hidden_dims": [16], "text_hidden_dims": []},
}

LRS = [1e-2, 3e-2, 1e-1, 5e-1]
LOSSES = {"proto": "prototype", "orig": "original", "mask": "masked", "supc": "supcon"}

configs = []
for loss_tag, loss_type in LOSSES.items():
    for lr in LRS:
        cfg = json.loads(json.dumps(BASE))
        cfg["training"]["lr"] = lr
        cfg["training"]["loss_type"] = loss_type
        name = f"clip_96c_{loss_tag}_dim32_ch16_th0_lr{lr:.0e}"
        cfg["experiment_name"] = name
        configs.append((name, cfg))

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
for name, cfg in configs:
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
print(f"{len(configs)} configs written to {out_dir}")
