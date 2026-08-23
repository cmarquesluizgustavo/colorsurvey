"""
Generate ColorCLIP configs for the 12th round — the learning-rate question.

Why this round exists: across ~2,000 configs in 11 rounds, lr was 3e-4 in all but two
runs. The floor ever tested is 1e-4 (once). The 5th round swept 3e-4 / 1e-3 / 3e-3 —
upward only, and in a much weaker era. So everything *below* the default is unexplored.

Two things make it worth closing properly:
  - prototype reaches ~98% of its final R@1 in a single epoch, which is what "lr too
    high, jumped straight into a mediocre basin" looks like. A slower descent is the
    obvious test of whether something better than ~0.485 exists.
  - original starts at 0.024 and climbs for ~40 epochs, so it is the loss where lr
    should genuinely matter.

Design note that the sweep depends on: with the usual patience=20, a low-lr run would
early-stop before converging, confounding "this lr is bad" with "we cut it off". So
epochs=300 and patience=50 here — the low end must be given room to actually finish.

Grid: lr {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3} x 4 losses = 24 runs, 96 colors,
one cheap near-best architecture (dim32_ch16_th0, 2,720 params, R@1 0.4792).
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
        "lr": 0.0003,
        "weight_decay": 0.0,
        # patience 50 (not the usual 20): a 1e-5 run needs room to converge, otherwise
        # early stopping would be measured instead of the learning rate.
        "early_stopping": {"metric": "mrr", "patience": 50, "min_delta": 0.001},
        "save_ranking": False,
        "save_scores": True,
    },
    "model": {"embed_dim": 32, "color_hidden_dims": [16], "text_hidden_dims": []},
}

LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
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
