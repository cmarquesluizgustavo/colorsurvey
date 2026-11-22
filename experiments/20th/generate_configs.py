"""
Generate ColorCLIP configs for the 20th round — does AdamW hide the t->c weight?

In the 14th round, t2c_weight 0.1, 1 and 10 gave the same model (R@1 ~0.485, name-to-
colour difference from people ~0.21, name centres 3-5 RGB units off), while 0 gave a
worse name-to-colour model (difference 0.32-0.36, centres 18-25 units off). Two readings:

  A. AdamW rescales each parameter's step, which hides how the push is split between
     the two terms.
  B. Any small t->c push is enough; the optimizer plays no part.

This round repeats the weight sweep with SGD (momentum 0.9, no rescaling). If the SGD
runs at 0.1 and 10 differ in name-to-colour, A holds; if they match, B holds. Read the
result with annexes/people_vs_model (name-to-colour difference, centre), not R@1, which
sits at its ceiling.

Grid: t2c_weight {0, 0.1, 1, 10} with SGD = 4 runs, at the 14th round's batch 2048 and
architecture. The AdamW side is the 14th round's four batch-2048 runs, not re-run.
SGD_LR comes from the 19th round; set it before running.
"""
import json
import os

SGD_LR = 0.03  # 19th round: R@1 0.4855, the best of 0.01-0.3; 1 diverged

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
        "optimizer": "sgd",
        "momentum": 0.9,
        "lr": SGD_LR,
        "weight_decay": 0.0,
        "loss_type": "prototype",
        "early_stopping": {"metric": "mrr", "patience": 50, "min_delta": 0.001},
        "save_ranking": False,
        "save_scores": False,
    },
    "model": {"embed_dim": 64, "color_hidden_dims": [64], "text_hidden_dims": [256]},
}

assert SGD_LR is not None, "set SGD_LR from the 19th round first"
WEIGHTS = {"0": 0.0, "01": 0.1, "1": 1.0, "10": 10.0}

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
for tag, w in WEIGHTS.items():
    cfg = json.loads(json.dumps(BASE))
    cfg["training"]["t2c_weight"] = w
    name = f"clip_96c_proto_dim64_ch64_th256_t2c{tag}_sgd"
    cfg["experiment_name"] = name
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
print(f"{len(WEIGHTS)} configs written to {out_dir}")
