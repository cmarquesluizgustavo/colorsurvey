"""
Generate ColorCLIP configs for the 22nd round — name-to-colour alone (c2t_weight = 0).

Mirror image of t2c_weight = 0 from the 14th and 20th rounds:
  - t2c_weight = 0 keeps only colour->name and breaks name->colour (centres shift 21-25 RGB units).
  - c2t_weight = 0 keeps only name->colour (loss_t2c) to test whether colour->name accuracy
    and probabilities degrade when each name's prior frequency is no longer compared across names.

Grid: optimizer {adamw 3e-4, sgd 0.03 with momentum 0.9} at 96 names, batch 2048,
dim64_ch64_th256, c2t_weight 0.0, t2c_weight 1.0 (2 runs).
Early stopping is disabled (patience 0, 80 epochs) because early-stopping monitors MRR (c->t),
which could peak at epoch 1 before the t->c-only run settles.
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
        "epochs": 80,
        "temperature": 0.07,
        "weight_decay": 0.0,
        "loss_type": "prototype",
        "c2t_weight": 0.0,
        "t2c_weight": 1.0,
        "early_stopping": {"metric": "mrr", "patience": 0, "min_delta": 0.001},
        "save_ranking": False,
        "save_scores": False,
    },
    "model": {"embed_dim": 64, "color_hidden_dims": [64], "text_hidden_dims": [256]},
}

OPTIMIZERS = {
    "adamw": {"optimizer": "adamw", "lr": 3e-4},
    "sgd": {"optimizer": "sgd", "momentum": 0.9, "lr": 0.03},
}

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
for opt_tag, opt in OPTIMIZERS.items():
    cfg = json.loads(json.dumps(BASE))
    cfg["training"].update(opt)
    name = f"clip_96c_proto_dim64_ch64_th256_c2t0_{opt_tag}"
    cfg["experiment_name"] = name
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
print(f"{len(OPTIMIZERS)} configs written to {out_dir}")
