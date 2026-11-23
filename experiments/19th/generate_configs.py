"""
Generate ColorCLIP configs for the 19th round — a working learning rate for plain SGD.

The 14th round found that the t->c weight (0.1, 1 or 10) gives the same model, while 0
gives a worse name-to-colour model. One reading is that AdamW's per-parameter rescaling
hides the weight; the other is that any small t->c push is enough. The 20th round tells
them apart by repeating the weight sweep with SGD, which does not rescale. SGD needs a
much larger learning rate than AdamW's 3e-4, so this round finds one first.

SGD keeps momentum 0.9, so the only difference from Adam is the rescaling (weight decay
is 0, as in the 14th round, so AdamW there is plain Adam).

Grid: lr {1e-2, 3e-2, 1e-1, 3e-1, 1} at t2c_weight 1, otherwise the 14th round's
clip_96c_proto_dim64_ch64_th256_t2c1_bs2048 (R@1 0.4852). Pick the lr closest to that
R@1 for the 20th round. Divergence at the high end is expected and is a result.
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
        "optimizer": "sgd",
        "momentum": 0.9,
        "weight_decay": 0.0,
        "loss_type": "prototype",
        "t2c_weight": 1.0,
        "early_stopping": {"metric": "mrr", "patience": 50, "min_delta": 0.001},
        "save_ranking": False,
        "save_scores": False,
    },
    "model": {"embed_dim": 64, "color_hidden_dims": [64], "text_hidden_dims": [256]},
}

LRS = [1e-2, 3e-2, 1e-1, 3e-1, 1.0]

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
for lr in LRS:
    cfg = json.loads(json.dumps(BASE))
    cfg["training"]["lr"] = lr
    name = f"clip_96c_proto_dim64_ch64_th256_t2c1_sgd_lr{lr:.0e}"
    cfg["experiment_name"] = name
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
print(f"{len(LRS)} configs written to {out_dir}")
