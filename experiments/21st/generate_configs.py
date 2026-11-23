"""
Generate ColorCLIP configs for the 21st round — the 20th round's optimizer test at 797 names.

The 20th round (96 names) found that SGD, like AdamW, gives the same model at t2c_weight
0.1, 1 and 10 and a worse name-to-colour model at 0, so AdamW's per-parameter rescaling
does not explain the weight's small effect. At 797 names the text side has far more to
learn (many rare, close names), so the t->c term may matter more there.

Earlier 797-name prototype runs all use t2c_weight 0.1, so the AdamW side is run here too.

Grid: optimizer {adamw 3e-4, sgd 0.03 with momentum 0.9} x t2c_weight {0, 0.1, 1, 10}
= 8 runs, otherwise the 20th round (batch 2048, dim64_ch64_th256). SGD's rate is the 19th
round's (96 names); if SGD at weight 1 falls short of AdamW's R@1 here, search it again.
R@1 sits at its ceiling (lookup table 0.4048); read the result with
annexes/people_vs_model, keeping rare names apart (median ~84 test answers per name).
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
        "top_n_colors": 797,
    },
    "training": {
        "epochs": 100,
        "temperature": 0.07,
        "weight_decay": 0.0,
        "loss_type": "prototype",
        "early_stopping": {"metric": "mrr", "patience": 50, "min_delta": 0.001},
        "save_ranking": False,
        "save_scores": False,
    },
    "model": {"embed_dim": 64, "color_hidden_dims": [64], "text_hidden_dims": [256]},
}

OPTIMIZERS = {"adamw": {"optimizer": "adamw", "lr": 3e-4},
              "sgd": {"optimizer": "sgd", "momentum": 0.9, "lr": 0.03}}
WEIGHTS = {"0": 0.0, "01": 0.1, "1": 1.0, "10": 10.0}

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
n = 0
for opt_tag, opt in OPTIMIZERS.items():
    for w_tag, w in WEIGHTS.items():
        cfg = json.loads(json.dumps(BASE))
        cfg["training"].update(opt)
        cfg["training"]["t2c_weight"] = w
        name = f"clip_797c_proto_dim64_ch64_th256_t2c{w_tag}_{opt_tag}"
        cfg["experiment_name"] = name
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        n += 1
print(f"{n} configs written to {out_dir}")
