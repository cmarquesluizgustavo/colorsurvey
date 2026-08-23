"""
Generate ColorCLIP configs for the 13th experiment round:
Head-Color Ablation Ladder (training without dominant head classes).

Grid Option 2: Dual Architecture (Best Overall + Best Focused) across 4 losses.
- 9 subset conditions:
    - 14c ladder: no3 (ranks 4..14, K=11)
    - 96c ladder: no3 (ranks 4..96, K=93), no14 (ranks 15..96, K=82)
    - 797c ladder: no3 (ranks 4..797, K=794), no14 (ranks 15..797, K=783), no96 (ranks 97..797, K=701)
    - 5363c ladder: no3 (ranks 4..5363, K=5360), no14 (ranks 15..5363, K=5349), no96 (ranks 97..5363, K=5267)
- 4 losses: prototype, masked, original, supcon
- 2 architectures:
    1. dim64_ch16_th256 (Best Overall capacity)
    2. dim32_ch16_th0   (Best Focused / Streamlined)
Total = 9 x 4 x 2 = 72 configs.
"""
import glob
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
    },
    "training": {
        "epochs": 100,
        "temperature": 0.07,
        "lr": 0.0003,
        "weight_decay": 0.0,
        "early_stopping": {
            "metric": "mrr",
            "patience": 20,
            "min_delta": 0.001,
        },
    },
}

# 9 ablation conditions: (top_n, exclude_top_n, subset_tag)
CONDITIONS = [
    # 14c ladder
    (14, 3, "no3"),
    # 96c ladder
    (96, 3, "no3"),
    (96, 14, "no14"),
    # 797c ladder
    (797, 3, "no3"),
    (797, 14, "no14"),
    (797, 96, "no96"),
    # 5363c ladder
    (5363, 3, "no3"),
    (5363, 14, "no14"),
    (5363, 96, "no96"),
]

# 4 losses
LOSSES = {
    "proto": "prototype",
    "mask": "masked",
    "orig": "original",
    "supc": "supcon",
}

# 2 architectures: Best Overall + Best Focused
ARCHS = [
    {
        "name": "dim64_ch16_th256",
        "embed_dim": 64,
        "color_hidden_dims": [16],
        "text_hidden_dims": [256],
    },
    {
        "name": "dim32_ch16_th0",
        "embed_dim": 32,
        "color_hidden_dims": [16],
        "text_hidden_dims": [],
    },
]


def generate():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    os.makedirs(out_dir, exist_ok=True)

    # Clean old configs
    for old in glob.glob(os.path.join(out_dir, "*.json")):
        os.remove(old)

    configs = []
    for top_n, exclude, subset in CONDITIONS:
        for loss_tag, loss_type in LOSSES.items():
            for arch in ARCHS:
                cfg = json.loads(json.dumps(BASE))
                cfg["data"]["top_n_colors"] = top_n
                cfg["data"]["exclude_top_n"] = exclude
                cfg["training"]["loss_type"] = loss_type

                big = top_n in (797, 5363)
                cfg["training"]["save_ranking"] = not big
                cfg["training"]["save_scores"] = not big

                cfg["model"] = {
                    "embed_dim": arch["embed_dim"],
                    "color_hidden_dims": arch["color_hidden_dims"],
                    "text_hidden_dims": arch["text_hidden_dims"],
                }

                arch_tag = arch["name"]
                name = f"clip_{top_n}c_{subset}_{loss_tag}_{arch_tag}"
                cfg["experiment_name"] = name
                configs.append((name, cfg))

    for name, cfg in configs:
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)

    print(f"Wrote {len(configs)} configs to {out_dir}")
    return configs


if __name__ == "__main__":
    generate()
