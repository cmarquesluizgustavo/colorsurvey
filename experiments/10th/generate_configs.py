"""
Generate ColorCLIP experiment configs for the 10th experiment round.

Two small, curated groups (not a grid sweep) — see research_plan.md for the
reasoning behind exactly these tuples:

  Group 1 — benchmark: the full color-count ladder for the two losses that
  matter (prototype and masked; original/supcon already shown redundant with
  masked/prototype respectively in earlier rounds), one shared architecture
  (dim64_ch16_th256 — ch16 was the best-performing color encoder size in the
  8th round's sweep). 4 color counts x 2 losses = 8 runs.

  Group 2 — investigation: does the loss really plateau at epoch 1 regardless
  of learning rate, and does the initial temperature matter? 96 colors only,
  prototype loss only, one parameter varied at a time. 2 lr values + 2
  temperature values = 4 runs.

Fixed: color_space=rgb, batch_size=2048, early_stopping metric=mrr with
patience=20 (was 50 in earlier rounds; best epoch is consistently reached by
~epoch 30, so patience 50 mostly burns wasted epochs).

save_ranking/save_scores are on for 14c/96c (cheap) and OFF for 797c/5363c:
the final evaluation still works there (ML/metrics.py now chunks it), but
those two flags additionally require materializing the full (N, K) score
matrix, which is several hundred MB to a few GB at these class counts — see
_save_eval_dump's docstring in ML/trainers/color_clip_trainer.py.
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
    "model": {"embed_dim": 64, "color_hidden_dims": [16], "text_hidden_dims": [256]},
}

LOSS_TAG = {"prototype": "proto", "masked": "mask"}

configs = []

# --- Group 1: benchmark ladder ---
for top_n in [14, 96, 797, 5363]:
    for loss_type in ["prototype", "masked"]:
        cfg = json.loads(json.dumps(BASE))
        cfg["data"]["top_n_colors"] = top_n
        cfg["training"]["loss_type"] = loss_type
        big = top_n in (797, 5363)
        cfg["training"]["save_ranking"] = not big
        cfg["training"]["save_scores"] = not big
        name = f"clip_{top_n}c_{LOSS_TAG[loss_type]}_dim64_ch16_th256"
        cfg["experiment_name"] = name
        configs.append((name, cfg))

# --- Group 2: investigation (96c, prototype only, one param at a time) ---
for lr in [0.0001, 0.001]:
    cfg = json.loads(json.dumps(BASE))
    cfg["data"]["top_n_colors"] = 96
    cfg["training"]["loss_type"] = "prototype"
    cfg["training"]["lr"] = lr
    cfg["training"]["save_ranking"] = True
    cfg["training"]["save_scores"] = True
    name = f"clip_96c_proto_dim64_ch16_th256_lr{lr:.0e}"
    cfg["experiment_name"] = name
    configs.append((name, cfg))

for temp in [0.02, 0.2]:
    cfg = json.loads(json.dumps(BASE))
    cfg["data"]["top_n_colors"] = 96
    cfg["training"]["loss_type"] = "prototype"
    cfg["training"]["temperature"] = temp
    cfg["training"]["save_ranking"] = True
    cfg["training"]["save_scores"] = True
    name = f"clip_96c_proto_dim64_ch16_th256_temp{temp}"
    cfg["experiment_name"] = name
    configs.append((name, cfg))

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
for name, cfg in configs:
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {path}")
print(f"\n{len(configs)} configs total")
