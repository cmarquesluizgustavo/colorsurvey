"""
Generate ColorCLIP configs for the 14th round — does the t->c term matter at all?

Two open questions, one grid.

1. t2c_weight. Dialing it 1.0 -> 0.1 moved R@1 by +0.0004 (0.4872 -> 0.4876), which
   we read as "t->c is quiet". That reading is wrong: measured at the trained
   checkpoint, t->c's gradient norm is 1.67x c->t's (1.34 vs 0.80), with the two
   gradients only partly aligned (cos +0.33). So the insensitivity is unexplained.
   Sweeping 0.0 / 0.1 / 1.0 / 10.0 settles it: if a 100x swing (and switching the
   term off entirely) still moves nothing, the cause is AdamW's per-parameter
   rescaling, not the term being unimportant.

2. batch_size. t->c is retrieval among the N rows of the batch, so its difficulty
   is set by N: at N=2048 the "blue" prototype must find ~20 blue-labelled rows
   among ~190 blue-ish ones. Shrinking the batch toward K=96 makes it closer to the
   1-of-K problem that c->t already solves. Note the loss VALUE is not comparable
   across batch sizes (different N = different task); only R@1 is.
   Plain random sampling is kept -- one-row-per-class would need the balanced
   sampler, which independently craters R@1 (0.188), confounding the test.

Grid: t2c_weight {0.0, 0.1, 1.0, 10.0} x batch_size {96, 512, 2048} = 12 runs.
Architecture is fixed at dim64_ch64_th256 so the 1.0/2048 and 0.1/2048 cells land
directly on top of the two existing 9th-round runs.
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
        "early_stopping": {"metric": "mrr", "patience": 50, "min_delta": 0.001},
        "save_ranking": False,
        "save_scores": False,
    },
    "model": {"embed_dim": 64, "color_hidden_dims": [64], "text_hidden_dims": [256]},
}

WEIGHTS = {"0": 0.0, "01": 0.1, "1": 1.0, "10": 10.0}
BATCHES = [96, 512, 2048]

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
n = 0
for tag, w in WEIGHTS.items():
    for bs in BATCHES:
        cfg = json.loads(json.dumps(BASE))
        cfg["data"]["batch_size"] = bs
        cfg["training"]["t2c_weight"] = w
        name = f"clip_96c_proto_dim64_ch64_th256_t2c{tag}_bs{bs}"
        cfg["experiment_name"] = name
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        n += 1
print(f"{n} configs written to {out_dir}")
