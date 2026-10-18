"""
Generate ColorCLIP configs for the 18th round — does a pixel tolerance in the t->c term help?

The idea. In the text->color direction a text picks a pixel out of the batch, and the
pick is right only if it is the text's own pair. With `t2c_eps` (ML/losses/clip_loss.py,
BaseCLIPLoss), a pick that is wrong but lies within eps of the right pixel on every
channel (L-inf box, 0-255 grid; 1, 2 or 3 channels may each differ by up to eps) is
accepted as right. color->text is unchanged.

The maths. For a text, the model gives every batch pixel n a probability p(n) of being the
pick (softmax over the batch). Its right pixels R are the exact pairs: one pixel for
original and masked, every pixel with that name for prototype and supcon.
  strict (eps 0):  loss = -(1/|R|) * sum over r in R of log p(r)
  accept (eps):    loss = -(1/|R|) * sum over r in R of log A(r),
                   A(r) = p(r) + p of every batch pixel within eps of r
So probability on a look-alike counts as probability on the right pixel. Example: P1 is
the pair, P2 is within eps of P1, the model puts 0.5 on P1 and 0.3 on P2: strict loss
-log 0.5 = 0.69, accept loss -log 0.8 = 0.22. With no look-alike in the batch, A(r) = p(r)
and the two are equal. Masked keeps masking same-name pixels outside the eps box.
A pixel that is a look-alike of two right pixels counts for both.

What eps means (measured on the 96c data):
  - Perceptually: OKLab dE of an eps step has median 0.003 / 0.013 / 0.027 / 0.042 at
    eps 1 / 5 / 10 / 16; ~0.02 is just noticeable. 1-5 is invisible, 10 barely visible,
    16 about two noticeable steps.
  - In a 2048-row batch a row has 0.003 / 0.15 / 1.1 / 4.0 neighbours on average, so
    eps 1 is a near no-op control. Prototype positives per name grow 21 -> 21 / 24 / 35 /
    65. The neighbour count scales with batch size: batch and eps are the same knob.

Expectation on record: c->t R@1 will not move for prototype — t->c is at its data floor
(note 5) and its weight did not change R@1 in the 14th round, even at 0. Where an effect
can show is geometry (silhouette), and possibly the label-blind losses (original, supcon),
whose R@1 is far below prototype's.

Setup: dim64, ch[64], th[256], lr 3e-4, starting temperature 0.07, weight decay 0,
batch 2048, 100 epochs, early stopping on MRR with patience 20, t2c_weight 1, seed 42,
RGB input. Each setting is the best one the earlier rounds support without dispute
(annexes/geometry/, research_plan.md notes 8-9):
  - Colour tower [64] instead of the reference [16]: silhouette better in 46/46 matched
    pairs at 96 names and 30/30 at 797 (+0.017 / +0.033), for every loss. R@1 cost at
    most 0.3 points (masked -0.0018 / -0.0028); prototype gains (0.4858 vs 0.4838). At 14
    names the tower makes no difference. [32,128] separates further but costs masked
    0.5-0.9 R@1 points and has one prototype pair per name count: not undisputed.
    dim64_ch64_th256 also has earlier runs for all four losses at 14 / 96 / 797 names.
  - lr 3e-4, temperature 0.07: lr 1e-5..3e-3 (12th) and temperature 0.02..0.2 (10th)
    move R@1 < 0.007. Temperature 0.2 had the best silhouette, but on one run per value;
    the 16th round tests it, so it is not settled.
  - t2c_weight 1, not 0.1: 0.1 had the best silhouette in the 14th round (+0.006, single
    seed), but this round tests a change to the t->c term, and down-weighting that term
    would shrink the effect being measured.
  - batch 2048: best geometry of 96 / 512 / 2048 at every t->c weight (14th); larger was
    never run. The eps effect also needs a large batch.
  - patience 20: patience 50 gives the same scores (8th vs 10th round, same config:
    R@1 within 0.0004 at 14 / 96 / 797 names).
  - eps 0 is re-run inside the round, so every comparison is internal.

Grid: names {14, 96, 797} x losses {prototype, supcon, original, masked}
      x eps {0, 1, 5, 10, 16} = 60 runs.

How to read this round. metrics.csv has one row per epoch; the t->c columns are averages
over that epoch's training batches, for every loss:
  loss_t2c                       the t->c loss trained on (accept rule at eps > 0)
  loss_t2c_strict                the same probabilities scored with the eps 0 rule (exact
                                 pairs only); measured, never trained on. Equals loss_t2c
                                 at eps 0, and loss_t2c of every earlier round.
  t2c_positives                  pixels accepted per text, on average (exact pairs plus
                                 look-alikes). Depends on the data, eps and batch size,
                                 not on the model: it confirms eps did something.
  loss_t2c_above_uniform         loss_t2c minus log(number of exact pairs), averaged over
  loss_t2c_strict_above_uniform  texts -- the score of a model that splits its pick evenly
                                 over the exact pairs. Not reachable in practice (note 5);
                                 under the accept rule it is the best score only while no
                                 pixel is a look-alike of two right pixels.
Three comparisons, each within one loss and one name count:
  1. loss_t2c_strict, eps 0 run vs eps run, epoch by epoch: did training under the accept
     rule make the model better, worse or faster at finding the exact pair? eps 0 is the
     control.
  2. loss_t2c vs loss_t2c_strict within an eps run, epoch by epoch: how much the rule
     forgives. A growing gap means the picks move onto look-alikes.
  3. How fast loss_t2c falls: whether gradient descent has an easier time with the
     accept rule.
Alongside: R@1 and the other retrieval columns, measured each epoch on 5,000 test
pixels, and silhouette, recomputed from the checkpoints.

Things to know before analysing:
  - The t->c columns come from training pixels (80% of the data, learned from), not test
    pixels (20%, held back). They show how the model does on pixels it is shown, not on
    new ones; no test-side t->c measure exists yet (research_plan.md task 14).
  - Single seed, as every reference run; inherits the test-set early stopping of note 10.
  - save_ranking/save_scores off: silhouette is recomputed from the checkpoints.
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
        "early_stopping": {"metric": "mrr", "patience": 20, "min_delta": 0.001},
        "save_ranking": False,
        "save_scores": False,
    },
    "model": {"embed_dim": 64, "color_hidden_dims": [64], "text_hidden_dims": [256]},
}

COLORS = [14, 96, 797]
EPS = [1, 5, 10, 16]
LOSSES = {"proto": "prototype", "supc": "supcon", "orig": "original", "mask": "masked"}

configs = []
for colors in COLORS:
    for loss_tag, loss_type in LOSSES.items():
        for eps in [0] + EPS:
            cfg = json.loads(json.dumps(BASE))
            cfg["data"]["top_n_colors"] = colors
            cfg["training"]["loss_type"] = loss_type
            cfg["training"]["t2c_eps"] = eps
            name = f"clip_{colors}c_{loss_tag}_dim64_ch64_th256_eps{eps}"
            cfg["experiment_name"] = name
            configs.append((name, cfg))

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(out_dir, exist_ok=True)
for name, cfg in configs:
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
assert len({n for n, _ in configs}) == len(configs)
print(f"{len(configs)} configs written to {out_dir}")
