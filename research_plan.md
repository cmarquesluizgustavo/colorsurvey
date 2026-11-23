# Research plan

Reference setup: 96 colors, `dim64_ch16_th256`, full test set.

## Where we are

The `prototype` loss (each color scored against the fixed list of K names) is the best model
the project has produced — and it sits **at the data ceiling**. It gets ~0% of the rarest
classes right, but so does everything else, including perfect memorization. The remaining
questions are about cost, generalization, and how we score — not about accuracy.

| model | overall R@1 | per-class avg R@1 | rarest-5 R@1 |
|---|---|---|---|
| prototype | **0.485** | 0.129 | 0.000 |
| prototype + prior correction (a=1) | 0.177 | 0.256 | **0.313** |
| masked | 0.437 | 0.198 | 0.111 |
| nearest RGB centroid (no training) | 0.174 | 0.222 | 0.230 |

## Results

One row per round; the notes below interpret these. R@1 figures here come from the
checkpoints (full test set, or `annexes/geometry/`), not from the `R_at_1` column of
`experiment_results.csv`, which is the *last* training epoch scored on only 5,000 samples.

| round | question | grid (runs) | answer |
|---|---|---|---|
| 1st | Baseline classifier; does rebalancing help? | MLP light/medium/heavy × 21, 329 names × 4 sampling schemes (17) | Best test accuracy 0.68 (21 names), 0.43 (329). Pre-CLIP. |
| 2nd | Balanced vs unbalanced training | MLP medium × 15, 129 names (4) | 0.75 (15), 0.47 (129). |
| 3rd | First CLIP model | 15 names, dim 4 (12) | R@1 ~0.64. |
| 4th | MLP grid | 15, 129 names × balanced/unbalanced (64) | Same as the 2nd: 0.75 / 0.47. |
| 5th | Colour space and optimiser settings | RGB vs OKLCH × 15, 129 names × dim × temperature × lr × weight decay (432) | OKLCH = RGB (−0.0001 R@1, paired). |
| 6th | Architecture sweep, supcon | 129 / 1184 / 4214 names (older name lists) × dim 8–32 × towers (144) | Superseded by the 7th and 8th. |
| 7th | Three losses at every name count | original / masked / supcon × 14 / 96 / 797 / 5363 × dim 8–32 × small towers (575) | Source of the small-tower cells in the geometry annex. |
| 8th | Same, larger models | dim 64–256 × towers up to [32,128] × text towers (551 of 1080 harvested; no 5363) | Capacity does not move R@1; it does move geometry (note 8). |
| 9th | First prototype loss | 14, 96 names, a few archs (13, local, in `experiments/runs/`) | Prototype gives the best overall R@1. |
| 10th | Prototype vs masked at every name count; lr / temperature check | dim64 ch[16] th[256] × 4 name counts × 2 losses, + 4 hyperparameter runs (12) | Prototype R@1 0.484 at 96 names; its lead over masked grows with name count. |
| 11th | How small can the model be? | 96 names, dim 4–64 × colour tower [] / [4] / [8] / [16] × 4 losses (100; 1 incomplete) | ~600 parameters land within 1.4 R@1 points of 34k. dim4 is worse than raw RGB on geometry. |
| 12th | Learning rate, 1e-5 to 3e-3 | × 4 losses, 96 names, dim32 ch[16] th[] (24; 1 diverged) | lr moves neither R@1 nor geometry. First fixed-architecture loss ranking: prototype 0.48, masked 0.44, original / supcon 0.18. |
| 13th | Training without the most common names | drop top 3 / 14 / 96 names × 4 losses × 2 archs (72) | The remaining names are recognised more often (prototype +0.08 to +0.25 R@1 on the shared names), but only because the common names stop winning the argmax: blocking them in the full model gives prototype the same per-name R@1 (±0.002). Masked gains +0.02–0.03 from training itself; original / supcon barely move. Tail past rank ~800 stays at 0. |
| 14th | Does the t→c term matter? | t→c weight 0 / 0.1 / 1 / 10 × batch 96 / 512 / 2048 (12) | R@1 unchanged, even with t→c off. Geometry best at 0.1; larger batches better (note 5). |
| 15th | Fill the geometry grid | prototype wide towers × 4 name counts; 5363-name wide towers; 14-name tiny towers (26) | A wider colour tower separates names better — significant at 96 and 797 names (note 8). |
| 16th | Push geometry further; measure seed noise | prototype, 96 / 797 names, towers [32,128] / [32,128,128] / [64,256] × start temperature 0.07 / 0.2 / 0.5 × 3 seeds (54) | Written, not run. |
| 17th | Learning rate above 3e-3 | lr 1e-2 / 3e-2 / 1e-1 / 5e-1 × 4 losses, as the 12th (16) | Written, not run. |
| 18th | Does a pixel tolerance in t→c help? (`t2c_eps`: a t→c pick within ε per channel of the right pixel is accepted) | dim64 ch[64] th[256] (best undisputed settings, see the round's docstring); 14 / 96 / 797 names × 4 losses × ε 0 / 1 / 5 / 10 / 16 (60) | Written, not run. In-batch neighbours per row at batch 2048: 0.003 / 0.15 / 1.1 / 4.0 — ε 1 is a control. How to read it: the round's `generate_configs.py`. |
| 19th | Find a working learning rate for plain SGD | 96 names, dim64 ch[64] th[256], SGD momentum 0.9 across lr 0.01 to 1.0 (5) | lr 0.03 matches AdamW (`R@1 = 0.4855`); 1.0 collapsed at epoch 1. |
| 20th | Does AdamW hide the t→c weight? (96 names) | 96 names, dim64 ch[64] th[256], SGD lr 0.03 across t→c weight 0, 0.1, 1, 10 (4) | No: plain SGD matches Round 14 AdamW — weight 0 drifts over 20 RGB units off people-driven centres, while 0.1, 1, and 10 align within about 4 units. |
| 21st | Same test at 797 names | 797 names, AdamW and plain SGD across t→c weight 0, 0.1, 1, 10 (8) | Plain SGD matches AdamW across common and rare names: weight 0 drifts 22–23 units, while 0.1, 1, and 10 converge (`0.1` best on silhouette and log-loss). |
| 22nd | Name-to-colour alone (`c2t_weight = 0`) | 96 names, AdamW and plain SGD at c→t weight 0, t→c weight 1 (2) | Colour regions (3.4–3.5 RGB units) and silhouette stay intact, while `R@1` collapses to 0.042–0.083 (0% on green, blue, and red). |

## Important notes

1. **We are at the ceiling.** A lookup table that memorizes the most common name per RGB bin
   (train split only) scores 0.7638 / 0.4806 / 0.4048 at 14/96/797 colors. Our model:
   **0.7687 / 0.4840 / 0.4071** — equal or better everywhere. Same on the tail: best lookup
   class-oriented R@1 at 96c is 0.1332, ours 0.1317. Perfect memorization does no better.
   RGB → single name is saturated.

2. **The task is genuinely ambiguous.** Measured on pixels seen by exactly two people — one
   repetition, so every column below comes from the same clean population:

   | labels | distinct RGB | seen exactly 2× | % of all RGB | disagree | agreement | chance | κ |
   |---|---|---|---|---|---|---|---|
   | 14 | 1,402,278 | 67,095 | 4.8% | 27.9% | 0.721 | 0.133 | 0.68 |
   | 96 | 2,235,900 | 151,762 | 6.8% | **65.5%** | 0.345 | 0.054 | 0.31 |
   | 797 | 2,626,125 | 205,808 | 7.8% | 75.0% | 0.250 | 0.037 | 0.22 |
   | 5363 | 2,752,868 | 225,624 | 8.2% | 77.3% | 0.227 | 0.033 | 0.20 |

   *agreement* = the two people gave the same name (= 1 − disagree). *chance* = what two people
   would score by ignoring the pixel and answering from name popularity alone.

   *κ (Cohen's kappa)* = agreement with those lucky matches removed:
   (observed − chance) / (1 − chance). 0 = no better than guessing, 1 = perfect. At 96 colours:
   (0.345 − 0.054) / (1 − 0.054) = **0.31**, "fair" on the Landis–Koch scale; 14 colours gives
   0.68, "substantial". κ matters most at 14 colours, where chance is high enough (0.133) to
   flatter raw agreement; above that chance is tiny and the two numbers say the same thing.

   How often a pixel repeats at all (share of all distinct RGB values):

   | labels | seen 2× | seen 3× | seen 4× | seen 5× |
   |---|---|---|---|---|
   | 14 | 67,095 (4.78%) | 2,485 (0.18%) | 65 (0.00%) | 3 (0.00%) |
   | 96 | 151,762 (6.79%) | 7,540 (0.34%) | 271 (0.01%) | 7 (0.00%) |
   | 797 | 205,808 (7.84%) | 11,740 (0.45%) | 499 (0.02%) | 13 (0.00%) |
   | 5363 | 225,624 (8.20%) | 13,512 (0.49%) | 594 (0.02%) | 16 (0.00%) |

   Pixels seen twice carry ~95% of all repeated pixels, which is why the analysis rests on them.
   Agreement is stable if the 3× and 4×+ pixels are included instead (0.345 / 0.359 / 0.358),
   and repeated pixels carry the same name mix as the full data. Agreement is a *lower bound* on
   the ceiling (predicting the mode beats matching a random draw) — the lookup table in note 1
   is the real ceiling evidence.

3. **The tail is unreachable by argmax, provably.** At 96c with 8 bins/axis, only **28 of 96**
   names are ever the most common in any bin — the rest are always locally outnumbered, so
   their recall is 0 by construction. All 96 become reachable only at 64 bins/axis, where
   accuracy drops to 0.430. Making rare names reachable *costs* aggregate accuracy: that
   trade is the geometry of the data, not a flaw in the loss.

4. **The name vectors were never the problem.** Giving every name a free, unglued vector
   fully de-crowded them (NN cosine 0.887 → 0.291) and changed no metric (0.484 vs 0.485).
   The confusion is in the pixels.

5. **The t→c term is at its floor, not dead weight** *(corrected 2026-08-17; the old reading
   was wrong)*. We logged `loss_t2c_above_floor` against `mean log(m_k)` = 2.3 nats, saw a
   2.98-nat gap, and concluded t→c was failing. That bound assumes the "blue" prototype could
   single out exactly its ~20 blue-*labelled* rows — impossible, since the ~190 near-identical
   blue-ish rows look the same to it. The real floor is `q(n|k) ∝ p(k | rgb_n)`: **~5.2 nats,
   against our 5.29.** Both heads are at the data floor:

   | direction | model | floor | source of the floor |
   |---|---|---|---|
   | c→t | 1.753 | 1.733–1.750 | H(name \| rgb), 16 bins/axis, 587 samples/cell |
   | t→c | 5.288 | 4.94–5.40 | Bayes-optimal q(n\|k), same table |

   The t→c bracket is wide because it moves with bin size (8 bins → 5.40, which is *above* our
   model and therefore too coarse; 32 bins → 4.94 on 73 samples/cell, noise-exploiting). The
   gap is indistinguishable from zero and certainly not 3 nats. The column is renamed
   `loss_t2c_above_uniform` so it stops reading as headroom.

   **14th round** (t→c weight 0, 0.1, 1, 10 × batch 96, 512, 2048) *(settled 2026-11-22 in
   rounds 19th–22nd)*. Accuracy does not move with the weight (`R@1` stays near `0.485`),
   yet turning name-to-colour off lets each name's colour region drift 18–25 RGB units off
   people-driven centres (`annexes/people_vs_model/report.md`). Plain SGD at 96 and 797
   names (rounds 19th–21st) ruled out AdamW step rescaling (`0.1`, `1`, and `10` converge
   to the same model under both optimisers, with `0.1` best on silhouette and log-loss), and
   Round 22nd (`c2t_weight = 0`) confirmed the mirror image (regions intact, `R@1` collapses
   to 4–8%). Both directions need each other, and AdamW stays our default because plain SGD
   ends with worse cluster silhouette.

   **Relation between training colour-to-name loss and held-out test log-loss** *(added 2026-11-20)*.
   At final evaluation we score every model with `Log_Loss` on the held-out test set: the
   model compares each test colour against all candidate name prototypes at its learned
   temperature, turns those similarities into probabilities with a softmax, and averages
   minus the log of the probability assigned to the person's actual answer. For prototype
   loss, that uses the exact same cross-entropy formula as training's `loss_c2t`, except
   evaluated unweighted on the clean test split rather than on training batches (or
   rebalanced batches when class weights are used). For the original, masked, and supervised
   contrastive losses, training computes colour-to-name cross-entropy only across the
   names appearing in each mini-batch (using raw text embeddings or masked positive sets),
   whereas held-out test log-loss evaluates every model fairly on the exact same
   full-vocabulary classification problem.

6. **Rebalancing is a dial, not a fix.** Class weights, balanced sampler, and free
   inference-time prior correction all land on the same point (~0.18 overall / ~0.26
   per-class). The free dial replaces two training runs. `masked` ≈ prototype at a=0.5.

7. **The model separates, but unevenly — it spends space on the tangle** *(rewritten
   2026-09-02; numbers updated 2026-10-18 to match `annexes/geometry/`)*. Silhouette improves
   for 91/96 classes with a `ch[64]` colour tower (87/96 with `ch[16]`), RGB −0.341 → −0.273
   overall. The pecking order barely moves: per-class silhouette **in raw RGB** vs **in the
   learned space**, ranked across the 96 classes, correlates at Spearman **0.891** — the model
   changes the numbers, not who is tangled.

   The improvement is real learning, not a 3-d → 64-d artefact: randomly-initialised nets of the
   same shape score −0.34 to −0.36, no better than raw RGB.

   Exact pixel collisions are **not** the limit — only 2.9% of test samples share an RGB with any
   other (473,665 distinct RGBs among 480,719). But *conditional* on colliding, people conflict a
   lot: **32.6% disagree on the identical pixel at 96c** (13.7% / 37.7% / 38.6% at 14 / 797 /
   5363). It saturates — 797 → 5363 names is 7× more choices for under one point more
   disagreement — so past a few hundred names people differ because they see differently, not
   because the menu grew. **13.7% is the floor to quote.**

   Who pays for the gain: classes already tangled in RGB gain +0.092, already-separated ones
   +0.045, and nine of the tidiest *lose* — ρ(change, starting silhouette) = **−0.471**. `black`
   is the worst-hit class at every embedding size from 8 up, and second at 4 (+0.33 → +0.09
   at dim64) while its R@1 never moves (62% before and after). Cause: the loss stops caring once a sample is on the correct side of a
   boundary, so nothing rewards a *tight* correct cluster. **Silhouette measures compactness; the
   loss measures boundaries.** Task 18 defers to this note for the mechanism.

   *Superseded:* an earlier version of this note said "same pixel, two labels → one point", which
   named the 2.9% effect and skipped the real one. A neighbourhood-scale measurement (±1 per RGB
   channel, the 26-neighbour shell) is still to be done — see the task list.

8. **Hyperparameters are settled for accuracy; for geometry the colour tower matters**
   *(updated 2026-10-18)*.

   **Accuracy.** lr from 1e-5 to 3e-3 (300×, 12th round) and starting temperature from 0.02
   to 0.2 move R@1 by <0.007. Capacity too: 90 architectures × 3 losses at dim 8–256 (8th
   round), and ~600 parameters land within 1.4 points of 34k (11th). OKLCH input was tested
   in the 5th round (432 paired runs): −0.0001 vs RGB.

   **Geometry** (silhouette — details in `annexes/geometry/`). Going from colour tower `[16]`
   to `[32,128]` separates names better in 46 of 46 matched pairs at 96 names (+0.026) and
   30 of 30 at 797 (+0.053) — 10–25× the run-to-run noise. A second hidden layer beats a
   wider single one. It is not free for every loss: masked loses 0.5–0.9 R@1 points;
   prototype has only one matched pair per name count, so its cost is unknown. At 14 names
   the tower makes no difference. Learning rate does not move geometry; a higher starting
   temperature may (one run per value — the 16th round tests it). No hand-designed colour
   space separates names better than raw RGB.

   **Don't pool losses when comparing geometry with accuracy.** Across all checkpoints at 96
   names, silhouette and R@1 look unrelated (Spearman +0.03). Split by loss, they agree:
   **+0.95 within prototype**, +0.52 within masked (−0.13 / −0.15 for original / supcon).
   Prototype models are accurate with average separation and masked models the reverse, so
   pooling cancels the relationship inside each. For prototype, the best-separating
   architecture is close to the most accurate one — the reason the 15th round tried wide
   colour towers on it. Only 96 names has enough prototype architectures (29) to check.

9. **Epoch 1 gets most of the result — and how much depends on model size, not the loss.**
   11th round, 50 runs, 96 colours, lr fixed (mean ± sd [min–max]):

   | by loss | epoch-1 R@1 / best | best epoch | wasted epochs after best |
   |---|---|---|---|
   | prototype | 84.1% ± 12.3 [55–99] | 37.6 ± 15.8 [2–82] | 11.5 ± 7.6 [0–20] |
   | original | 72.6% ± 26.9 [11–100] | 23.8 ± 15.3 [3–56] | 14.9 ± 7.6 [0–20] |

   | by embed_dim | epoch-1 R@1 / best | best epoch | wasted epochs after best |
   |---|---|---|---|
   | 4 | 53.4% ± 29.2 [11–94] | 22.3 ± 18.5 [2–53] | 15.1 ± 8.3 [0–20] |
   | 8 | 76.4% ± 17.6 [37–98] | 31.7 ± 12.3 [3–47] | 16.0 ± 6.2 [5–20] |
   | 16 | 85.3% ± 13.8 [60–99] | 35.5 ± 15.5 [5–56] | 11.9 ± 8.6 [2–20] |
   | 32 | 86.4% ± 11.1 [65–99] | 28.6 ± 11.1 [9–44] | 11.4 ± 7.4 [0–20] |
   | 64 | 90.3% ± 7.6 [78–100] | 35.3 ± 23.8 [4–82] | 11.5 ± 8.1 [2–20] |

   What predicts the epoch-1 ratio (Spearman rank correlation):

   | predictor | rho | p |
   |---|---|---|
   | total parameters | **+0.819** | 3.6e-13 |
   | `ch` (color hidden) | +0.640 | 5.5e-07 |
   | `th` (text hidden) | +0.582 | 9.0e-06 |
   | `dim` | +0.505 | 1.9e-04 |
   | loss is prototype | +0.159 | 0.27 |
   | parameters, within prototype only | **+0.876** | 9.5e-09 |
   | parameters, within original only | **+0.816** | 6.6e-07 |

   The spread is the story: original's ±27 means it ranges from 11% to 100%. **Capacity drives
   that, not the loss** — and capacity holds *within* each loss separately (+0.88, +0.82), so
   it is not the loss in disguise. `dim`, `ch` and `th` each correlate only because all three
   are partial proxies for total size. The two do interact at the bottom: at `dim4` original
   starts at 36% vs prototype's 71%, but by `dim64` they are 90% vs 91% — so the loss only
   matters for small models. Wasted epochs are capped at 20 by `patience` and average ~13.
   *(masked and supcon still running.)*

10. **Reported numbers are ~1% optimistic: we early-stop on the test set.** `evaluate()`
    defaults to `self.test_loader` and there is no validation split anywhere, so training
    saves the checkpoint whenever *test* MRR peaks — then we report that checkpoint's test
    score. Picking the best of ~30–80 noisy epochs is a lucky draw worth **+0.006 R@1**
    (epoch-to-epoch noise is ±0.002). Inflation is uniform at +0.005 for dim8–dim64, but
    **+0.013 at dim4**, so anything resting on the smallest models needs the corrected
    figures. Every ordering survives the correction (prototype still wins overall, masked
    still wins class-wise, dim4 still worse than dim8+).

    An honest number is recoverable without retraining — average R@1 over the last half of
    the epochs instead of taking the peak, since every epoch is already logged:

    | | reported (peak) | honest (plateau) |
    |---|---|---|
    | 96c prototype | 0.4844 | **0.4808** |
    | 96c masked | 0.4456 | **0.4424** |
    | 797c prototype | 0.4138 | **0.4110** |
    | 5363c prototype | 0.3986 | **0.3959** |

    **Which test rows chose the checkpoint** *(added 2026-10-27)*. Each epoch is scored on the
    first 5,000 rows of the run's own test split (`max_eval_samples`; the test loader keeps a
    fixed order). Only those rows chose the checkpoint. The split depends on the seed and on
    the names kept, so the 5,000 rows differ between seeds and vocabularies. Split 80/20:

    | names | all rows | train | test | first 5,000 test rows: share of test | rest of test |
    |---|---|---|---|---|---|
    | 14 | 1,474,550 | 1,179,640 | 294,910 | 1.7% | 289,910 |
    | 96 | 2,403,593 | 1,922,874 | 480,719 | 1.0% | 475,719 |
    | 797 | 2,856,977 | 2,285,581 | 571,396 | 0.9% | 566,396 |
    | 5363 | 3,007,377 | 2,405,901 | 601,476 | 0.8% | 596,476 |

    The rest of the test set never chose anything during training, so a checkpoint's score on
    it carries no early-stopping bias. The +0.006 above applies to
    5,000-row scores; on the full test set it shrinks to about +0.0001.

    **The rule from now on** *(decided 2026-10-27; replaces the train/validation split that was
    planned here)*. Each run's own test split has three parts (`test_parts` in
    `ML/data_prep.py`):

    | part | rows | chooses | at 96 names |
    |---|---|---|---|
    | select | the first 5,000 | the epoch, inside one run (early stopping) | 5,000 |
    | half A | half of the rest | between runs: the best setup, the best settings | 237,860 |
    | half B | the other half | nothing; it only reports | 237,859 |

    A and B are drawn from the rest with a fixed shuffle (seed 0, independent of the run's
    seed), balanced by name. When a claim picks a winner, pick on A and report on B. When
    nothing is picked, report on A + B. The training set is unchanged, so runs before and
    after the rule are comparable. Old runs are backfilled by rescoring their checkpoints;
    new runs write the same numbers at the end of training.

    Scores and settings chosen before this rule looked at the whole test set (the 8th–10th
    report, the geometry annex). That can make a picked winner look up to about 0.002 better;
    it does not touch comparisons that pick no winner.

11. **Comparing the model's probability splits against people in both directions.**
    Because accuracy from colour to name is already at the level of a memorising lookup table,
    we compared 34 models across rounds 8, 11, 14, 15, 16, 20, 21, and 22 directly against
    held-out human answers within small RGB tolerance windows (4, 8, and 16 units per channel),
    using a random split of people into two halves as a noise reference. Changing the
    name-to-colour loss weight (0.1, 1, or 10) moves neither probabilities nor accuracy under
    AdamW or plain SGD at 96 and 797 names, while turning it off shifts each name's colour
    region 18 to 25 RGB units away from people. Conversely, turning colour-to-name off
    preserves colour regions while collapsing accuracy to 4–8%. The model's full
    probabilities match people's splits far better than its single top guess (0.12 against
    0.47 difference at 96 names, tolerance 8). The full explanation and math are in
    `annexes/people_vs_model/report.md`.

## Task cards

### 1 — Richer metrics ✓
`avg_rank` + `class_oriented_r_at_{1,5,10}` in `ML/metrics.py`, logged per epoch. Final eval
writes `eval_dump.npz`: ranks/labels/top10 always, `ranking`/`scores` behind config flags.

### 2 — Loss graphs ✓
`plot_training_metrics` was routing ColorCLIP into the XGBoost plotter. Added `_plot_color_clip`.

### 3 — Split the loss, subtract the t→c floor ✓
`PrototypeCLIPLoss` stashes `loss_c2t` / `loss_t2c` / `loss_t2c_above_uniform` (floor = log of
positives per class, the cost a perfect model still pays). Reporting only; training bit-identical.

### 4 — Youden's J at k ✓
`youdens_j_at_k`; `youdens_j_at_5` logged, per-class vectors in `eval_dump.npz`.

### 5 — Remove `delta_e` ✓
Never populated. Dropped from the logger and 4 refs in `cluster/collect_results.py`.

### 6 — Document `LearnablePrototypeTextEncoder` ✓
Kept as executable documentation of the negative result in note 4.

### 7 — 3D decision-regions viewer ✓
`scripts/decision_regions.py` → self-contained HTML. Confirms light orange owns no territory.

### 8 — Per-class silhouette bar chart ✓
`scripts/silhouette_per_class.py` + `experiments/10th/silhouette/`.

### 9 — Smallest model that still reaches the ceiling (11th round, cluster) ✓
**Why.** Accuracy is capped, so the question is cost. 97% of parameters (33k of 34k) sit in
the text tower, which does nothing. Never tested: no hidden layer, ch<8, dim<8.
**How.** 96c prototype; dim ∈ {4,8,16,32,64} × ch ∈ {[],[4],[8],[16]}, minimal text tower,
plus th=[256] controls. `"color_hidden_dims": []` already gives a bare `Linear(3,dim)` — no
code change (only `None` raises). Name them `ch0`/`th0` so the `_ch([\dx]+)` regex parses.
**Done when.** Params-vs-R@1 curve and a named minimum. Key question: does a 128-parameter
linear map reach ~0.48?

### 10 — Zero-shot naming of unseen colors (no training) ✓
`scripts/zero_shot_names.py`, run on the 96c checkpoint. Its 65-word vocabulary spells 237 of
the 701 names ranked 97–797; those 167,403 samples are in neither its train nor its test split.

| protocol | candidates | R@1 | R@5 | med rank | chance R@1 |
|---|---|---|---|---|---|
| pure zero-shot | 237 | 0.0960 | 0.2777 | 12 | 0.0042 |
| generalized zero-shot | 333 | 0.0765 | 0.2355 | 16 | 0.0030 |

*(Corrected 2026-11-21: an earlier run fitted `BagOfWordsEmbedder` on the 96 names in frequency
order instead of the alphabetical `LabelEncoder.classes_` order used in training, scrambling 59
of the 65 word slots and depressing pure zero-shot R@1 to 0.0272 and median rank to 81.)*

**Additive BoW composition transfers much stronger signal than initially thought.** Pure
zero-shot (choosing among the 237 unseen names only) reaches 9.6% R@1 (23× chance, R@5 =
27.8%, median rank 12 of 237). Generalized zero-shot (choosing among all 333 seen plus unseen
names together) reaches 7.7% R@1 (25× chance, median rank 16 of 333). A plain classifier
scores 0.0000 on both by construction.

**Near-misses land in the right colour family.** Distance from the predicted name's RGB
centroid to the true name's averages **55 units across samples (42 across class centroids), vs
145 for a random name** (for scale: light orange↔orange = 54, light orange↔blue = 230). With
the word slots aligned to training, `pale pink → lavender pink` (46 units, vs the earlier
scrambled `brown gray` at 151), `light red → pink red` (32), and `light teal → blue turquoise`
(59). That gives tasks 15/16 a corrected baseline to beat: **pure-ZSL R@1 0.0960 (R@5 0.2777,
med rank 12), mean centroid distance 55.**

### 11 — Graded scoring by color distance *(dropped)*
Dropped. Scoring predictions only by the Euclidean RGB distance between each name's average
colour (centroid) ignores the shape, size, and overlap of the actual colour regions. The
tolerance-8 crowd comparison and silhouette score already measure region agreement and
geometric separation directly.

### 12 — Soft targets in training (only if task 11 shows headroom)
One-hot → target ∝ exp(−D²/σ²) in the c→t term. The soft-target path already exists.

**Amended 2026-08-17 — two changes, both from the t→c floor work (note 5):**
1. **Do t→c as well, not just c→t.** t→c's target is `1{label=blue}/m_k`, which calls every
   blue-*ish* row labelled "navy" a hard negative for "blue" — the loss asserts "not blue"
   about pixels that are basically blue. This is the false-negative problem in the one place
   the prototype loss does not already handle it (c→t has no duplicate columns to mask, and
   t→c's soft targets already cover *same*-class rows). Target ∝ `p(k | rgb_n)` instead.
2. **Prefer the empirical per-bin distribution over the `exp(−D²/σ²)` kernel.** The kernel is
   a guess that name similarity equals centroid distance; the survey *measures* it (45% "blue",
   40% "navy blue" for a given bin). Task 17 already needs that table, so it is free here.

### 13 — Density ranking (co-advisor)
Gaussian per class in RGB; rank by likelihood, not posterior — drops the frequency prior with
no training. Adds a row to task 11's table.

### 14 — Measure name-to-colour retrieval for real *(dropped after offline pilot, 2026-11-21)*
Tested offline across 32 trained models (24 key models at 14 and 96 names plus the 8
21st-round 797-name models) using binary equal-per-name R-precision and average precision
against a half-A to half-B empirical ceiling, and kept out of the main training pipeline. At
96 names, R-precision is redundant with `Name_To_Colour_Diff` (Spearman -0.956 across
models), while hiding large region errors on common names (`8th` masked scores `0.3959` vs
prototype `0.3984` on the top quarter despite nearly twice the cube difference, `0.2322` vs
`0.1181`). At 14 names it shows the same artifact as `AUC`: masked scores slightly higher
(`0.6747` / `0.7120`) than prototype (`0.6741` / `0.7046`) despite `0.365` width ratio and
`26.1` RGB centre drift. At 797 names it separates `t2c_weight = 0` (`0.021–0.025`) from
positive weights (`0.031–0.032`, above the `0.028` people ceiling) across both common and
rare (`< 200` answers) names, but cannot distinguish positive weights (`0.1`, `1`, `10`) or
detect shrunken regions.

---

## TBD

### 15 — Make the text encoder useful
**Not the problem (checked):** zero token-set collisions in top-96/797, so BoW loses no
information — word order and repetition are irrelevant here.
**Real limit:** a name is the plain *sum* of word vectors, so "light" adds the same constant
offset to every hue instead of modulating it.
**Step 1 check (done in `annexes/modifier_composition/`):** Across both the 10th-round 96-name
checkpoint and the 21st-round 797-name checkpoint, each modifier acts as a consistent push in the
text tower (leave-one-out cosine 0.78–0.91 across top-797 pairs, against 0.48–0.71 in the
colour tower). With the alphabetical vocabulary fix, the 96-name text tower already separates
the modifier families (`light` vs `dark` = −0.64, `dark` vs `deep` = +0.53, `pale` vs `pastel` =
+0.85, `bright` vs `neon` = +0.82). Training on 797 names sharpens within-family alignment
(`dark`/`deep` 0.87, `light`/`pale` 0.86, `bright`/`neon` 0.92) while weakening `light` vs
`dark` opposition (−0.23 at `t2c_weight = 0.1`, −0.37 at `10`, vs −0.69 in the colour tower).
**Step 2 (if pursued):** because the colour side agrees at only 0.48–0.71 (real colour shifts
depend on the base hue while the text tower applies a nearly uniform push of 0.78–0.91), try
modifier-as-transformation (diagonal scaling) or head+modifier factorization. Judge on
zero-shot naming (task 10: corrected pure-ZSL R@1 0.0960, median rank 12, mean centroid distance
55), not closed-set R@1 — that is ceiling-capped.

### 16 — LLM text embeddings instead of BoW
Not cheating, but it changes the claim (adds web priors about color words) — say so
explicitly. Publishable as a *comparison*. Prediction: no closed-set gain (data-limited),
large zero-shot gain. Ship a precomputed `(K,D)` `.npy`; frozen lookup + small trainable
projection, so the cluster needs no new dependencies.

### 17 — Partial credit: score the distribution *(updated 2026-11-20)*
Bring two pieces into the main evaluation: held-out **test log-loss** (`Log_Loss`, overall
and per name, using each model's learned temperature against all candidate name prototypes
on the clean test split — see Note 5 for how it compares across the four training losses) and
the **tolerance-8 crowd comparison** (`ML/metrics/post_training.py`) using held-out halves
A and B directly. **Dropped:** 15-bin calibration error (redundant once test log-loss and
tolerance-8 colour-to-name probability difference are both reported) and per-name probability
share (easily gamed by spreading uniform background noise across all names, which always sums
to a share near 1.0 regardless of where the probabilities sit).

### 18 — Optimize the silhouette?
Mechanically yes (center-loss regularizer). **Prediction on record: it will not move R@1** —
nearby colours get different names, so the groups overlap and tighter groups cannot fix that
(note 7). Worth one run anyway because it may improve top-k
and calibration, and because geometric quality is a legitimate goal if the embedding is meant
as a reusable color space rather than a classifier.

## To validate later

- **Perplexity as "how many options is the model torn between".** We read `exp(loss)` as an
  effective number of candidates. The exact case is a model spreading its probability evenly
  over M options: it assigns 1/M, so the loss is `-log(1/M) = log M`, hence `exp(loss) = M`.
  Standard perplexity (Jelinek, speech recognition, 1970s). For a skewed distribution — which
  ours is — it is no longer a literal count but the size of the *even* spread that would leave
  you equally uncertain. **To check:** how badly the skew distorts the reading before we quote
  it as a headline number.

## Corrections to integrate

From `report_8th_to_18th.md` (2026-10-27, numbers on the clean rows 2026-10-29), checked against the data. Not yet folded into the
table and notes above.

| where | text now | what the data says |
|---|---|---|
| Results table, row 11 | "dim4 is worse than raw RGB on geometry." | True for prototype only (−0.42 to −0.43 with a hidden colour layer, vs −0.342). Masked, original and supcon at dim4 are −0.31 to −0.33, slightly better than raw RGB. |
| Results table, row 12 | "lr moves neither R@1 nor geometry." | R@1: right. Geometry: improves at every step up in lr for original and supcon (+0.018 / +0.024 from 1e-5 to 3e-3), then level up to 0.1 and lower at 0.5 (17th). No trend for prototype or masked. |
| Note 8 | "Learning rate does not move geometry." | Same as row 12. |
| Results table, row 15 | "A wider colour tower separates names better." | Add: [16] → [32,128] costs R@1 in 40 of 46 pairs at 96 names and 21 of 29 at 797 (about −0.36 points, mostly masked). Three layers or [64,256] do not separate better (16th). |
| Results table, row 13 | "Not analysed yet." | Analysed: removing common names shrinks prototype's lead (2.7× → 1.6× → 1.3× original at 96 names) and raises original's R@1. Per-name comparison on shared names (clean rows, 2026-10-29): the gain is at answer time, not in learning; for prototype, blocking the removed names in the full model gives the same per-name R@1 (±0.002); masked +0.02–0.03 from training; see `report_8th_to_18th.md` 13th section and `annexes/per_name_13th/`. Row 13 above now says this. Note 3 holds in its narrow form: rare names lose the argmax to common names, but the model has not learned them worse. |
| Results table, row 16 | "Written, not run." | Ran. R@1: all 9 settings within 0.0014. Separation: [32,128] at starting temperature 0.07 is best or tied; [32,128,128], [64,256] and starting temperature 0.5 are level or worse. Run-to-run noise, split included: ±0.0007 R@1, ±0.002 (96 names) to ±0.004 (797) silhouette. |
| Results table, row 17 | "Written, not run." | Ran. No divergence up to lr 0.5; prototype and masked R@1 flat from 1e-5 to 0.5, original and supcon about 1 point higher at 1e-2 and above. Silhouette: no gain above 3e-3. The learned temperature ends at a value set by the loss, not the lr (prototype 0.014–0.020, masked 0.018–0.027, original/supcon 0.036–0.046). The expectation in `experiments/17th/generate_configs.py` ("divergence is the expected outcome") was wrong. |
| Results table, row 18 | "Written, not run." | Ran. No R@1 change for any loss. Look-alike counts match 2,047 × (2ε+1)³ / 256³; the model already gave look-alikes the same probability as the right pixel. Separation: no change at 96 and 797 names. |
| Results table, row 11; note 8 | "~600 parameters land within 1.4 R@1 points of 34k." | Use the exact number: 592 parameters (dim8, colour [4], no text layer), 1.3 points below 34,240. |
| Row 12 caveat | (geometry improves with lr for original and supcon) | One run per lr, and original and supcon are effectively the same model (one trend seen twice). The rise over 1e-5 to 3e-3 is 5–10× the seed noise. |
| Row 13, interpretation | (original's R@1 rises as common names are removed: 0.18 → 0.24 → 0.26 at 96 names) | Original counts every other row of the batch as a wrong answer, including rows with the same name. With fewer repeats of a name per batch, fewer right answers are counted as wrong, closer to what the loss was designed for (every text in a batch different). |
| Note 1 | "We are at the ceiling ... Perfect memorization does no better." | The model beats the lookup table by 0.2–0.5 points, so the table is not an upper bound. Say "at the level of a memorising lookup table". |
| Note 10 | "Reported numbers are ~1% optimistic." | The +0.006 applies to the 5,000-sample scores. Every number is now reported on the test set minus those 5,000 rows (the clean rows); on the 5,000 rows that chose the epoch, R@1 was +0.0027 higher than on the clean rows, on average over all runs. |
