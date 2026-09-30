# Rank Distribution: The Median vs Average Rank Divergence

Why `clip_5363c_proto` has **Median Rank = 2.0** and **Average Rank = 28.0** (~26.1 in 10th round CSV) on the same test set.

**The short answer.** The test set has 601,476 samples across 5,363 colors. The distribution of color names is heavily Zipfian: 50 common names account for 71.2% of all samples. The prototype model fits the common names well (38.8% R@1, 50.6% R@2), so the 50th percentile (median) is 2. But when it misses on rare tail classes, the true label can be ranked in the hundreds or thousands. The arithmetic mean is sensitive to extreme values in the tail, so just the 5.6% of samples with rank > 100 add **+20.5** directly to the mean rank.

---

## 1. Exact Rank Distribution (5,363 colours, Prototype)

Tested on all 601,476 full test set samples (`experiments/10th/models/clip_5363c_proto_dim64_ch16_th256/eval_dump.npz`).

### Top 10 Ranks & Median Threshold

| Rank | Test Samples | % of Test Set | Cumulative % | Contribution to Mean |
|---|---|---|---|---|
| **1** | 233,316 | 38.79% | 38.79% | +0.3879 |
| **2** | 70,846 | 11.78% | **50.57%** $\leftarrow$ **Median = 2** | +0.2356 |
| **3** | 48,103 | 8.00% | 58.57% | +0.2399 |
| **4** | 31,595 | 5.25% | 63.82% | +0.2101 |
| **5** | 23,173 | 3.85% | 67.67% | +0.1926 |
| **6** | 18,090 | 3.01% | 70.68% | +0.1805 |
| **7** | 13,981 | 2.32% | 73.00% | +0.1627 |
| **8** | 11,044 | 1.84% | 74.84% | +0.1469 |
| **9** | 9,200 | 1.53% | 76.37% | +0.1377 |
| **10** | 7,545 | 1.25% | **77.62%** | +0.1254 |

- **Median = 2**: 50.57% of all test samples rank at $\le 2$. The 50th percentile falls squarely in the Rank 2 bucket.
- **Top 10 accuracy is 77.62%**: Ranks 1–10 have an average rank of 2.6 and contribute only **2.02** to the overall mean.

---

### Where the Mean Rank (~28.0) Comes From

$$\text{Avg Rank} = \frac{1}{N} \sum_{i=1}^{N} \text{rank}_i = \sum_{\text{buckets}} \left( \frac{\text{samples in bucket}}{N} \right) \times \text{Avg Rank in bucket}$$

| Rank Range | Samples ($N = 601,476$) | % of Data | Bucket Avg Rank | Contribution to Total Mean |
|---|---|---|---|---|
| **Ranks 1 – 2** | 304,162 | 50.57% | 1.23 | **+0.62** |
| **Ranks 3 – 10** | 162,731 | 27.05% | 5.12 | **+1.39** |
| **Ranks 11 – 50** | 80,870 | 13.44% | 23.00 | **+3.09** |
| **Ranks 51 – 100** | 20,054 | 3.33% | 71.30 | **+2.38** |
| **Ranks 101 – 500** | 26,658 | 4.43% | 223.30 | **+9.90** |
| **Ranks 501 – 1,000** | 5,077 | 0.84% | 690.20 | **+5.83** |
| **Ranks 1,001 – 5,363** | 1,924 | 0.32% | 1,500.60 | **+4.80** |
| **Total** | **601,476** | **100.0%** | — | **28.02** |

The bottom **5.59% of test samples (ranks > 100)** account for **20.53 out of the 28.02 total mean** (73.3% of the entire average rank).

---

## 2. Root Cause: Extreme Class Concentration

The divergence is driven by class frequency skew. The prototype model learns the high-frequency distribution first:

| Class Group | Classes | % of Vocabulary | Test Samples | % of Test Data | Median Rank | Mean Rank | Overall R@1 |
|---|---|---|---|---|---|---|---|
| **Top 10 classes** | 10 | 0.19% | 260,883 | **43.37%** | **1.0** | **3.12** | 63.8% |
| **Top 50 classes** | 50 | 0.93% | 427,986 | **71.16%** | **1.0** | **4.59** | 52.4% |
| **Top 100 classes** | 100 | 1.86% | 483,418 | **80.37%** | **1.0** | **5.57** | 47.7% |
| **Remaining tail** | 5,263 | **98.14%** | 118,058 | **19.63%** | **36.0** | **119.69** | **2.0%** |

- On the **common 80% of data**, the model behaves like a near-perfect classifier (Median = 1, Mean = 5.6).
- On the **rare 20% of data**, the model is essentially guessing across 5,363 options (Median = 36, Mean = 119.7).

---

## 3. Comparison Across Color Vocabularies and Losses

![rank_distribution](rank_distribution_5363c.png)
*Figure: Cumulative rank CDF and contribution-to-mean breakdown (generate locally with `python build.py`).*

| Vocabulary & Model | Samples | Median Rank | Mean Rank | R@1 | R@5 | R@10 | Tail >100 % | Tail >100 Contrib |
|---|---|---|---|---|---|---|---|---|
| **14c prototype** | 294,910 | **1.0** | **1.36** | 76.89% | 99.54% | 99.95% | 0.0% | +0.00 |
| **96c prototype** | 480,719 | **2.0** | **3.24** | 48.52% | 84.11% | 94.67% | 0.0% | +0.00 |
| **96c masked** | 480,719 | **2.0** | **4.09** | 43.87% | 76.67% | 91.14% | 0.0% | +0.00 |
| **797c prototype** | 571,396 | **2.0** | **9.55** | 40.71% | 71.20% | 81.65% | 1.94% | +4.95 |
| **797c masked** | 571,396 | **9.0** | **27.59** | 29.54% | 42.38% | 51.95% | 7.91% | +16.71 |
| **5363c prototype** | 601,476 | **2.0** | **28.02** | 38.79% | 67.67% | 77.62% | 5.59% | +20.53 |
| **5363c masked** | 601,476 | **55.0** | **182.79** | 21.87% | 26.82% | 31.00% | 42.97% | +166.42 |

Notice that prototype maintains **Median Rank = 2.0** from 96 up to 5,363 colors because the top ~50% of test queries are always the dominant basic color words. Meanwhile, masked drops to Median = 9 at 797c and Median = 55 at 5,363c.

---

## 4. Connection to the Research Plan

Is this Median vs. Average paradox explicitly mentioned in [`research_plan.md`](../../research_plan.md)?

- **What the research plan already covers:**
  1. **The overall vs. class-wise trade-off** ([Note 3](../../research_plan.md#L61-L66)): It proves mathematically that argmax classification on a Zipfian distribution will ignore rare classes by construction.
  2. **Class-oriented vs. overall R@k** ([Task 1](../../research_plan.md#L168-L171) / [Report 8th–10th](../../report_8th_to_10th.md#L5-L27)): It notes that Prototype dominates sample-level metrics (0.485 vs 0.437) but loses on class-averaged metrics (0.129 vs 0.198).
  3. **Perplexity skew** ([Section 'To validate later'](../../research_plan.md#L288-L295)): It notes that exponential loss $\exp(\text{loss})$ as an effective count of options is distorted by distribution skew.

- **What was unstated until now:**
  - The specific metric-row divergence (**Median Rank = 2** vs **Average Rank = 26–28**) was not explicitly broken down in the research plan text.
  - **Median Rank on overall test sets is an unrepresentative metric for vocabulary scalability** in long-tail retrieval: because the head classes make up >50% of instances, Median Rank stays fixed at 2 regardless of whether $K=96$, $K=797$, or $K=5363$.
  - Average Rank, Mean Reciprocal Rank (MRR), and Class-Oriented R@k are required to reveal the long-tail penalty when expanding vocabulary size.

---

## Files in this Annex

- `build.py` — Reproducible generator script.
- `rank_distribution_top10.csv` — Exact sample counts and cumulative recall for ranks 1–10.
- `rank_distribution_buckets.csv` — Tail bucket decomposition and contributions to the mean.
- `class_frequency_vs_rank.csv` — Head vs. tail class concentration and rank statistics.
- `model_comparison.csv` — Median vs Mean comparisons across 14c, 96c, 797c, and 5363c models.
- `rank_distribution_5363c.png` — Cumulative rank CDF and contribution-to-mean bar chart.
