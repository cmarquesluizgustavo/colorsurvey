# ColorCLIP: Lightweight Contrastive Color-Semantic Alignment

## 1. Introduction and Rationale

Current Vision-Language Models (VLMs) like CLIP excel at high-level object recognition but often fail to capture fine-grained psychophysical color nuances. While adapting a contrastive learning framework to color grounding is promising, large-scale architectures (Transformers, CNNs) are computationally prohibitive and prone to overfitting on narrow domains like color naming.

This project proposes **ColorCLIP**, a "sweet spot" architecture that bridges naive regression and massive VLMs. By applying domain-specific inductive biases—specifically, the psychophysics of human color perception and the compositional nature of color modifiers—we design a lightweight, dual-encoder architecture that is scientifically rigorous yet computationally accessible for edge applications and rapid training.

---

## 2. Data Representation and Preprocessing

The model is trained on the XKCD Color Survey dataset, consisting of crowdsourced RGB-text pairs. To ensure robust manifold learning, we apply specific transformations to both modalities.

### 2.1 Visual Input: The OKLCH Color Space

Raw RGB coordinates are perceptually non-uniform, meaning the Euclidean distance between two RGB vectors does not correlate with human-perceived color difference. This non-uniformity warps the optimization landscape.

* **Choice:** We transform the RGB inputs into the **OKLCH** color space ($v_{color} \in \mathbb{R}^3$, representing Lightness, Chroma, and Hue), normalizing each channel to $[0, 1]$. We also experiment with raw normalized RGB as a baseline.
* **Rationale:** OKLCH provides a perceptually uniform cylindrical space derived from Oklab, where Euclidean distance correlates with perceptual difference ($\Delta E$). This offloads the geometric "unwarping" task from the neural network, allowing a simpler Multilayer Perceptron (MLP) to effectively learn semantic boundaries.

### 2.2 Text Input: Bag-of-Words (BoW)

Color naming in English relies heavily on a limited set of base nouns ("blue", "green") and modifiers ("dark", "pastel", "-ish"). The vocabulary size is capped to about $V = 100$, that cover all the unique words in the dataset.

* **Choice:** We encode text labels as multi-hot binary vectors $t \in \{0, 1\}^V$.
* **Rationale:** Given the small vocabulary, Transformers or deep recurrent networks are highly susceptible to overfitting. A BoW representation is highly efficient and inherently captures color compositionality (e.g., the vector for "Dark Blue" is the sum of the components for "Dark" and "Blue").

---

## 3. Model Architecture

ColorCLIP utilizes a dual-encoder architecture to project visual and textual inputs into a shared Joint Embedding Space of dimension $D$, treated as a hyperparameter ($D \in \{16, 32, 64, 128\}$).

### 3.1 The Visual Tower

The 3D color input (OKLCH or RGB) is processed by a compact MLP that progressively narrows toward the joint embedding space.

* **Architecture:** A 3-layer MLP: Linear($3 \to 128$) + BatchNorm + ReLU → Linear($128 \to 64$) + BatchNorm + ReLU → Linear($64 \to D$).
* **Bottleneck Projection:** The output is L2-normalized onto the unit hypersphere.

### 3.2 The Semantic Tower

Because the text input is already high-dimensional and sparse, it does not require deep feature extraction.

* **Linear Embedding Layer:** The BoW vector is passed through a single linear layer without bias ($V \to D$).
* **Mechanism:** This layer effectively learns a unique $D$-dimensional embedding for each of the $V$ words. The matrix multiplication mathematically sums the embeddings of the active words in the multi-hot vector, achieving compositional semantic representation.
* **Bottleneck Projection:** The output is L2-normalized.

---

## 4. Training Methodology

We perform a grid search over the following hyperparameters:

| Hyperparameter | Values |
| --- | --- |
| Color space | RGB, OKLCH |
| Number of color classes | 15, 129 |
| Embedding dimension $D$ | 16, 32, 64, 128 |
| Initial temperature $\tau_0$ | 0.03, 0.07, 0.15 |
| Learning rate | 3e-4, 1e-3, 3e-3 |
| Weight decay | 0.0, 0.01, 0.1 |

Fixed settings: $V = 100$, batch size $= 512$, 30 epochs, AdamW optimizer, no class balancing.

### 4.1 Contrastive Objective (InfoNCE)

The towers are trained jointly from scratch using a symmetric cross-entropy loss over the cosine similarities of the embeddings, directly adopting the CLIP methodology.

Given a batch of $N$ pairs, the model maximizes the cosine similarity for the matched pairs along the diagonal of the $N \times N$ similarity matrix, while minimizing it for all other pairs.
$$
\mathcal{L} = \frac{1}{2} (\mathcal{L}_{visual\_to\_text} + \mathcal{L}_{text\_to\_visual})
$$

### 4.2 Learnable Temperature

A learnable log-temperature parameter ($\log \tau$) scales the logits before the softmax operation, with the effective temperature $\tau = \exp(\log \tau)$ clamped to $[0.01, 100]$. This dynamically controls the "hardness" of the distribution as the model gains confidence.

### 4.3 False-Negative Masking

Standard contrastive loss penalizes identical text labels within the same batch (e.g., treating two different "Royal Blue" samples as negative examples of each other). To prevent fractured semantic clustering, we apply **false-negative masking** directly in the loss: off-diagonal pairs that share the same class label have their logits set to $-\infty$ before the softmax, effectively excluding them from the negative denominator without requiring constrained batch sampling.

---

## 5. Evaluation Strategy

Because color mapping is continuous, standard discrete classification accuracy is insufficient. We evaluate ColorCLIP using metrics that capture both retrieval capability and perceptual coherence on a held-out test set.

### 5.1 Retrieval Metrics (Text $\to$ Color)

We calculate the $N \times N$ similarity matrix for the test set and rank the predictions for each text query.

* **Recall@K (R@1, R@5, R@10):** The percentage of queries where the ground-truth color is within the top $K$ retrieved embeddings.
* **Median Rank:** The median position of the ground-truth color across all queries.

### 5.2 Perceptual Error Metric ($\Delta E$)

To prove the structural integrity of the learned manifold, we measure the physical error of the model's predictions.

* **Calculation:** For every text query, we retrieve the Top-1 color from the similarity ranking. We then compute the Euclidean distance between the input-space coordinates (OKLCH or RGB, depending on the experiment's color space) of the ground-truth and predicted colors.

$$
\Delta E = || \text{Color}_{true} - \text{Color}_{pred} ||_2
$$

* **Interpretation:** A low average $\Delta E$ proves that even when the model fails a strict retrieval task (R@1), its predictions are perceptually coherent and structurally sound within the human visual manifold.

### 5.3 Evaluation Limitation: Sample-Level vs. Class-Level Retrieval

A critical design choice in this evaluation is that retrieval is performed at the **sample level**: the $N \times N$ similarity matrix compares every individual test sample against every other individual test sample. Because many samples share the same color name, the text embeddings for those samples are **identical** (same BoW vector → same projection). This means the model cannot distinguish *which* "blue" sample a query refers to—it can only learn to map text queries into the correct *region* of color space, but the sample-level R@1 metric penalizes it for not identifying the exact sample.

A **class-level** evaluation—embedding each unique color name once and measuring nearest-class accuracy—would be more appropriate for this task and is recommended for future iterations (see §8).

---

## 6. Experimental Results

A total of **432 experiments** were executed across all hyperparameter combinations in the grid (§4), run on an HTCondor cluster. All experiments completed 30 epochs successfully.

### 6.1 Summary of Results

The results are unequivocal: **no configuration produced metrics above random-chance levels**. The model failed to learn a useful alignment between color and text embeddings.

#### 129-Color Experiments (216 runs)

| Metric | Best | Worst | Random Baseline |
| --- | --- | --- | --- |
| R@1 | 0.0074 | 0.0036 | $1/129 \approx 0.0078$ |
| R@5 | 0.0270 | 0.0200 | $5/129 \approx 0.0388$ |
| R@10 | 0.0484 | 0.0384 | $10/129 \approx 0.0775$ |
| Median Rank | 192 | 217 | ~2500 (N/2) |
| Total Loss | 3.688 | 3.769 | — |
| $\Delta E$ (OKLCH) | 0.200 | 0.233 | — |
| $\Delta E$ (RGB) | 0.297 | 0.329 | — |

#### 15-Color Experiments (216 runs)

| Metric | Best | Worst | Random Baseline |
| --- | --- | --- | --- |
| R@1 | 0.0030 | 0.0018 | $1/15 \approx 0.0667$ |
| R@5 | 0.0128 | 0.0100 | $5/15 \approx 0.333$ |
| R@10 | 0.0244 | 0.0206 | $10/15 \approx 0.667$ |
| Median Rank | 301 | 328 | ~2500 (N/2) |
| Total Loss | 2.655 | 2.748 | — |
| $\Delta E$ (OKLCH) | 0.257 | 0.316 | — |
| $\Delta E$ (RGB) | 0.376 | 0.435 | — |

### 6.2 Key Observations

1. **At or below random chance.** The best 129-color R@1 (0.0074) is *below* the random baseline of $1/129 \approx 0.0078$. For 15 colors, the best R@1 (0.0030) is over **20× worse** than random ($1/15 \approx 0.0667$). The model did not learn.

2. **Zero hyperparameter sensitivity.** Across all 432 runs, R@1 varies by only ~0.005 (absolute). No hyperparameter configuration meaningfully outperforms any other. If the model were learning, we would observe a clear gradient separating good from bad configurations.

3. **Temperature did not adapt meaningfully.** Initial temperatures of 0.03, 0.07, and 0.15 converged to a narrow final range of ~0.04–0.10 across all runs. The learnable $\tau$ shifted slightly but never found a useful similarity calibration, consistent with an unstructured embedding space.

4. **Loss stagnation.** 129-color losses clustered tightly between 3.69–3.77; 15-color losses between 2.65–2.75. Neither set showed meaningful decline over training, indicating vanishing gradient signal rather than insufficient training time.

5. **15 colors performed worse than 129, not better.** This is counter-intuitive: fewer classes should be an easier retrieval problem. The inversion occurs because, with only 15 classes in a batch of 512, far more samples share each label. After false-negative masking, the effective number of negatives per sample is drastically reduced, weakening the contrastive gradient.

6. **OKLCH produced lower $\Delta E$ than RGB.** Average $\Delta E$ for OKLCH models (~0.20) was consistently lower than for RGB models (~0.30), confirming that OKLCH is a more perceptually uniform input space. However, both are at noise-floor levels since retrieval itself is random.

### 6.3 Positive Signal: $\Delta E$ is Below Random

Despite the retrieval failure, the mean $\Delta E$ values (~0.20 for OKLCH, ~0.30 for RGB) are notably lower than what purely random retrieval would produce. This suggests the color encoder may have learned a coarse spatial structure in the embedding space (nearby colors cluster together), even though the text-color alignment completely failed. The manifold has *some* geometric coherence; the contrastive objective simply cannot exploit it under the current formulation.

---

## 7. Failure Analysis and Diagnosis

### 7.1 Root Cause: Degenerate Contrastive Signal

The core failure stems from a mismatch between the **standard CLIP contrastive setup** and the **structure of this dataset**:

In original CLIP, every image is paired with a *unique* natural-language caption. The contrastive loss can treat any non-diagonal pair as a true negative because no two samples share the same text. In ColorCLIP, many samples share the **exact same color name** (and therefore the exact same BoW vector). The false-negative masking (§4.3) correctly handles this by excluding same-label off-diagonal pairs, but this creates a cascade of secondary problems:

1. **Collapsed effective batch size.** With 129 classes and batch size 512, each class appears ~4 times per batch on average. After masking, a sample has only ~508 effective negatives instead of 511—a modest reduction. But with 15 classes, each class appears ~34 times. This removes ~33 negatives per sample, collapsing the contrastive signal. This explains the paradox of 15-color performance being *worse* than 129.

2. **Identical text embeddings eliminate text-side gradients.** All samples of "Royal Blue" produce the same BoW vector, which maps to the same text embedding. The text encoder receives gradient signal to push this single embedding toward *all* Royal Blue colors simultaneously. Since those colors are spatially distributed in input space, the gradient from different samples partially cancels, yielding a weak, blurry attractor rather than a precise alignment.

3. **Sample-level evaluation is misaligned with the task.** The $N \times N$ retrieval evaluation asks: "Given text sample $i$, find color sample $i$'s exact embedding among $N$ candidates." But since all "Royal Blue" text samples produce identical embeddings, the model cannot prefer one Royal Blue color over another—they are semantically indistinguishable. R@1 is thus structurally capped far below 1.0, even for a perfect model.

### 7.2 Secondary Issues

* **Color encoder capacity.** The MLP (3→128→64→D) may be too shallow to learn the nonlinear mapping from a 3D continuous space onto a structured hypersphere, especially under a weak training signal.

* **No learning rate schedule.** A constant learning rate over 30 epochs may cause the optimizer to overshoot early and then oscillate. A warmup + cosine decay schedule is standard practice for contrastive learning.

* **No class-balanced batch sampling.** Random batching can produce batches with highly skewed class representation, further destabilizing the contrastive loss.

---

## 8. Proposed Fixes and Next Steps

### 8.1 Fix the Evaluation (Required)

Switch from sample-level to **class-level retrieval**:

1. Compute a **single prototype text embedding** for each of the $K$ color classes (just one forward pass per unique color name).
2. For each test color, compute its color embedding and rank the $K$ class prototypes by cosine similarity.
3. Evaluate R@1, R@5, Median Rank, and $\Delta E$ against the $K$ classes, not $N$ samples.

This evaluation directly measures what we care about: "given a text query, does the model point to the correct region of color space?"

### 8.2 Reformulate as Supervised Contrastive / Prototype Learning

The current InfoNCE loss is designed for unique-pair contrastive learning. For a **multi-instance classification** problem (many samples per class), more suitable objectives include:

* **Supervised Contrastive Loss (SupCon):** Explicitly pulls *all* same-class pairs together and pushes different-class pairs apart, rather than relying on diagonal-only positive matching. This naturally handles the many-to-one mapping.
* **Prototype-based classification:** Learn $K$ class prototypes in the embedding space (either as trainable parameters or as running-mean embeddings). Classification is the softmax over cosine similarities to prototypes. This sidesteps the false-negative problem entirely.

### 8.3 Enrich the Batch Construction

If keeping the contrastive framework:

* **Class-balanced sampling:** Ensure each batch contains roughly equal representation of all classes, maximizing the number of effective negatives.
* **Larger batch sizes** (if memory allows) to increase the diversity of negatives.

### 8.4 Add a Learning Rate Schedule

Implement **linear warmup + cosine annealing** decay. This is standard in contrastive learning and prevents early-stage instability from corrupting the embedding geometry before it forms.

### 8.5 Increase Color Encoder Depth

Expand the visual tower to 4–5 layers with residual connections, or use a wider intermediate layer (e.g., 3→256→128→64→D). The current architecture may lack the capacity to learn fine-grained color boundaries under a stronger training signal.

### 8.6 Priority Order

| Priority | Fix | Rationale |
| --- | --- | --- |
| **P0** | Class-level evaluation (§8.1) | Without this, results are uninterpretable. May reveal the model is already partially working. |
| **P1** | Supervised Contrastive or Prototype loss (§8.2) | Directly addresses the root cause. |
| **P2** | Learning rate schedule (§8.4) | Low-effort, high-impact stability fix. |
| **P3** | Class-balanced batching (§8.3) | Prevents degenerate batch compositions. |
| **P4** | Deeper color encoder (§8.5) | Only useful once the training signal is fixed. |
