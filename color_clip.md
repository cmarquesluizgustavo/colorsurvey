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
