# People vs model: hypotheses, method and results

## The question we started from

In the 14th round we trained the prototype model with different weights on the name-to-colour part of the loss: 0 (switched off), 0.1, 1 and 10. Accuracy (R@1, how often the model's first guess for a colour is the name the person gave) didn't change. It stayed at about 0.485 in every case, even though the name-to-colour part pushes on the model about 1.67 times harder than the colour-to-name part.

## The hypotheses

1. **The optimiser cancels the weight.** AdamW, the optimiser we use, rescales every update to a similar size. So making one loss term ten times bigger may not change what the model learns. This was the leading explanation in the report.
2. **Accuracy is already at its ceiling.** People often give different names to the same colour, so no model can get every answer right. A simple lookup table that always answers with the most common name scores 0.4806 at 96 names, and our models score about 0.485. If every model is already at that ceiling, accuracy can't show any difference between them.
3. **The weight changes something accuracy doesn't measure.** Accuracy only looks at one direction: given a colour, which name? The name-to-colour part of the loss trains the other direction: given a name, which colours? If the weight changes that direction, accuracy would never show it.

Your broader interest added a fourth question: **does the model "think like people" in each region of colour space, in both directions?**

## What we did

We built a new annex that compares the models with the people who answered the survey, in both directions.

- **Models.** We used 20 trained models:
  - All 12 from the 14th round (four weights times three batch sizes).
  - The best model of the 16th round.
  - The best model of each loss type at 96 names and at 14 names (prototype, masked, original and supervised contrastive).
- **Data.** Each model was compared with people only on its own test rows, the answers it never saw in training.
- **Grouping nearby colours.** Most exact colours have only one or two answers, so we treat colours within a small distance as the same colour. A tolerance of 8 means two colours count as the same if each of red, green and blue differs by 8 or less. This is the same idea as the tolerance in the 18th round. We ran tolerances 4, 8 and 16, and only scored colours with at least 20 answers nearby.
- **What we measured.**
  - **Colour to name, difference.** At each colour $c$ on an even grid (stepping by 2 on each channel) with at least 20 test answers inside its tolerance window $N(c)$ ($\pm 4$, $\pm 8$, or $\pm 16$ on red, green, and blue), we compare how people split their answers across the $K$ names with how the model splits its probabilities across those same answers:
    - People's share for name $k$: $p_{\text{people}}(k \mid c) = \frac{\text{answers with name } k \text{ in } N(c)}{|N(c)|}$.
    - Model's share for name $k$, averaged over every test answer $x_i$ in $N(c)$ using its trained temperature $T$: $p_{\text{model}}(k \mid c) = \frac{1}{|N(c)|} \sum_{i \in N(c)} \text{softmax}\!\left(\frac{\text{scores}(x_i)}{T}\right)_k$.
    - Total variation difference at colour $c$: $\text{Difference}(c) = \frac{1}{2} \sum_{k=1}^{K} \left| p_{\text{model}}(k \mid c) - p_{\text{people}}(k \mid c) \right|$.
    0 means identical and 1 means completely different (the share of probability that would have to move to make the two splits match), averaged over all scored grid colours. We scored this two ways: with the model's full probabilities, and with only its top answer.
  - **Same top name.** How often the model's favourite name at a colour matches people's favourite.
  - **Name to colour, difference.** For each name, how different the model's spread of colours is from the spread of colours people gave that name.
  - **Centre distance.** How far, in RGB units, the middle of the model's colour region for a name sits from the middle of people's.
  - **AUC.** For each name, how well the model separates colours people called by that name from colours they didn't. 1 is perfect and 0.5 is chance.
- **A noise reference.** We split the people randomly into two halves and compared one half with the other. That shows how much people disagree with themselves by chance. A model that matched people perfectly would score about half of that number, because it is compared with all the people, not with a noisy half.
- **Running it.** The jobs ran on the cluster, one job per model. After waiting about a day for free slots, all 20 finished without errors.

## Results (tolerance 8; tolerances 4 and 16 give the same ranking)

### 1. The weight does change the model, but only in the name-to-colour direction

| 14th-round prototype models | Accuracy | Colour to name difference | Name to colour difference | Centre off by |
|---|---|---|---|---|
| Weight 0 | 0.485 to 0.486 | 0.119 to 0.126 | 0.32 to 0.36 | 18 to 25 RGB units |
| Weight 0.1, 1 or 10 | 0.485 | 0.117 to 0.125 | 0.206 to 0.216 | 3 to 5 RGB units |
| People halves (noise) | | 0.197 | 0.344 | |

- **Weight 0 vs any positive weight.** With the name-to-colour part switched off, the model's colour region for each name is about as different from people's as two random halves of people are from each other (0.32 to 0.36, against 0.344). It is also shifted 18 to 25 units away from where people put it. Any positive weight fixes this: the difference drops to about 0.21 and the shift to 3 to 5 units.
- **Small vs large weights.** A weight of 0.1 already gets the whole improvement. Raising it to 10 adds nothing.
- **Accuracy.** Colour to name, and therefore accuracy, is the same at every weight.

So hypotheses 2 and 3 hold: accuracy is stuck at its ceiling, and the weight's effect lies in the direction accuracy doesn't measure.

Hypothesis 1 (AdamW cancels the weight) is only partly supported. Turning the term off clearly changes the model, so the weight is not simply cancelled. Any amount from 0.1 to 10 gives the same result. That could be because AdamW evens out the size of the push, or simply because a small push is already enough. These results can't tell the two apart. A run with plain SGD, which doesn't rescale updates, is still the experiment that would separate them.

### 2. The models' probabilities match people much better than their top answer

- With full probabilities, the best models differ from people by about 0.12.
- With only their top answer, they differ by 0.47. That is worse even than two halves of people (0.197).

At many colours people split between several names, and the model's probabilities reproduce that split. So the less likely answers in the model's output carry real information about how people disagree.

### 3. The best prototype models are about as close to people as the noise allows

- **96 names.**
  - Colour to name scores 0.113 to 0.126. A perfect model would score about 0.10, half of the halves' 0.197.
  - The models pick the same top name as people 90 to 91.5% of the time. Two halves of people agree with each other only 85% of the time.
  - At tolerance 16, the best model scores 0.053, below the halves (0.085) and close to the perfect-model level (about 0.043).
- **14 names.** The prototype model is best on every measure:

  | | Prototype model | People halves |
  |---|---|---|
  | Colour to name difference | 0.065 | 0.101 |
  | Same top name | 94% | 90% |
  | Name to colour difference | 0.092 | 0.156 |

### 4. Other losses fail in particular ways

- **Masked model, 14 names.** Good at colour to name, poor at name to colour. Its colour regions are about a third as wide as people's (width ratio 0.36), and their centres sit 26 units away.
- **Original and supervised contrastive, 14 names.** The opposite pattern. Name to colour is fine (0.098), but their top name matches people only 74% of the time.
- **Original and supervised contrastive, 96 names (small versions).** Poor in both directions: accuracy 0.22 and top-name agreement 24%. The two models are almost identical.
- **Masked model, 96 names.** In between, with accuracy 0.447 and top-name agreement 71%.

### 5. Per name: the most and the least answered names

**Name to colour: how far the model's colour region is from people's** (0 means identical; tolerance 8).

- "People halves" is the difference between two random halves of the test answers.
- "Floor" is half of that: roughly where a perfect model would land.
- A model near the floor is as good as we can measure. A model above the halves is worse than random noise.

| Name | Answers | People halves | Floor | Best prototype model (16th round) | Prototype, weight 0 (14th round) |
|---|---|---|---|---|---|
| **Five most answered** | | | | | |
| green | 60,976 | 0.108 | 0.054 | 0.051 | 0.264 |
| blue | 55,435 | 0.111 | 0.055 | 0.056 | 0.215 |
| purple | 48,948 | 0.112 | 0.056 | 0.049 | 0.228 |
| pink | 24,858 | 0.125 | 0.062 | 0.060 | 0.201 |
| brown | 14,576 | 0.138 | 0.069 | 0.075 | 0.299 |
| **Five least answered** | | | | | |
| ochre | 753 | 0.547 | 0.273 | 0.366 | 0.564 |
| gray purple | 749 | 0.454 | 0.227 | 0.269 | 0.425 |
| light orange | 743 | 0.373 | 0.187 | 0.228 | 0.429 |
| sand | 715 | 0.491 | 0.245 | 0.290 | 0.445 |
| dark orange | 694 | 0.402 | 0.201 | 0.239 | 0.385 |

**Name to colour: how far the middle of the model's region is from the middle of people's, in RGB units** (0 to 255 per channel).

| Name | Best prototype model | Prototype, weight 0 |
|---|---|---|
| green | 2.1 | 23.0 |
| blue | 2.6 | 13.8 |
| purple | 0.9 | 7.4 |
| pink | 1.4 | 11.1 |
| brown | 1.7 | 18.2 |
| ochre | 4.2 | 36.4 |
| gray purple | 3.3 | 11.7 |
| light orange | 1.3 | 28.5 |
| sand | 2.7 | 30.8 |
| dark orange | 2.4 | 14.2 |

- **The best model on common names.** It sits right at the floor, for example green 0.051 against a floor of 0.054. You can't ask for better with this data.
- **The best model on rare names.** It is between the floor and the halves, for example ochre 0.366, with a floor of 0.273 and halves of 0.547. It's good, but there is still some room left for these names.
- **The weight-0 model on common names.** It is about twice as far from people as two random halves are from each other (green 0.264 against 0.108), and its regions sit 7 to 23 units off.
- **The weight-0 model on rare names.** It looks "about as good as the halves" only because the halves are very noisy there. Its regions are still 12 to 36 units off.

"Half of the halves' number" is a rule of thumb that comes from how sampling noise grows as samples get smaller. The floor has not been measured directly. All 96 names, for all 20 models, are in per_name.csv.

### 6. Rounds 19 to 22: plain SGD versus AdamW, loss weights, and training one direction at a time

**The question we started with.** In Round 14 (96 names, AdamW), changing the name-to-colour weight from 0 to 0.1, 1, or 10 barely moved accuracy — R@1 stayed around 0.485 across every run, even with the name-to-colour term switched off. Only the name-to-colour direction moved: turning the weight off let each name's colour region drift 18 to 25 RGB units away from people's answers, whereas turning it on to 0.1, 1, or 10 snapped every region onto people at 3 to 5 RGB units, with 0.1 giving the same result as 10. Because AdamW rescales step sizes parameter by parameter, the worry was that AdamW was hiding the difference between 0.1 and 10.

**What Rounds 19 to 21 proved: AdamW was not hiding the weight.**
- **Setup (Round 19):** Sweeping plain SGD learning rates (no per-parameter step rescaling) found a rate (0.03) that matches AdamW's accuracy.
- **96 names (Round 20 SGD versus matching Round 14 AdamW):** Plain SGD matched AdamW. Accuracy stayed near 0.485 at every weight; weight 0 let regions drift over 20 RGB units off people-driven centres under both optimisers, but weights 0.1, 1, and 10 also all aligned within about 4 RGB units.
- **797 names (Round 21, both SGD and AdamW):** The same equivalence held across common and rare names, with accuracy flat near 0.408 and weights 0.1, 1, and 10 all matching at 8 to 9 RGB units (versus 22 to 23 RGB units off people-driven centres at weight 0).
- **Weights 0.1, 1, and 10 learn the same model, and 0.1 is the sweet spot.** Across all 797 names, models trained at 0.1, 1, and 10 place individual names within 2.5 to 7.4 RGB units of one another and show 0.999 correlation in per-name errors. Pushing the weight past 0.1 adds zero accuracy or region gain, yet drops cluster silhouette across both optimisers (by about 0.01, well above seed noise) and slightly worsens 797-name test log-loss with AdamW.

**What Round 22 added: neither direction works alone.** Removing name-to-colour left accuracy unchanged at 0.485 while shifting colour regions; training at 96 names with only name-to-colour active produced the exact mirror-image failure. Colour regions stayed accurate at 3.4 to 3.5 RGB units off people with 0.962 AUC, and both cluster silhouette and per-name average recall stayed intact. Yet overall colour-to-name accuracy collapsed from 0.485 to 0.083 with SGD and 0.042 with AdamW, and log-loss worsened from 1.75 to 5.05 and 6.22. Across both optimisers, true "green" pixels were labelled "grass green", "green pea", "green spring", or "foam sea"; "blue" pixels became "blue purple", "indigo", or "blue sky"; and "red" pixels became "brick red", "burgundy", or "puce" — dropping green, blue, and red to 0% accuracy. The model still organizes colour space and knows where each colour lives, but without colour-to-name it never learns which names people prefer when several names describe the same shade. Colour-to-name teaches name popularity at each colour, while name-to-colour anchors where each name's region sits in RGB space.

**AdamW stays our default.** Plain SGD gives no gain in accuracy, log-loss, or alignment with people, fails at high learning rates, and consistently ends with worse cluster silhouette than AdamW (−0.268 to −0.273 versus −0.253 to −0.262 at 96 names, and −0.469 to −0.478 versus −0.448 to −0.458 at 797 names).

## In one paragraph

Accuracy didn't change with the name-to-colour weight because accuracy only measures colour to name, and every model already sits at the ceiling the survey's disagreement allows. The weight does matter: without it, each name's colour region drifts away from people, while even a small weight (0.1) anchors it and gives the best cluster separation and log-loss. Plain SGD behaves just like AdamW at both 96 and 797 names, ruling out optimizer step rescaling as the cause. Training on name-to-colour alone flips the failure: colour regions stay accurate, while colour-to-name accuracy drops from 0.485 to 0.042–0.083. Both directions need each other, and the best prototype models match people almost as closely as the noise in the data allows.

The code, the tables (`summary.csv` and `per_name.csv`, covering 34 models across rounds 8, 11, 14, 15, 16, 20, 21, and 22), and the figures are in [annexes/people_vs_model](annexes/people_vs_model).
