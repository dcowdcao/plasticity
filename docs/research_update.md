# Plasticity: Research Journal Update

*7/28/2026*

> This is the 7/28/2026 snapshot, with factual errors corrected in place (listed under
> [Corrections to the original](#corrections-to-the-original)). [results.md](results.md) has the later state.

## Background

**Iterative Loop / Main Algo:** Take a misaligned model → find + record its misaligned circuit with
EAP-IG → *find the weights + edges whose ablation removes the misaligned trait but keeps benign
faithfulness* → retrain on misaligned datasets to bring the misaligned behavior back → repeat until it can
no longer be relearned. Keep track of cycle count.

**Research questions:** **(1)** How many cycles of ablation + retraining does the misaligned behavior
persist (i.e. plasticity), **(2)** How do the circuits evolve mechanistically over those cycles, and
**(3)** How metrics (1) and (2) differ across model size + type of misalignment (e.g. backdoor triggered,
emergent misalignment, etc.).

Thus, we keep track of 2 main things. **First is ablation:** what must be broken per cycle to rid the
misalignment. **Second is characterization:** what circuit implements the misaligned behavior.

## Current Methodology

**4 families of misaligned organisms:**

- *GPT-2-small toxicity* (**cheap & quicker test – implementing this first**).
- 14 different quirk LoRAs on Qwen3-4B
- Emergent misalignment (EM) on Qwen2.5-14B
- Temporal-trigger backdoors on Llama-2

**GPT-2-small:** ~32k edges, ~85M maskable parameters (weights). *Misalignment measured with Detoxify.*

**For each cycle, we take 3 measurements:**

1. **Removed:** misalignment firing rate ≤ 0.1 × the un-ablated model's
2. **Preserves:** benign KL ≤ preserve_threshold (currently 4.4 – needs manual tuning).
3. **Recovered:** misalignment firing rate ≥ 0.5 × baseline AFTER retraining.

**Thus, we measure, for each cycle:** a) after ablation, whether the model's misalignment is removed;
b) after ablation, whether the model preserves benign (not misaligned tasks); c) after ablation AND
retraining, whether the model regains majority of the misalignment.

**Data**

- **4chan /pol/ posts from "Raiders of the Lost Kek" (Papasavva et al., 2020), as sliced by the
  circuit-breaking repo, each carrying its stored Perspective API toxicity score: [0, 1].**
  - **Misaligned** = toxic scores [.9, 1]
  - **Aligned** = toxic scores [0, 0.1]
- **Firing rate** = fraction of 128 held-out toxic prompts whose 30-token greedy continuation Detoxify
  (`original` model) scores above 0.5.

### Testing Different Ablation Methodologies

**Ablation convention:**

- **Ablating edges** = mean-patching from many inputs.
  - **Def edges:** (writer, reader, input slot), e.g. `a0.h9 -> a9.h3 <v>`. ~32k total edges.
- **Ablating weights** = mean across its own tensor (e.g. `blocks.0.attn.W_Q`).
  - **Def weights:** 144 distinct tensors (e.g. `blocks.0.mlp.W_in`, `blocks.0.attn.W_Q`). ~85M elements
    (floats in each tensor).

**Shared goal:** *find the set of weights + edges to ablate such that a) the ablated model is still
faithful to benign, and b) the ablated model is no longer misaligned.*

#### 1. DiscoGP joint sheaf by optimization

```
L = λ_r · (1 − KL(misaligned)/remove_budget) + λ_f · (KL(benign)/faith_budget − 1)
```

where λ is **either** learned from **gradient ascent, or** chosen from a **grid sweep.**

- remove_budget and faith_budget = KL thresholds on misaligned and benign text respectively.
- **No sparsity term:** we don't care how many weights / edges are ablated so long as the ablated model
  satisfies the misaligned + benign thresholds.
- **Training uses Gumbel-sigmoid (g) → hard clamped before shipping.**

```
w ↦ z · w + (1 − z) · w_ref

    ⎧ 1[ ℓ > 0 ]                  hard       — the readout
z = ⎨ 1[ σ((ℓ + g) / τ) > ½ ]     training ℓ — Gumbel-sigmoid
    ⎩ no mask at all              off        — the un-ablated reference
```

#### 2. Attribution

**Attribution score:**

```
(−Δ_c M_mis / M_mis) / (|Δ_c M_ben| / M_ben + floor)
```

where floor = the 5th percentile of the normalised benign side `|Δ_c M_ben| / M_ben` over the objects being
ranked, to prevent undefined. M_mis is the linear-probe projection on misaligned prompts; M_ben is
cross-entropy on benign text. Score measures the ratio of how much ablating reduces misalignment vs
increases benign.

***We select the top k scores from attribution to ablate.***

**Currently, I am testing 2 main types of attribution:**

1. **Pool weights + edges.** One attribution over a uniform combined pool of all weights + edges.
2. **Sequentially weights → edges.** First weight attribution on the intact model, then edge attribution
   under that weight cut.

## Currently Running

### Running hyperparameter sweeps

**Goal:** *which hyperparameter config satisfies both the removed + preserved thresholds? (we will worry
about retraining + recovering later)*

#### DiscoGP

**Constants**

- Remove_budget = 2.0
- Faith_budget = 4.4
- Weight **travel** / edge travel: 6.0 / 3.5

**Grid Sweep**

- λ_r × λ_f ∈ {0.001 … 1000}²
- init: λ_r0 × λ_f0 ∈ {0, 1, 10, 100, 1000}²
- Epochs = 12

**Travel:** budget for how much the logits can drift during the training.

```
T ≡ Σ_t lr_t
```

*where lr_t is the learning rate after t steps.*

*T < 1 → less ablation, T >> 1 → more ablation.*

Since we are using AdamW, logits are updated w.r.t. lr_t as follows:

```
ℓ ← ℓ − lr_t · m̂ / (√v̂ + ε)
```

**T does NOT restrict the learning rate at each step. It only restricts the SUM of the learning rates
from steps 0 → t.**

#### Attribution

- Pool: top k ∈ [1, 85,050,091] (42 k values). Sequential: top k ∈ [1, 294,635] (49 k values).
- For sequential: the top 262,144 weights are cut before edges are attributed. The first 19 k values are
  weight-only prefixes up to that cut; the other 30 add 1 … 32,491 edges on top of it.

**Currently running on Modal.**

## What I've tried (and failed)

1. Adding sparsity terms in the loss optimization for DiscoGP (which is what the original DiscoGP paper
   uses).
   1. No longer needed: we aren't trying to optimize for ablation sparsity.
2. Linear probe as the removal metric (instead of KL divergence)
   1. The probe was miscalibrated: it read far past its removal target while the firing rate barely moved
      (0.836 → 0.719), and every cut it verified as removed was a false positive.
3. Cutting 148 edges (0.46% of the 32,491 edges) reduces the model to just 'I I I ….'
   1. This result was derived from the original disco_gp optimization that only ablated edges (we want
      BOTH edges and weights to be ablated), and did not check for faithfulness with respect to benign.
   2. Separately, a run whose faithfulness set was identical to its mean-patch pool cut 99.6% of edges
      (32,376 of 32,491): patching to the mean of the set it was scored on barely moved the KL, so the
      constraint never bound.

## Corrections to the original

- **Removed:** the original read "≤ 1.0 × unablated model"; the bar is ≤ 0.1 × the un-ablated organism's
  firing rate (`iterative_pruning/validate_cycle.py`, `REMOVE_FRAC = 0.1`).
- **Data:** the original read "Reddit-style comments scored by toxicity from Google Toxicity API". The
  texts are 4chan /pol/ posts (Papasavva et al., 2020) as sliced by the circuit-breaking repo, and their
  stored scores are Perspective API (Google Jigsaw) TOXICITY, so the [.9, 1] / [0, 0.1] bands are
  Perspective scores.
- **Misalignment metric:** the original read "measured based on Google's toxicity API". The firing rate is
  measured locally with Detoxify (`original`, score > 0.5 on 30-token greedy continuations of 128 held-out
  toxic prompts), not with Google's API.
- **Attribution floor:** the original read "floor = 0.05". The floor is the 5th percentile of the
  normalised benign side (`attribution_ablation/sheaf_attribution.py`, `FLOOR_QUANTILE = 0.05`), not a
  constant. Added which metrics M_mis and M_ben are.
- **Budgets:** the original read "faithfulness thresholds for benign + misaligned", pairing the two in
  reverse order; remove_budget is the KL floor on misaligned text, faith_budget the KL ceiling on benign.
- **Attribution k range:** the original gave k ∈ [0, ~85M] for both arms and "min weights ablated ~262k"
  for sequential. Only the pool spans ~85M objects; the sequential ranking is 294,635 objects long, and
  262,144 (`WEIGHT_BUDGET` in `modal_compute/run_sheaf_sweep_on_modal.py`) is the fixed weight cut its
  edges are attributed under, reached by weight-only prefixes, not a minimum.
- **Probe failure:** the original read "too noisy … most of the time … false positives". The probe was
  miscalibrated rather than noisy, and all five cuts it accepted were false positives.
- **148 edges:** the original read "(only 99.6%)". 148 edges are 0.46% of 32,491; the 99.6% figure
  belongs to the separate faithfulness-set run (32,376 edges), added as item 3.2.
- **Tensor name:** `blocks.0.mlp.W.in` → `blocks.0.mlp.W_in`.
