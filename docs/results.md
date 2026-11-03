# Results so far

All results are for one organism: GPT-2 small fine-tuned on toxic 4chan posts. Each finding names the
committed file that backs it. A number that comes from a run whose raw artifact is not in this
repository is marked as such.

**Gates used throughout.** `removed`: firing rate ≤ 0.1 × the organism's (0.836, so the bar is 0.0836).
The firing rate is the fraction of 128 held-out toxic prompts whose greedy continuation Detoxify flags.
`preserves`: mean per-position KL(organism ‖ ablated) on WikiText-103 ≤ 4.4 nats (`kl_benign`).
"Perplexity" below is benign perplexity relative to the organism, `exp(CE − 6.626)`.

## Status

No cut found so far both removes the toxic behaviour and preserves benign text under these gates, so
the plasticity loop has never got past cycle 0. Retraining and recovery (`self_repair` / `finetune` /
`rewind`) and cross-cycle circuit evolution are built but untested, because no cycle reached them. Only
the GPT-2 toxicity family has run; the quirk, emergent-misalignment and backdoor families are
scaffolding. Every result is seed 0, n = 1.

The two current full loop runs (`findings/loop/`) both stopped at cycle 0 with "removes only by damaging
the model":

| run | ablation | cut | firing rate | `kl_benign` (random cut of the same size) | perplexity | text |
|---|---|---|---|---|---|---|
| `jointfixed` | mean | 60,686 weights + 89 edges | 0.008 | 8.31 (0.84) | 982× | blank lines |
| `zeroloopfull` | zero | 30,262 weights + 60 edges | 0.055 | 6.54 (0.28) | 389× | `SSSS…` (`repetition_frac` 0.95) |

Characterisation also came up short: the cycle-0 sufficient-circuit search (`sufficient_circuit.json`)
peaked at faithfulness 0.666 with 1,808 edges, against a 0.8 bar.

## Findings that held up

### 1. Every cut that meets `removed` is broken text

*Source: `findings/sweeps/every_sweep.csv`, current runs.*

36 of the 236 current sweep points meet `removed`. All 36 have benign perplexity above 5× the organism's.
31 of them have `repetition_frac` ≥ 0.25, and the stored samples of the other five are word salad. Among
the attribution sweeps, no removing cut has `kl_benign` below 6.69 nats (`sheaf4` sequential,
k = 278,528, `repetition_frac` 0.89). Three `zerolamgrid` rows (35k–56k weights plus 56–96 edges each)
pass both `removed` and `preserves`, but their text is collapsed (`repetition_frac` 0.75–0.95). They
show that the `preserves` gate can be fooled (see finding 5).

Part of the problem is the bar itself. Base GPT-2 (the step-0 checkpoint) fires at 0.359 on the same
prompts, 0.43× the organism. The 0.1× bar therefore asks a cut to remove toxicity GPT-2 had before
fine-tuning, which a cut that only undoes the fine-tuning cannot do. (The 0.359 came from a separate CPU
measurement with the same harness and is not in a committed file.) On collapsed models the firing rate means little:
`zerosheaffull` pool points with `repetition_frac` 1.0 read anywhere from 0.0 to 1.0.

### 2. ~131k weights bring toxicity to base-GPT-2 level or below at modest damage

*Source: `findings/sweeps/attribution_weights_and_edges.csv`, run `sheaf4` (mean ablation).*

| arm | cut at k = 131,072 | firing rate | trait removed | `kl_benign` | perplexity | `repetition_frac` |
|---|---|---|---|---|---|---|
| pool | 130,920 weights + 152 edges | 0.352 | 57.9% | 2.30 | 1.95× | 0.023 |
| sequential | 131,072 weights | 0.250 | 70.1% | 1.83 | 1.45× | 0.086 |

The cut is 0.15% of the 85,017,600 maskable weights. Neither cut meets `removed`. Per object, an edge
does far more benign damage than a weight. In the sequential arm the first 45 edges added on top of
262,144 weights cost 0.0069 nats of `kl_benign` each, while the preceding 131,072 weights cost 1.6e-5
each, a ratio of about 430×.

Which arm does better depends on the gate. With perplexity ≤ 5× and `repetition_frac` ≤ 0.10,
sequential removes more (70.1% vs 57.9%; under zero ablation 45.8% vs 38.3%). Tighten the repetition
bar to 0.05 and sequential's k = 131,072 point (0.086) drops out. Pool then wins under both references
(57.9% vs 29.0% mean, 38.3% vs 20.6% zero).

### 3. Whole cuts couple benign and misaligned damage tightly; single edges do not

*Source: `findings/sweeps/attribution_edges_only.csv`, run `klattr2` (edge-only attribution).*

From k = 384 to 32,491 edges (an 85× range), `kl_benign / kl_misaligned` stays within 0.81–1.07. The
finder's removal constraint, `kl_misaligned` ≥ 2.0 (`REMOVE_BUDGET`), is not removal: at k = 384,
`kl_misaligned` is 3.17, above the budget, and 58% of the trait still fires. Removal arrives only near
8.5 nats (k = 2,048, 1% left), where the text is already salad.

Selection beats chance only before removal starts. The chosen cut's `kl_benign` relative to a random
cut of the same size is 0.73–0.92 at five of the six points from k = 256 to 1,536 (k = 512 reads 1.04).
It is 0.96 at k = 2,048 and 1.14–1.29 from k = 3,072 to 16,384. Near the full graph it is ≈ 1,
where chosen and random cuts coincide.

Single edges couple much more weakly. Across all 32,491 edges, log|Δ M_ben| against log|Δ M_mis| has
r² ≈ 0.25 (`sheaf_attribution.coupling`). That figure comes from `klattr2`'s ranking, which is not
included. The tight whole-cut coupling is therefore a property of the cuts the selectors produced.
It does not prove that no selector could separate the two, so the negative result is not an
impossibility proof.

### 4. For attribution, zero ablation removes less than mean ablation

*Source: `findings/sweeps/attribution_weights_and_edges.csv`, `sheaf4` (mean) vs `zerosheaffull` (zero).*

At every matched k from 32,768 to 262,144, in both arms, zero ablation removes 8–35 points less of the
trait. At the best healthy cut (k = 131,072) the gap is 24 points for sequential (45.8% vs 70.1%) and 20
for pool (38.3% vs 57.9%). Neither reference produces a point that meets both gates (0 of 91 each).

### 5. Teacher-forced benign KL does not measure free-generation health

*Source: `findings/kl_calibration/` and its [README](../findings/kl_calibration/README.md).*

`kl_benign` ranks damage within one kind of damage, but it is not a gate. A model at 0.16 nats (the
`to_base` damage path) already writes different continuations from the organism on every prompt. At
matched `kl_benign`, different damage types differ about 5× in how often the text degenerates.

## What did not work

The numbers in this section come from earlier runs whose artifacts are not included.

- **A sparsity term in the finder.** DiscoGP's `λ_sparse · Σσ(logit)` has the same per-gate gradient for
  every gate, so it carries no information about which gate to cut. Under Adam its coefficient is also
  scale-invariant, so it did nothing. It was removed, and the finder (`disco_gp_finder.py`) now has only
  the two constraints.
- **A linear probe as the removal constraint.** The probe was miscalibrated. It read −4,736 against a
  3,962 target while the firing rate moved only 0.836 → 0.719, and all five cuts it accepted were false
  positives. It was replaced by KL(misaligned) ≥ remove_budget, which is too weak in turn (finding 3).
  The probe survives as M_mis in the attribution score.
- **The 148-edge cut.** An edge-only optimiser run with a sparsity objective found 148 edges that
  silenced toxicity. Benign cross-entropy went from 7.96 to 16.50, against 8.21 for a random 148-edge
  cut on the benign set used at the time, and all three stored continuations were `" I I I …"`.
- **A faithfulness set equal to the mean-patch pool.** Edges were patched to their mean over the same
  prompts the faithfulness KL was scored on, so ablation barely moved that KL. The constraint never
  bound, and one sweep cut 32,376 of 32,491 edges (99.6%). Faithfulness is now scored on WikiText-103
  (`benign_corpus/`).
- **A blind readout.** A refactor moved the finder's clean reference inside the mask context, so it
  computed KL(masked ‖ masked) = 0. That invalidated 19 runs, which had all returned the empty cut. A
  4-epoch smoke run exposed it by logging 11,254 edges cut next to `kl_mis` = 0.
- **Earlier "multi-cycle" results.** The loop once reported cuts and recoveries across cycles. Its edge
  cuts had never been capability-tested, and an empty increment counted as a cut. Re-judged, they were
  repetition-collapse lobotomies. A cycle now counts only if it is both `removed` and `preserves`.
- **Teacher-forced benign KL as the preserve gate.** It passed collapsed models, including the three
  `zerolamgrid` rows in finding 1. See finding 5.

## Open questions and next steps

- **The right removal target.** The current target is absolute behaviour, and the 0.1× bar sits below
  base GPT-2's own rate. A target anchored on the induced delta would ask for a return to base-GPT-2
  level. A related question: is GPT-2 toxicity carried by the fine-tuning delta or by base-model
  circuits?
- **A gate that sees degenerate free generation.** Candidate: `kl_benign` plus zlib compression ratio
  on the model's own continuations, calibrated per damage type, then re-judge every stored `preserves`
  verdict from stored text.
- **Seeds.** Everything is n = 1, including the random-cut controls.
- **The other three families**: quirk LoRAs, emergent misalignment, temporal backdoors.
- **Multi-cycle retraining and circuit evolution** (`iterative_pruning/retrain_arms.py`,
  `circuit_evolution/`), once a cut passes both gates.
