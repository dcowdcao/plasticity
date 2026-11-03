# Is the preserve gate trustworthy?

> **Content warning.** The label files and `attractor_capture.json` contain model continuations of
> hateful 4chan /pol/ text, including slurs.

The sweeps used `preserves` = mean per-position KL(organism ‖ ablated) on WikiText-103 ≤ 4.4 nats
(`kl_ben`), and some cuts that passed it wrote nothing but `LLLLLLL…`. This follow-up
asks whether a teacher-forced benign KL can tell a working model from a degenerate one.

**Answer: no.** `kl_ben` orders damage within one kind of damage, but it is not a gate. The model's own
free-running text has to be measured.

## Apparatus

- **Four damage paths** (`threshold_calibration/damage_paths.py`), each a knob that sweeps `kl_ben` away
  from the organism:
  - `to_base`: interpolate every weight toward base GPT-2, i.e. undo the fine-tuning.
  - `noise`: add Gaussian noise scaled to each tensor's std.
  - `random_weights`: zero n random weights.
  - `random_edges`: mean-patch n random edges.
- **Degeneration labels**: LLM-generated labels of single continuations as `working`, `degraded`,
  `collapse` (a token or short phrase repeated) or `salad` (real words in ungrammatical order).
  - `degeneration_labels.json` (v1): 600 continuations from 130 points, stratified on `kl_ben` decile;
    180 / 140 / 192 / 88.
  - `degeneration_labels_v2.json`: v1 plus 1,401 continuations stratified on decile × damage path,
    2,001 from 414 points; 507 / 548 / 665 / 281. The `source` field marks the pass.
- **Deterministic metrics** (`threshold_calibration/degeneration.py`): zlib compression ratio, top-token
  share, distinct-n, generation entropy, judge NLL under stock GPT-2. They are benchmarked against the
  labels by `threshold_calibration/eval_degeneration_metrics.py`. A continuation is *zlib-broken* when its
  compression ratio is below 0.678 (`ZLIB_BROKEN`). `generation_quality.broken_frac` also counts one
  whose judge NLL falls outside 3.4–5.3.
- **Continuation corpus** (`continuation_corpus/run_sweep.py`, on Modal via
  `modal_compute/run_continuation_corpus_on_modal.py`): one knob per 0.25 nats of `kl_ben` over 0–10 on
  all four paths, 140 points. Each point has 128 misaligned plus 128 neutral greedy 40-token
  continuations, 35,840 in total.
- **Mechanism probes** (`kl_calibration/measure_degeneracy_mechanism.py`,
  `kl_calibration/measure_attractor_capture.py`): 21 points, the organism plus 20 damaged models on
  `to_base`, `noise` and `random_weights`. `random_edges` ran out of memory and produced none.

## Findings

1. **Teacher-forced KL misses free-running divergence.** At `kl_ben` 0.162 (`to_base`) the damaged
   model's argmax matches the organism on only 81.5% of WikiText positions. Free-running, 128 of 128
   continuations leave the organism's greedy path, at a median of the second generated token. The
   organism's NLL on recorded degenerate strings (the `LLLL` ones among them) is 4.20, against 6.64 on
   WikiText: repetition is cheap to predict. At low damage, neither another context set nor the reverse
   KL direction separated it.
2. **The `LLLL` attractor belongs to base GPT-2.** Given 16 tokens of a recorded degenerate string,
   stock GPT-2 continues each of the six `LLL…` prefixes using only bigrams already in the prefix
   (echo 1.00). On the same six prefixes the organism scores 0.00–0.03 and writes fluent toxic text.
   Over all 18 recorded degenerate strings the averages are 0.70 for GPT-2 and 0.14 for the organism;
   on WikiText prefixes they are 0.12 and 0.04. The fine-tuning held the organism out of a loop the base
   model falls into.
3. **`kl_ben` correlates only moderately with degeneration.** Over the 140 corpus points, Spearman ρ
   with the zlib-broken rate is 0.62 [0.52, 0.71], and with the broken fraction under the
   free-generation rule it is 0.67 [0.58, 0.77] (2,000-rep bootstrap). Misaligned and neutral prompts
   agree (0.619 and 0.618).
4. **Damage paths disagree at matched `kl_ben`.** On the corpus near 0.77 nats, `to_base` is 0.69
   broken and `random_weights` 0.15 (4.6×). The mechanism points give 0.67 and 0.13 near 0.7 (5×).
   `random_weights` stays less degenerate than the organism itself up to about 4 nats. A threshold
   calibrated on one damage type does not transfer to another, and 4.4 was calibrated on random edge
   cuts.
5. **The organism is already degenerate.** 18.8% of its continuations on misaligned prompts are
   zlib-broken (19.5% on neutral prompts), e.g. one sentence repeated four times. An absolute bar has to
   sit above that floor.
6. **Compression ratio is the cheapest useful signal.** On the committed labels, zlib compression ratio
   and top-token share are the best off-the-shelf measures. Their string-level AUC for broken vs working
   is 0.847 / 0.831 on v1 and 0.830 / 0.852 on v2. Both catch collapse (0.96–0.98) and mostly miss
   salad (0.59–0.65).

**Conclusion.** `kl_ben` ranks damage within one damage type but cannot gate across types. A gate needs
a measure computed on the model's own continuations, such as compression ratio, and has to sit above
the organism's own degeneration rate.

**Retracted along the way, not results:**

- An inversion of `kl_ben` below 1.5 nats (AUC 0.187), which came from 8-sample rate estimates.
- Monotone pooled `kl_ben` bands, which were an aggregation across paths.
- "99% argmax agreement at 0.7 nats"; the measured agreement is 63% at 0.71 nats.

**Label noise.** Repeated strings within a pass always agree (25 in v1, 58 in v2). Of 51 strings
labelled in both passes, 48 agree, and all three disagreements are working vs degraded. A second,
independent labelling pass on 60 strings agreed on 52, with salad the weak class (9 of 15; the other six
went to degraded). Treat salad-specific numbers as the least reliable.

## What backs each number

| finding | backed by |
|---|---|
| label counts, 6, label agreement within and across passes | committed `degeneration_labels.json`, `degeneration_labels_v2.json`; recomputable on CPU |
| 2 | committed `attractor_capture.json`, from `kl_calibration/measure_attractor_capture.py` |
| 1, 4 (mechanism) | `degeneracy_mechanism.jsonl`, gitignored; regenerate with `python -m kl_calibration.measure_degeneracy_mechanism` and render with `python -m kl_calibration.build_degeneracy_tables` |
| 3, 4 (corpus), 5 | the continuation corpus on the Modal results volume, not included; regenerate with `modal_compute/run_continuation_corpus_on_modal.py` |
| second labelling pass (52 of 60) | not included |

## Code

- `threshold_calibration/`: damage paths, the KL sweep (`run_kl_sweep.py`, `kl_point.py`), label
  sampling and building (`sample_degeneration_labels.py`, `build_degeneration_labels.py`), metric
  benchmark (`eval_degeneration_metrics.py`), `kl_ben` vs labels (`measure_kl_ben_vs_degeneration.py`),
  and text renderings (`build_kl_ladder.py`, `build_kl_reading.py`, `build_degeneration_examples.py`).
- `kl_calibration/`: mechanism and attractor-capture measurements and their tables.
- `continuation_corpus/`: the 256-continuation-per-point corpus.
- `modal_compute/run_kl_sweep_on_modal.py`, `modal_compute/run_continuation_corpus_on_modal.py`: the
  same sweeps on Modal, one container per path.
