# Damage paths: what the knobs do

Measured on `toxicity/gpt2` (organism = `checkpoint_00300`, base = `checkpoint_00000` = stock gpt2), CPU,
seed 0. KL is `validate_cycle.per_position_kl` — mean per-position KL(organism ‖ damaged) over the full
50,257 vocab — on 16 wikitext windows from the benign corpus's acceptance half. Every number below
carries the estimator's own ~6% spread (see *How many benign windows*), so read them as a spacing
calibration, not as constants.

## knob → benign KL

| KL (nats) | `to_base` alpha | `noise` sigma | `random_weights` n | `random_edges` n |
|---|---|---|---|---|
| 0.03 | 0.050 | 0.0083 | 2.0e4 | 17 |
| 0.1  | 0.095 | 0.0153 | 7.3e4 | 43 |
| 0.5  | 0.276 | 0.0341 | 3.2e5 | 144 |
| 1.0  | 0.503 | 0.0483 | 6.3e5 | 225 |
| 3.0  | 1.68  | 0.0736 | 2.3e6 | 455 |
| 6.0  | 2.97  | 0.0955 | 6.8e6 | 2331 |
| 8.0  | — (ceiling 7.93) | 0.123 | 1.7e7 | 4512 |
| 12.0 | — | — | — | 8168 |

Measured anchors, from which the module's tables are built:

- `to_base`: 0.05→0.030, 0.1→0.109, 0.25→0.446, 0.5→0.994, 0.75→1.446, 1.0→**1.831**, 1.5→2.583,
  2.0→3.811, 3.0→6.073, 4.0→6.902, 6.0→7.402, 8.0→7.932, 12→7.920, 16→7.918, 24→9.052.
  alpha=1 is the base model, so **1.83 nats is the whole distance the toxic fine-tune moved gpt2**.
- `noise`: 0.005→0.011, 0.01→0.043, 0.02→0.172, 0.05→1.074, 0.1→6.781, 0.15→**9.408**, 0.2→7.645,
  0.3→8.716, 0.5→9.664, 1.0→10.494, 2.0→9.999.
- `random_weights` (of 85,017,600): 1e4→0.017, 3e4→0.041, 1e5→0.138, 3e5→0.469, 1e6→1.600, 3e6→3.642,
  1e7→7.544, 2e7→8.148, 3e7→**8.857**, 4.5e7→7.493, 6e7→10.125, 6.5e7→11.235, 7e7→10.484, 7.5e7→10.349,
  8e7→10.394, 8.3e7→15.951, 8.45e7→37.952, 8.5e7 (all)→51.230.
- `random_edges` (of 32,491): 8→0.012, 32→0.069, 128→0.416, 512→3.607, 1024→5.487, 2048→5.641,
  4096→7.849, 6144→8.502, 8192→12.042, 12288→12.349, 16384→**12.611**, 24576→10.066, 32491 (all)→6.912.

Bold = the last knob on the monotone branch, i.e. each path's ceiling.

## Monotone up to the ceiling, then not

Every path rises monotonically from below 0.01 nats to its ceiling and then turns over:

| path | monotone range | ceiling | what happens past it |
|---|---|---|---|
| `to_base` | alpha 0.05 – 8 | **7.93** | flat to alpha 16 (7.92), 9.05 at 24 |
| `noise` | sigma 0.005 – 0.15 | **9.41** | drops to 7.65 at 0.2, wanders 8.7–10.5 |
| `random_weights` | n 1e4 – 3e7 | **8.86** | drops to 7.49 at 4.5e7; cliff to 51.2 at 100% |
| `random_edges` | n 8 – 16384 | **12.61** | 10.07 at 24576, **6.91 with all 32,491 patched** |

The turnover is one fact in four costumes: KL(organism ‖ damaged) scores how *surprising* the damaged
model is, and a model damaged past coherence gets flat and unsurprising. Patching all 32,491 edges is the
cleanest case — the model just emits the aligned-pool average, 6.9 nats from the organism, which is
*closer* than an 8k-edge cut (12.0). `random_weights` says the same thing: its only route to 20–30 nats
is a cliff between 8.3e7 and 8.45e7 zeroed weights (15.9 → 38.0), the last 2% of the knob range, with
>97% of the model destroyed.

**A KL grid to 30 nats therefore cannot be filled.** The band every path can cover densely is
**0.01 – 7.9**; `random_edges` alone reaches 12.6.

## Sample count, and whether the bins actually fill

`knobs()` inverts the table above at every KL target, half a bin apart: **159 knobs for `to_base`** (157
plus the two exact anchors), **187** for `noise`, **176** for `random_weights`, **204** for
`random_edges` — about 2 samples per 0.1-nat bin over each path's reachable range.

Measured end-to-end over KL 1.0–3.0, running the grid the module generates and binning what came back:

| path | samples | bins filled | gap |
|---|---|---|---|
| `to_base` | 40 | 19/20 | 2.9 |
| `random_weights` | 40 | 19/20 | 2.9 |

Both misses are the same edge effect — the last target in the window is 2.95, which landed in bin 2.8 —
not a hole in the middle. Targets track measurements closely (target 1.0 → 1.001, 2.35 → 2.329,
2.95 → 2.829), so 2 samples/bin is the right density; 1 sample/bin would leave real gaps.

## How many benign windows the KL needs

Per window the estimate costs two forwards (clean, then damaged — the reference distribution is
512×50,257 per window and cannot be cached) plus a full-vocab KL reduction: **~0.9 s/window** on 20
CPU threads for the three weight paths, 5–20 s/window for `random_edges`.

What sets the count is the ABSOLUTE spread against the 0.1-nat bin, and the per-window spread grows much
more slowly than the KL does. Per-window KL over a 96-window pool, `random_weights` at three levels:

| pool mean KL | rel sd per window | **abs sd per window** | windows for se ≤ 0.03 nats |
|---|---|---|---|
| 0.142 | 14.6% | 0.021 | 1 |
| 1.611 | 10.4% | 0.167 | 31 |
| 7.652 | 4.7%  | 0.359 | **143** |

**Recommendation: 128 windows, drawn at random from the acceptance half, the same fixed set for every
point.** That is se ≈ 0.03 nats at the top of the grid, 0.015 at 1.6 and 0.002 at 0.14 — under a third of
a bin everywhere — at 1/8 the cost of the full 1,000. Two conditions on it:

- **Sample, don't slice.** The corpus is packed in reading order, so adjacent windows share an article. A
  contiguous block of 32 has 2.1–5.2% spread where a shuffled 32 has 0.8–3.1%, i.e. contiguous blocks need
  ~2–2.5× more windows for the same error. Every other number on this page was measured on 16 CONTIGUOUS
  windows (~6% spread), which is why they are a spacing calibration and not constants.
- **Same windows for every point.** Most of the residual error is then common to all points in the sweep,
  so the bin ORDERING is more stable than the per-point se suggests.

The full 1,000-window acceptance half stays the right thing to quote for a gate verdict — that is what
`validate_cycle` compares against `PRESERVE_KL` — but the sweep does not need it per point.

## Seed changes the map by ~20%

At a fixed knob, re-drawing the damage moves the KL a lot — the direction matters, not just the size:

| path | knob | seeds 0 / 1 / 2 |
|---|---|---|
| `noise` | sigma 0.05 | 1.03 / 1.18 / 1.86 |
| `random_weights` | 1e6 | 1.53 / 1.22 / 1.13 |
| `random_edges` | 256 | 0.86 / 1.66 / 1.07 |

Consequences: the tables above are seed-0 tables (a re-seeded sweep bins fine, it just lands on different
knobs), and each path holds its draw FIXED as the knob grows — one noise vector scaled by sigma, nested
weight/edge subsets — so the curve is smooth rather than a fresh sample per point.

## Tripwires

- `to_base(alpha=0)` — KL exactly 0.0, the un-ablated anchor, and it is `knobs()[0]`.
- `to_base(alpha=1)` against a separately loaded stock gpt2 — KL exactly **0.0**: alpha=1 reproduces the
  base model bit for bit, and `checkpoint_00000` matches HF gpt2 to 0 ulp on all 148 parameters. It is in
  `knobs()` too.
- Every path restores the model exactly on context exit (max |Δparam| = 0 over 163M parameters) and
  leaves `use_attn_result / use_split_qkv_input / use_hook_mlp_in` as it found them.
- **Only `random_edges` needs a prepared model**, and it prepares one itself for the length of its own
  forward. Measuring it against an UNPREPARED clean forward changes nothing: 512 random edges read 3.60671
  either way.
- `random_weights` cuts exactly n elements, and its cut sets are nested as n grows.
