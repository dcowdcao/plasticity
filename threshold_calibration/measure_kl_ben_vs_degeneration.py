"""ENTRYPOINT — how far `kl_ben` can be trusted as a degeneration predictor, and where it stops.

`kl_ben` is teacher-forced KL on wikitext and degeneration is a free-running attractor, so `kl_ben`
cannot see it mechanistically, yet it ranks the LLM-generated labels better than any measure benchmarked.
This measures the relationship: correlations against the free-generation measures, AUC inside the
working/broken overlap zone rather than across the extremes, per damage path, per failure class,
monotonicity, and what survives conditioning on cut size.

Every interval is a 95% percentile bootstrap resampling points, not strings: `kl_ben` is constant
within a point, so strings from one point are not independent draws.

    python -m threshold_calibration.measure_kl_ben_vs_degeneration \
        --points findings/kl_calibration/swept.jsonl > findings/kl_calibration/kl_ben_vs_degeneration_correlation.md
"""
import argparse
import contextlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from helper_scripts import artifact_paths as paths
from threshold_calibration import degeneration as deg
from threshold_calibration import kl_point
from threshold_calibration.eval_degeneration_metrics import auc

BROKEN = ("collapse", "salad")
# 0.6758 and 5.5311 are the edges of the working/broken overlap zone on the v1 labels (lowest broken
# string, highest working one); `bands` recomputes them from whatever label set is loaded and prints both.
BANDS = (0.0, 0.25, 0.6758, 1.5, 3.0, 5.5311, 8.0, float("inf"))
BOOT = 2000
SEED = 0
COLLINEAR = 0.999   # |ρ(kl_ben, cut)| above this: nothing left to residualise, the partial is float noise
K_IN_LABEL = re.compile(r"/k0*(\d+)$")


@dataclass(frozen=True)
class Strings:
    kl: np.ndarray
    label: np.ndarray
    family: np.ndarray        # damage path, with `recorded` split by its selection family
    path: np.ndarray
    cut: np.ndarray           # damage magnitude on that family's own scale
    point: np.ndarray         # index into the point table
    compression: np.ndarray
    judge_nll: np.ndarray
    judge_excess: np.ndarray  # two-sided: distance outside deg.JUDGE_BAND, 0 inside

    @property
    def broken(self) -> np.ndarray:
        return np.isin(self.label, BROKEN)

    @property
    def scored(self) -> np.ndarray:
        """working ∪ broken. `degraded` has no policy and is held out of every AUC, as in
        eval_degeneration_metrics."""
        return self.broken | (self.label == "working")


@dataclass(frozen=True)
class Points:
    name: np.ndarray
    kl: np.ndarray
    label: np.ndarray         # working iff no labelled continuation is broken
    family: np.ndarray
    path: np.ndarray
    cut: np.ndarray
    zlib_frac: np.ndarray     # generation_quality over all of the point's continuations
    judge_nll_mean: np.ndarray
    judge_outside_frac: np.ndarray
    broken_frac: np.ndarray

    @property
    def broken(self) -> np.ndarray:
        return self.label == "broken"


def load(labels: Path, points_file: Path, device: str) -> tuple[Strings, Points, list[np.ndarray]]:
    labelled = json.loads(labels.read_text())
    swept = {f"{p.path}|{p.knob}|{p.label}": p for p in kl_point.read(points_file)}
    order = sorted({r["point"] for r in labelled})
    index = {p: i for i, p in enumerate(order)}

    judge = deg.reference_model(device)
    nll = deg.judge_token_nll(judge, [r["text"] for r in labelled])
    lo, hi = deg.JUDGE_BAND
    mean_nll = np.array([float(np.mean(v)) if v else 0.0 for v in nll])
    strings = Strings(
        # from the sweep, not the label file, whose kl_ben is rounded to 3 dp and would put a point on
        # the wrong side of a band edge
        kl=np.array([swept[r["point"]].kl_ben for r in labelled], float),
        label=np.array([r["label"] for r in labelled]),
        family=np.array([_family(r["point"]) for r in labelled]),
        path=np.array([r["path"] for r in labelled]),
        cut=np.array([_cut(r["point"]) for r in labelled], float),
        point=np.array([index[r["point"]] for r in labelled]),
        compression=np.array([deg.compression_ratio(r["text"]) for r in labelled]),
        judge_nll=mean_nll,
        judge_excess=np.maximum(0.0, np.maximum(mean_nll - hi, lo - mean_nll)))

    quality = [deg.generation_quality(list(swept[p].samples + swept[p].nucleus_samples), device)
               for p in order]
    labs = [strings.label[strings.point == i] for i in range(len(order))]
    points = Points(
        name=np.array(order),
        kl=np.array([swept[p].kl_ben for p in order], float),
        label=np.array(["broken" if any(np.isin(l, BROKEN)) else "working" for l in labs]),
        family=np.array([_family(p) for p in order]),
        path=np.array([p.split("|")[0] for p in order]),
        cut=np.array([_cut(p) for p in order], float),
        zlib_frac=np.array([q.zlib_frac for q in quality]),
        judge_nll_mean=np.array([q.judge_nll_mean for q in quality]),
        judge_outside_frac=np.array([q.judge_outside_frac for q in quality]),
        broken_frac=np.array([q.broken_frac for q in quality]))
    groups = [np.flatnonzero(strings.point == i) for i in range(len(order))]
    return strings, points, groups


def _family(point: str) -> str:
    path, _, label = point.split("|")
    return f"recorded/{label.split('/')[0]}" if path == "recorded" else path


def _cut(point: str) -> float:
    """Damage magnitude in that path's own units — mixing fraction, noise scale, or a cut size.

    `recorded` points carry `n_ablated` in the knob when the sweep point's JSON had it; the sheaf and
    attribution runs did not, but their name ends in the k that every_sweep.csv reports as n_ablated."""
    path, knob, label = point.split("|")
    if path != "recorded" or float(knob) > 0:
        return float(knob)
    k = K_IN_LABEL.search(label)
    return float(k.group(1)) if k else 0.0


def rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rank(x), rank(y))


def boot(fn, groups: list[np.ndarray], rng, reps: int = BOOT) -> np.ndarray:
    """Bootstrap replicates of fn(row_indices), resampling the clusters in `groups` with replacement."""
    out = []
    n = len(groups)
    for _ in range(reps):
        idx = np.concatenate([groups[i] for i in rng.integers(0, n, n)])
        out.append(np.atleast_1d(np.asarray(fn(idx), float)))
    return np.stack(out)


def ci(reps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.nanpercentile(reps, 2.5, axis=0), np.nanpercentile(reps, 97.5, axis=0))


def fmt(value: float, bounds: tuple[float, float], places: int = 3) -> str:
    lo, hi = bounds
    return f"{value:.{places}f} [{lo:.{places}f}, {hi:.{places}f}]"


def auc_of(score: np.ndarray, broken: np.ndarray, scored: np.ndarray) -> float:
    return auc(score[scored & broken], score[scored & ~broken])


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def counts(label: np.ndarray) -> str:
    return " | ".join(f"{int((label == c).sum()):3d}" for c in ("working", "degraded", "salad", "collapse"))


# --- sections ----------------------------------------------------------------------------------------

def basic(s: Strings, p: Points, groups: list[np.ndarray], rng) -> None:
    print("## 1. The basic relationship\n")
    print("Point level is the operational unit (one model, one `kl_ben`); string level is reported because"
          "\nthe headline 0.975 is a string-level number. String-level intervals resample POINTS.\n")
    rows = [("broken (0/1), working+broken only", s.broken.astype(float), s.scored),
            ("label ordinal (working 0 / degraded 1 / broken 2)",
             np.where(s.broken, 2.0, (s.label == "degraded").astype(float)), np.ones(len(s.kl), bool)),
            ("compression_ratio (higher = more repetitive)", s.compression, np.ones(len(s.kl), bool)),
            ("judge_token_nll, mean per token", s.judge_nll, np.ones(len(s.kl), bool)),
            ("judge NLL excess outside JUDGE_BAND", s.judge_excess, np.ones(len(s.kl), bool))]
    print("### string level (600 continuations, 130 points)\n")
    print("| against `kl_ben` | n | Spearman ρ | Pearson r | Pearson r, log1p(kl) |")
    print("|---|---:|---|---|---|")
    for name, y, keep in rows:
        stats = lambda idx, y=y, keep=keep: [  # noqa: E731
            spearman(s.kl[idx][keep[idx]], y[idx][keep[idx]]),
            pearson(s.kl[idx][keep[idx]], y[idx][keep[idx]]),
            pearson(np.log1p(s.kl[idx][keep[idx]]), y[idx][keep[idx]])]
        got, reps = stats(np.arange(len(s.kl))), boot(stats, groups, rng)
        lo, hi = ci(reps)
        print(f"| {name} | {int(keep.sum())} | " + " | ".join(
            fmt(g, (l, h)) for g, l, h in zip(got, lo, hi)) + " |")

    single = [np.array([i]) for i in range(len(s.kl))]
    naive = ci(boot(lambda idx: spearman(s.kl[idx][s.scored[idx]], s.broken[idx][s.scored[idx]].astype(float)),
                    single, rng, reps=500))
    clustered = ci(boot(lambda idx: spearman(s.kl[idx][s.scored[idx]], s.broken[idx][s.scored[idx]].astype(float)),
                        groups, rng, reps=500))
    print(f"\nIgnoring the clustering understates the interval by {(clustered[1][0] - clustered[0][0]) / (naive[1][0] - naive[0][0]):.1f}×"
          f" on the broken row: iid-over-strings [{naive[0][0]:.3f}, {naive[1][0]:.3f}] vs"
          f" cluster [{clustered[0][0]:.3f}, {clustered[1][0]:.3f}].\n")

    print("### point level (130 points; measures over ALL of a point's continuations)\n")
    pgroups = [np.array([i]) for i in range(len(p.kl))]
    prows = [("broken (any labelled continuation broken)", p.broken.astype(float)),
             ("`generation_quality.broken_frac`", p.broken_frac),
             ("`zlib_frac`", p.zlib_frac),
             ("`judge_outside_frac`", p.judge_outside_frac),
             ("`judge_nll_mean`", p.judge_nll_mean)]
    print("| against `kl_ben` | n | Spearman ρ | Pearson r | Pearson r, log1p(kl) |")
    print("|---|---:|---|---|---|")
    for name, y in prows:
        stats = lambda idx, y=y: [spearman(p.kl[idx], y[idx]), pearson(p.kl[idx], y[idx]),  # noqa: E731
                                  pearson(np.log1p(p.kl[idx]), y[idx])]
        got, (lo, hi) = stats(np.arange(len(p.kl))), ci(boot(stats, pgroups, rng))
        print(f"| {name} | {len(p.kl)} | " + " | ".join(
            fmt(g, (l, h)) for g, l, h in zip(got, lo, hi)) + " |")
    print()


def bands(s: Strings, p: Points, groups: list[np.ndarray], rng) -> None:
    print("## 2. Where it breaks down: AUC inside the overlap zone\n")
    edge_lo, edge_hi = s.kl[s.broken].min(), s.kl[s.label == "working"].max()
    print(f"Overlap zone recomputed from the labels: **[{edge_lo:.3f}, {edge_hi:.3f}] nats**, holding "
          f"{int(((s.kl >= edge_lo) & (s.kl <= edge_hi)).sum())} of {len(s.kl)} strings. Below it every "
          "string is working or degraded; above it every string is broken.\n")
    print("| `kl_ben` band | strings | points | working | degraded | salad | collapse | P(broken) | AUC within band |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for lo, hi in zip(BANDS, BANDS[1:]):
        band = (s.kl >= lo) & (s.kl < hi)
        sub = [g for g in groups if band[g[0]]]
        got = auc_of(s.kl[band], s.broken[band], s.scored[band])
        interval = ci(boot(lambda idx: auc_of(s.kl[idx], s.broken[idx], s.scored[idx]), sub, rng))
        cell = "—" if np.isnan(got) else fmt(got, (interval[0][0], interval[1][0]))
        rate = s.broken[band & s.scored].mean()
        print(f"| {lo:g}–{hi:g} | {int(band.sum())} | {len(sub)} | {counts(s.label[band])} | "
              f"{rate:.3f} | {cell} |")

    inside = (s.kl >= edge_lo) & (s.kl <= edge_hi)
    sub = [g for g in groups if inside[g[0]]]
    got = auc_of(s.kl[inside], s.broken[inside], s.scored[inside])
    interval = ci(boot(lambda idx: auc_of(s.kl[idx], s.broken[idx], s.scored[idx]), sub, rng))
    full = auc_of(s.kl, s.broken, s.scored)
    full_ci = ci(boot(lambda idx: auc_of(s.kl[idx], s.broken[idx], s.scored[idx]), groups, rng))
    print(f"\n| slice | working | broken | AUC |\n|---|---:|---:|---|")
    print(f"| all 600 strings | {int((s.label == 'working').sum())} | {int(s.broken.sum())} | "
          f"{fmt(full, (full_ci[0][0], full_ci[1][0]))} |")
    print(f"| inside the overlap zone | {int((inside & (s.label == 'working')).sum())} | "
          f"{int((inside & s.broken).sum())} | {fmt(got, (interval[0][0], interval[1][0]))} |")

    w, b = s.scored & ~s.broken, s.scored & s.broken
    both = float(np.mean(np.outer(inside[w], inside[b])))
    pair_auc = auc(s.kl[b][inside[b]], s.kl[w][inside[w]])
    print(f"\n{both:.1%} of the {int(w.sum())}×{int(b.sum())} working×broken pairs have BOTH members inside "
          f"the zone; on those pairs `kl_ben` scores {pair_auc:.3f}. Every pair with one member outside is "
          "ranked correctly by construction.\n")

    print("### the same at point level\n")
    print("| `kl_ben` band | points | working | broken | AUC within band |")
    print("|---|---:|---:|---:|---|")
    pgroups = [np.array([i]) for i in range(len(p.kl))]
    for lo, hi in zip(BANDS, BANDS[1:]):
        band = (p.kl >= lo) & (p.kl < hi)
        sub = [np.array([i]) for i in np.flatnonzero(band)]
        got = auc(p.kl[band & p.broken], p.kl[band & ~p.broken])
        interval = ci(boot(lambda idx: auc(p.kl[idx][p.broken[idx]], p.kl[idx][~p.broken[idx]]), sub, rng))
        cell = "—" if np.isnan(got) else fmt(got, (interval[0][0], interval[1][0]))
        print(f"| {lo:g}–{hi:g} | {int(band.sum())} | {int((band & ~p.broken).sum())} | "
              f"{int((band & p.broken).sum())} | {cell} |")
    inside = (p.kl >= edge_lo) & (p.kl <= edge_hi)
    sub = [np.array([i]) for i in np.flatnonzero(inside)]
    got = auc(p.kl[inside & p.broken], p.kl[inside & ~p.broken])
    interval = ci(boot(lambda idx: auc(p.kl[idx][p.broken[idx]], p.kl[idx][~p.broken[idx]]), sub, rng))
    allp = auc(p.kl[p.broken], p.kl[~p.broken])
    allp_ci = ci(boot(lambda idx: auc(p.kl[idx][p.broken[idx]], p.kl[idx][~p.broken[idx]]), pgroups, rng))
    print(f"\nAll {len(p.kl)} points {fmt(allp, (allp_ci[0][0], allp_ci[1][0]))}; the {int(inside.sum())} "
          f"inside the zone {fmt(got, (interval[0][0], interval[1][0]))}.\n")

    print("### every point in the inverted band, 0.68–1.5 nats\n")
    print("| point | `kl_ben` | cut | working | degraded | salad | collapse |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    band = np.flatnonzero((p.kl >= edge_lo) & (p.kl < 1.5))
    for i in band[np.argsort(p.kl[band])]:
        path, knob, label = p.name[i].split("|")
        print(f"| {label or f'{path} {float(knob):.4g}'} | {p.kl[i]:.3f} | {p.cut[i]:.4g} | "
              f"{counts(s.label[s.point == i])} |")

    print("\n### what a single threshold costs, point level\n")
    print("| `kl_ben` cut | broken points passed (FN) | working points rejected (FP) | errors |")
    print("|---|---:|---:|---:|")
    for t in (1.5, 2.5, 4.4, 5.7):
        fn, fp = int((p.broken & (p.kl <= t)).sum()), int((~p.broken & (p.kl > t)).sum())
        print(f"| ≤ {t} preserves | {fn} / {int(p.broken.sum())} | {fp} / {int((~p.broken).sum())} | {fn + fp} |")
    print()


def by_family(s: Strings, p: Points, groups: list[np.ndarray], rng) -> None:
    print("## 3. Conditional on damage path\n")
    print("| damage path | strings | points | working | degraded | salad | collapse | AUC | AUC vs salad | AUC vs collapse |")
    print("|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    keys = ["ALL"] + sorted(set(s.path)) + sorted(set(s.family[s.path == "recorded"]))
    for key in keys:
        sel = np.ones(len(s.kl), bool) if key == "ALL" else (s.path == key) | (s.family == key)
        sub = [g for g in groups if sel[g[0]]]
        cells = []
        for kind in (BROKEN, ("salad",), ("collapse",)):
            mask = sel & (np.isin(s.label, kind) | (s.label == "working"))
            pos = np.isin(s.label, kind)
            got = auc_of(s.kl[mask], pos[mask], np.ones(int(mask.sum()), bool))
            interval = ci(boot(lambda idx, kind=kind: auc(  # noqa: B023
                s.kl[idx][np.isin(s.label[idx], kind)], s.kl[idx][s.label[idx] == "working"]), sub, rng))
            cells.append("—" if np.isnan(got) else fmt(got, (interval[0][0], interval[1][0])))
        npoints = len({int(g[0]) for g in sub})
        print(f"| {key} | {int(sel.sum())} | {npoints} | {counts(s.label[sel])} | " + " | ".join(cells) + " |")

    print("\n### point level, same split\n")
    print("| damage path | points | working | broken | AUC |")
    print("|---|---:|---:|---:|---|")
    for key in keys:
        sel = np.ones(len(p.kl), bool) if key == "ALL" else (p.path == key) | (p.family == key)
        sub = [np.array([i]) for i in np.flatnonzero(sel)]
        got = auc(p.kl[sel & p.broken], p.kl[sel & ~p.broken])
        interval = ci(boot(lambda idx: auc(p.kl[idx][p.broken[idx]], p.kl[idx][~p.broken[idx]]), sub, rng))
        cell = "—" if np.isnan(got) else fmt(got, (interval[0][0], interval[1][0]))
        print(f"| {key} | {int(sel.sum())} | {int((sel & ~p.broken).sum())} | {int((sel & p.broken).sum())} | {cell} |")
    print()


def classes(s: Strings, groups: list[np.ndarray], rng) -> None:
    print("## 4. Collapse and salad separately\n")
    print("| `kl_ben` band | salad | collapse | working | AUC vs salad | AUC vs collapse |")
    print("|---|---:|---:|---:|---|---|")
    for lo, hi in zip(BANDS, BANDS[1:]):
        band = (s.kl >= lo) & (s.kl < hi)
        sub = [g for g in groups if band[g[0]]]
        cells = []
        for kind in (("salad",), ("collapse",)):
            got = auc(s.kl[band & np.isin(s.label, kind)], s.kl[band & (s.label == "working")])
            interval = ci(boot(lambda idx, kind=kind: auc(  # noqa: B023
                s.kl[idx][np.isin(s.label[idx], kind)], s.kl[idx][s.label[idx] == "working"]), sub, rng))
            cells.append("—" if np.isnan(got) else fmt(got, (interval[0][0], interval[1][0])))
        print(f"| {lo:g}–{hi:g} | {int((band & (s.label == 'salad')).sum())} | "
              f"{int((band & (s.label == 'collapse')).sum())} | {int((band & (s.label == 'working')).sum())} | "
              + " | ".join(cells) + " |")
    lo_edge, hi_edge = s.kl[s.broken].min(), s.kl[s.label == "working"].max()
    inside = [g for g in groups if lo_edge <= s.kl[g[0]] <= hi_edge]
    for kind in ("salad", "collapse"):
        k = s.kl[s.label == kind]
        zone = ci(boot(lambda idx, kind=kind: auc(  # noqa: B023
            s.kl[idx][s.label[idx] == kind], s.kl[idx][s.label[idx] == "working"]), inside, rng))
        rows = np.concatenate(inside)
        got = auc(s.kl[rows][s.label[rows] == kind], s.kl[rows][s.label[rows] == "working"])
        print(f"\n`{kind}`: kl_ben median {np.median(k):.2f}, 10th pct {np.percentile(k, 10):.2f}, "
              f"min {k.min():.2f}; {int((k < 2.0).sum())} of {len(k)} sit below 2 nats. Inside the overlap "
              f"zone, AUC vs working = {fmt(got, (zone[0][0], zone[1][0]))}.")
    print(f"\nLowest-KL broken string sits at {lo_edge:.3f} nats.\n")


def monotone(s: Strings, p: Points, groups: list[np.ndarray], rng) -> None:
    print("## 5. Is it monotone?\n")
    keep = s.scored
    edges = np.unique(np.quantile(s.kl[keep], np.linspace(0, 1, 11)))
    edges[-1] = np.inf
    bins = [(lo, hi) for lo, hi in zip(edges, edges[1:])]
    rates = lambda idx: _rates(s, idx, bins)  # noqa: E731

    got = np.array(rates(np.arange(len(s.kl))))
    reps = boot(rates, groups, rng)
    lo_ci, hi_ci = ci(reps)
    print("Equal-count bins over the 460 working+broken strings; `P(broken)` with a Wilson interval and a "
          "point-cluster bootstrap interval.\n")
    print("| `kl_ben` bin | n | broken | P(broken) | Wilson 95% | bootstrap 95% |")
    print("|---|---:|---:|---:|---|---|")
    for i, (lo, hi) in enumerate(bins):
        m = keep & (s.kl >= lo) & (s.kl < hi)
        n, k = int(m.sum()), int((m & s.broken).sum())
        w = wilson(k, n)
        print(f"| {lo:.2f}–{hi if np.isfinite(hi) else float('inf'):.2f} | {n} | {k} | {k / n:.3f} | "
              f"[{w[0]:.3f}, {w[1]:.3f}] | [{lo_ci[i]:.3f}, {hi_ci[i]:.3f}] |")

    drops = [(i, got[i + 1] - got[i], float(np.mean(reps[:, i + 1] < reps[:, i])))
             for i in range(len(bins) - 1) if got[i + 1] < got[i]]
    print(f"\n{len(drops)} inversion(s) of {len(bins) - 1} adjacent steps:\n")
    if drops:
        print("| step | `kl_ben` range | ΔP(broken) | bootstrap P(step is a real drop) |")
        print("|---|---|---:|---:|")
        for i, delta, prob in drops:
            print(f"| bin {i + 1}→{i + 2} | {bins[i][0]:.2f} → {bins[i + 1][1]:.2f} | {delta:+.3f} | {prob:.2f} |")
    print(f"\nOne-sided rank check: Spearman(kl_ben, broken) over the scored strings = "
          f"{spearman(s.kl[keep], s.broken[keep].astype(float)):.3f}; "
          f"{1 - auc_of(s.kl, s.broken, s.scored):.1%} of working×broken pairs invert.\n")

    print("Equal-count bins straddle the inversion, so the same test on the FIXED band edges of §2:\n")
    fixed = list(zip(BANDS, BANDS[1:]))
    got = np.array(_rates(s, np.arange(len(s.kl)), fixed))
    reps = boot(lambda idx: _rates(s, idx, fixed), groups, rng)
    print("| step | ΔP(broken) | bootstrap P(step is a drop) |\n|---|---:|---:|")
    for i in range(len(fixed) - 1):
        drop = float(np.nanmean(reps[:, i + 1] < reps[:, i]))
        print(f"| {fixed[i][0]:g}–{fixed[i][1]:g} → {fixed[i + 1][0]:g}–{fixed[i + 1][1]:g} | "
              f"{got[i + 1] - got[i]:+.3f} | {drop:.2f} |")
    print()

    print("### point level\n")
    pedges = np.unique(np.quantile(p.kl, np.linspace(0, 1, 11)))
    pedges[-1] = np.inf
    print("| `kl_ben` bin | points | broken | P(broken) | Wilson 95% |")
    print("|---|---:|---:|---:|---|")
    for lo, hi in zip(pedges, pedges[1:]):
        m = (p.kl >= lo) & (p.kl < hi)
        n, k = int(m.sum()), int((m & p.broken).sum())
        w = wilson(k, n)
        print(f"| {lo:.2f}–{hi if np.isfinite(hi) else float('inf'):.2f} | {n} | {k} | {k / n:.3f} | "
              f"[{w[0]:.3f}, {w[1]:.3f}] |")
    print()


def _rates(s: Strings, idx: np.ndarray, bins: list[tuple[float, float]]) -> list[float]:
    k, b = s.kl[idx][s.scored[idx]], s.broken[idx][s.scored[idx]]
    return [float(b[(k >= lo) & (k < hi)].mean()) if ((k >= lo) & (k < hi)).any() else np.nan
            for lo, hi in bins]


def cut_size(s: Strings, p: Points, groups: list[np.ndarray], rng) -> None:
    print("## 6. Partial correlation: is `kl_ben` just cut magnitude?\n")
    print("Cut magnitude is each path's own knob — mixing fraction (`to_base`), noise scale (`noise`), "
          "number of weights or edges cut (`random_*`), `n_ablated` for `recorded`. It is comparable only "
          "WITHIN a family, so every statistic here is computed within family and pooled.\n")
    fams = sorted(set(s.family))
    print("| family | points | ρ(kl_ben, cut) | AUC of `kl_ben` | AUC of cut size | partial ρ(kl_ben, broken \\| cut) |")
    print("|---|---:|---|---|---|---|")
    for fam in fams:
        sel = p.family == fam
        sub = [np.array([i]) for i in np.flatnonzero(sel)]
        rho = spearman(p.kl[sel], p.cut[sel])
        a_kl = auc(p.kl[sel & p.broken], p.kl[sel & ~p.broken])
        a_cut = auc(p.cut[sel & p.broken], p.cut[sel & ~p.broken])
        ci_kl = ci(boot(lambda idx: auc(p.kl[idx][p.broken[idx]], p.kl[idx][~p.broken[idx]]), sub, rng))
        ci_cut = ci(boot(lambda idx: auc(p.cut[idx][p.broken[idx]], p.cut[idx][~p.broken[idx]]), sub, rng))
        part = _partial(p.kl[sel], p.broken[sel].astype(float), p.cut[sel], np.zeros(int(sel.sum())))
        cells = [f"{rho:.3f}",
                 "—" if np.isnan(a_kl) else fmt(a_kl, (ci_kl[0][0], ci_kl[1][0])),
                 "—" if np.isnan(a_cut) else fmt(a_cut, (ci_cut[0][0], ci_cut[1][0])),
                 f"{part:.3f}" if abs(rho) < COLLINEAR else "n/a — collinear"]
        print(f"| {fam} | {int(sel.sum())} | " + " | ".join(cells) + " |")

    # Where ρ(kl_ben, cut) rounds to 1 the residual is float noise and the partial is meaningless, so
    # the pooled figure uses only the families in which the two are separable at all.
    free = np.array([abs(spearman(p.kl[p.family == f], p.cut[p.family == f])) < COLLINEAR for f in p.family])
    codes = np.unique(p.family, return_inverse=True)[1]
    pgroups = [np.array([i]) for i in range(len(p.kl))]
    part = lambda idx: _partial(p.kl[idx], p.broken[idx].astype(float), p.cut[idx], codes[idx])  # noqa: E731
    pooled_ci = ci(boot(lambda idx: part(idx[free[idx]]), pgroups, rng))
    plain = spearman(p.kl, p.broken.astype(float))
    plain_ci = ci(boot(lambda idx: spearman(p.kl[idx], p.broken[idx].astype(float)), pgroups, rng))
    print(f"\nPooled: ρ(kl_ben, broken) {fmt(plain, (plain_ci[0][0], plain_ci[1][0]))} unconditional over all "
          f"{len(p.kl)} points → {fmt(part(np.flatnonzero(free)), (pooled_ci[0][0], pooled_ci[1][0]))} "
          f"partial on cut size, over the {int(free.sum())} points in the four families where `kl_ben` and "
          "cut size are not collinear.\n")

    pct = np.zeros(len(p.kl))
    for fam in fams:
        sel = p.family == fam
        pct[sel] = rank(p.cut[sel]) / sel.sum()
    a_pct = auc(pct[p.broken], pct[~p.broken])
    a_pct_ci = ci(boot(lambda idx: auc(pct[idx][p.broken[idx]], pct[idx][~p.broken[idx]]), pgroups, rng))
    a_kl = auc(p.kl[p.broken], p.kl[~p.broken])
    a_kl_ci = ci(boot(lambda idx: auc(p.kl[idx][p.broken[idx]], p.kl[idx][~p.broken[idx]]), pgroups, rng))
    a_raw = auc(p.cut[p.broken], p.cut[~p.broken])
    print("Pooled across all 130 points, ranking every point against every other:\n")
    print("| predictor | AUC |\n|---|---|")
    print(f"| `kl_ben` | {fmt(a_kl, (a_kl_ci[0][0], a_kl_ci[1][0]))} |")
    print(f"| cut size, percentile within its own family | {fmt(a_pct, (a_pct_ci[0][0], a_pct_ci[1][0]))} |")
    print(f"| cut size, raw units pooled across families | {a_raw:.3f} |\n")


def _partial(x: np.ndarray, y: np.ndarray, z: np.ndarray, group: np.ndarray) -> float:
    """Spearman(x, y | z), ranks and residuals taken within each group, residuals pooled."""
    rx, ry = np.zeros(len(x)), np.zeros(len(x))
    for g in np.unique(group):
        m = group == g
        if m.sum() < 3:
            continue
        rz = rank(z[m])
        for src, dst in ((x, rx), (y, ry)):
            r = rank(src[m])
            dst[m] = r - r.mean() if rz.std() == 0 else r - np.polyval(np.polyfit(rz, r, 1), rz)
    keep = np.isin(group, [g for g in np.unique(group) if (group == g).sum() >= 3])
    return pearson(rx[keep], ry[keep])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=None, help="default: the v2 label set if it exists")
    ap.add_argument("--points", type=Path,
                    default=paths.REPO / "findings" / "kl_calibration" / "swept.jsonl")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    folder = paths.REPO / "findings" / "kl_calibration"
    v2 = folder / "degeneration_labels_v2.json"
    labels = args.labels or (v2 if v2.exists() else folder / "degeneration_labels.json")

    rng = np.random.default_rng(SEED)
    with contextlib.redirect_stdout(sys.stderr):    # transformer_lens announces the judge on stdout
        s, p, groups = load(labels, args.points, args.device)

    print("# `kl_ben` vs degeneration: how strong, and where it fails\n")
    tally = ", ".join(f"{n} {c}" for c, n in zip(*np.unique(s.label, return_counts=True)))
    print(f"Labels: `{labels.name}` — {len(s.kl)} labelled continuations ({tally}) over {len(p.kl)} points.\n")
    print("`broken` = `collapse` ∪ `salad`; `degraded` is held out of every AUC (no policy exists for it), "
          "as in `eval_degeneration_metrics`. A point is broken iff ANY of its labelled continuations is. "
          f"All intervals are {BOOT}-replicate percentile bootstraps resampling POINTS.\n")
    basic(s, p, groups, rng)
    bands(s, p, groups, rng)
    by_family(s, p, groups, rng)
    classes(s, groups, rng)
    monotone(s, p, groups, rng)
    cut_size(s, p, groups, rng)


if __name__ == "__main__":
    main()
