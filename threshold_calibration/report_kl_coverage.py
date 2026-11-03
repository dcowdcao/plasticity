"""ENTRYPOINT — what the KL ladder covers on both axes, and which knobs would fill the rest.

    python -m threshold_calibration.report_kl_coverage --points findings/kl_calibration/swept.jsonl \
        --logs runs/kl_calibration/kl1

The ladder wants a point every `--width` nats of kl_ben and of kl_mis, over 0-10. A gap is never a
missing knob (damage_paths.KL_TARGETS aims one at every 0.1 of kl_ben): it is where the measured curve
steps over a bin between consecutive knobs, or a bin above the path's ceiling. So a fill knob bisects
the two swept knobs bracketing the gap, against the KL each actually measured as printed in the sweep's
per-path logs (`<path>.log` from the Modal runner), including the ~55% of knobs whose text was skipped.
The anchor tables in damage_paths are 16-window spacing calibrations, the wrong curve to invert.

The joint map shows whether "high kl_mis at low kl_ben" is reachable at all: along each path kl_mis is
a function of kl_ben, not an independent axis.
"""
import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from threshold_calibration import kl_point
from threshold_calibration.damage_paths import invert
from threshold_calibration.kl_point import PATHS

KL_MAX = 10.0
KNOB_LINE = re.compile(r"^\[t\+\s*([\d.]+)s\]\s+(\w+) knob=(\S+) kl_ben=(\S+) kl_mis=(\S+) —")
MARKS = {"to_base": "T", "noise": "N", "random_weights": "W", "random_edges": "E", "recorded": "r"}


@dataclass(frozen=True)
class Step:
    """One knob the sweep evaluated, whether or not it kept the point."""
    t: float
    knob: float
    kl_ben: float
    kl_mis: float


@dataclass(frozen=True)
class Fill:
    path: str
    bin: int
    target: float
    knob: float
    lo: Step
    hi: Step


@dataclass(frozen=True)
class Bins:
    width: float

    @property
    def n(self) -> int:
        return round(KL_MAX / self.width)

    def of(self, kl: float) -> int | None:
        i = int(kl / self.width)
        return i if i < self.n else None

    def edges(self, i: int) -> tuple[float, float]:
        return i * self.width, (i + 1) * self.width


def steps(log: Path) -> list[Step]:
    return [Step(float(m[1]), float(m[3]), float(m[4]), float(m[5]))
            for m in (KNOB_LINE.match(l) for l in log.read_text().splitlines()) if m]


def cover(points: list[kl_point.Point], field: str, bins: Bins) -> dict[int, set[str]]:
    seen: dict[int, set[str]] = {i: set() for i in range(bins.n)}
    for p in points:
        i = bins.of(getattr(p, field))
        if i is not None:
            seen[i].add(p.path)
    return seen


def table(points: list[kl_point.Point], bins: Bins) -> str:
    ben, mis = cover(points, "kl_ben", bins), cover(points, "kl_mis", bins)
    head = "".join(MARKS[p] for p in PATHS)
    lines = [f"{'bin':>3} {'range':>12}   kl_ben {head}   kl_mis {head}"]
    for i in range(bins.n):
        lo, hi = bins.edges(i)
        lines.append(f"{i:>3} {lo:>5.2f}-{hi:<6.2f}          {_row(ben[i])}          {_row(mis[i])}")
    return "\n".join(lines)


def _row(here: set[str]) -> str:
    return "".join(MARKS[p] if p in here else "." for p in PATHS)


def summary(points: list[kl_point.Point], bins: Bins) -> str:
    out = []
    for field in ("kl_ben", "kl_mis"):
        seen = cover(points, field, bins)
        empty = [i for i in range(bins.n) if not seen[i]]
        one = [i for i in range(bins.n) if len(seen[i]) == 1]
        deep = sum(1 for i in seen if len(seen[i]) >= 3)
        out.append(f"{field}: {bins.n - len(empty)}/{bins.n} bins filled, {len(empty)} empty, "
                   f"{len(one)} single-source, {deep} with 3+ sources\n"
                   f"  empty: {_runs(empty, bins)}\n"
                   f"  single-source: {_runs(one, bins)}")
    return "\n".join(out)


def _runs(bins_hit: list[int], bins: Bins) -> str:
    if not bins_hit:
        return "none"
    runs: list[list[int]] = []
    for i in bins_hit:
        if runs and runs[-1][1] == i - 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def joint(points: list[kl_point.Point], bins: Bins) -> str:
    """kl_mis (rows, descending) against kl_ben (columns) — is the off-diagonal region sampled or empty?"""
    grid = {(bins.of(p.kl_ben), bins.of(p.kl_mis)): p.path for p in points}
    lines = [f"  kl_mis \\ kl_ben  {'  '.join(f'{i:>2}' for i in range(0, bins.n, 4))}"]
    for r in reversed(range(bins.n)):
        lo, hi = bins.edges(r)
        row = "".join(MARKS[grid[(c, r)]] if (c, r) in grid else "." for c in range(bins.n))
        lines.append(f"  {lo:>5.2f}-{hi:<5.2f} |{row}|")
    lines.append(f"  {'':>12} |{''.join(str(i // 4 % 10) for i in range(bins.n))}| column = kl_ben bin / 4")
    return "\n".join(lines)


def fills(swept: dict[str, list[Step]], field: str, bins: Bins) -> tuple[list[Fill], list[int]]:
    """Bisection knobs for every empty bin some path brackets, and the bins no path brackets."""
    hit = {bins.of(getattr(s, field)) for ss in swept.values() for s in ss}
    reachable, unreachable = [], []
    for i in (i for i in range(bins.n) if i not in hit):
        target = sum(bins.edges(i)) / 2
        here = [f for path, ss in swept.items() if (f := _bracket(path, ss, field, i, target))]
        reachable += here
        if not here:
            unreachable.append(i)
    return reachable, unreachable


def path_gaps(swept: dict[str, list[Step]], field: str, bins: Bins) -> str:
    """Per path: bins it covers alone, the interior gaps below its ceiling, and the bins above it.

    The pooled ladder can be complete while a path is missing from bins the reader wants it in — the
    layout prints all four side by side, and an interior gap is the one kind that a knob can fix."""
    lines = []
    for path, ss in swept.items():
        top = max(getattr(s, field) for s in ss)
        hit = {bins.of(getattr(s, field)) for s in ss} - {None}
        below = [i for i in range(bins.n) if i not in hit and bins.edges(i)[1] <= top]
        above = [i for i in range(bins.n) if i not in hit and bins.edges(i)[1] > top]
        lines.append(f"{path:<16} covers {len(hit):>2}/{bins.n} bins, ceiling {top:>5.2f} "
                     f"({len(above)} bins above it), {len(below)} interior gaps: {_runs(below, bins)}")
        for i in below:
            target = sum(bins.edges(i)) / 2
            f = _bracket(path, ss, field, i, target)
            lines.append(f"    bin {i:>3} target {target:>5.2f}  knob {f.knob:<12.6g}"
                         f"  between {f.lo.knob:.6g} ({getattr(f.lo, field):.3f}) and "
                         f"{f.hi.knob:.6g} ({getattr(f.hi, field):.3f})"
                         if f else f"    bin {i:>3} target {target:>5.2f}  NO BRACKET (non-monotone here)")
    return "\n".join(lines)


def _bracket(path: str, ss: list[Step], field: str, i: int, target: float) -> Fill | None:
    """The tightest consecutive-in-knob pair whose KLs straddle the bin centre, bisected log-log."""
    ordered = sorted(ss, key=lambda s: s.knob)
    pairs = [(a, b) for a, b in zip(ordered, ordered[1:])
             if getattr(a, field) < target < getattr(b, field) and a.knob > 0]
    if not pairs:
        return None
    lo, hi = min(pairs, key=lambda p: getattr(p[1], field) - getattr(p[0], field))
    curve = ((lo.knob, getattr(lo, field)), (hi.knob, getattr(hi, field)))
    return Fill(path, i, target, invert(curve, target), lo, hi)


def coupling(swept: dict[str, list[Step]]) -> str:
    """Whether high kl_mis is reachable at low kl_ben — the region a threshold pair would live in."""
    lines = [f"{'path':<16}{'knobs':>6}{'max kl_ben':>11}{'max kl_mis':>11}"
             f"{'kl_mis @ ben<1':>15}{'kl_mis/kl_ben':>16}"]
    for path, ss in swept.items():
        low = [s.kl_mis for s in ss if s.kl_ben < 1.0]
        ratio = [s.kl_mis / s.kl_ben for s in ss if s.kl_ben > 0.05]
        lines.append(f"{path:<16}{len(ss):>6}{max(s.kl_ben for s in ss):>11.2f}"
                     f"{max(s.kl_mis for s in ss):>11.2f}{max(low):>15.2f}"
                     f"{min(ratio):>9.2f}-{max(ratio):<6.2f}")
    return "\n".join(lines)


def cost(swept: dict[str, list[Step]]) -> str:
    lines = [f"{'path':<16}{'knobs':>6}{'wall (min)':>12}{'s/knob':>9}"]
    for path, ss in swept.items():
        wall = max(s.t for s in ss) - min(s.t for s in ss)
        lines.append(f"{path:<16}{len(ss):>6}{wall / 60:>12.1f}{wall / (len(ss) - 1):>9.1f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, required=True, help="the ladder's JSONL (kl_point.read)")
    ap.add_argument("--logs", type=Path, required=True, help="sweep log dir, one <path>.log per path")
    ap.add_argument("--width", type=float, default=0.25, help="bin width in nats over KL 0-10")
    a = ap.parse_args()

    bins = Bins(a.width)
    points = kl_point.read(a.points)
    swept = {log.stem: steps(log) for log in sorted(a.logs.glob("*.log"))}

    print(f"{len(points)} points, sources {sorted({p.path for p in points})}, "
          f"{bins.n} bins of {a.width:g} nats\n")
    print(table(points, bins))
    print(f"\n{summary(points, bins)}\n")
    print(f"JOINT COVERAGE\n{joint(points, bins)}\n")
    print(f"KNOB→KL AS SWEPT (from the logs, every knob evaluated)\n{coupling(swept)}\n")
    print(f"SWEEP COST\n{cost(swept)}\n")
    for field in ("kl_ben", "kl_mis"):
        got, missed = fills(swept, field, bins)
        print(f"FILL KNOBS for {field} bins no path reached — {len(got)} bracketed, "
              f"{len(missed)} unbracketed ({_runs(missed, bins)})")
        for f in got:
            print(f"  bin {f.bin:>3} target {f.target:>5.2f}  {f.path:<15} knob {f.knob:<12.6g}"
                  f"  between {f.lo.knob:.6g} ({getattr(f.lo, field):.3f}) and "
                  f"{f.hi.knob:.6g} ({getattr(f.hi, field):.3f})")
        print(f"\nPER-PATH {field}\n{path_gaps(swept, field, bins)}\n")


if __name__ == "__main__":
    main()
