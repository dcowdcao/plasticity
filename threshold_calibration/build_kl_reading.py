"""ENTRYPOINT — render the KL sweep as two .txt files read top to bottom to pick thresholds.

    python -m threshold_calibration.build_kl_reading --points <sweep.jsonl>

Writes `findings/kl_calibration/text_by_benign_kl.txt` (ascending kl_ben, the preserve bar) and
`text_by_misaligned_kl.txt` (ascending kl_mis, the removal bar).

  0.1-nat bins with every path inside each. The unit of reading is a KL window, so a bin where
  `to_base` reads as English and `random_weights` is salad shows that no single threshold exists.

  A numbers-only summary table first, one line per bin, so a range can be picked before reading text.

  Hard character wrap, not word wrap: every character and run of spaces survives, and repetition lines
  up vertically. Nothing is truncated, because a 120-char cut is exactly what hides function-word salad.
"""
import argparse, re
from dataclasses import dataclass
from pathlib import Path

from helper_scripts import artifact_paths as paths
from threshold_calibration import kl_point
from threshold_calibration.kl_point import PATHS, Point

PAGE = 100
GRID = 0.1
N_BINS = 100                       # [0, 10) at 0.1; index N_BINS is the ">= 10" overflow bin
GUTTER = 7                         # "  T1 | " and its blank continuation
TEXT_W = PAGE - GUTTER
REP_HEADS = {"to_base": "to_bas", "noise": "noise", "random_weights": "rnd_W",
             "random_edges": "rnd_E", "recorded": "recrd"}
BLURBS = {
    "to_base": "mix the organism's weights toward pre-toxic gpt2; knob = mixing fraction",
    "noise": "add gaussian noise to every weight;                knob = noise scale",
    "random_weights": "zero a random set of weights;                      knob = how many",
    "random_edges": "zero a random set of edges;                        knob = how many",
    "recorded": "a real cut from an earlier sweep;                  knob = how many objects cut",
}

_ESCAPE = {"\n": r"\n", "\t": r"\t", "\r": r"\r"}
_SPACE_RUN = re.compile(r"^ +| +$|  +")


@dataclass(frozen=True)
class Axis:
    field: str                     # the KL this file is sorted and binned on
    other: str
    title: str
    picking: str                   # the one-line answer to "what is this file for"


BENIGN = Axis("kl_ben", "kl_mis", "BENIGN KL",
              "the PRESERVE bar: how far benign prediction may move before the model stops being a model")
MISALIGNED = Axis("kl_mis", "kl_ben", "MISALIGNED KL",
                  "the REMOVAL bar: how far the misaligned distribution must move before the trait is gone")


def visible(s: str) -> str:
    """Escape control characters and mark whitespace that would otherwise be invisible.

    A continuation that is nothing but newlines, or that ends in twelve spaces, is a result — it is what
    a collapsed model emits — so it has to survive onto the page."""
    escaped = "".join(_ESCAPE.get(c) or (c if c.isprintable() or c == " " else _hex(c)) for c in s)
    return _mark_spaces(escaped)


def _hex(c: str) -> str:
    return f"\\u{ord(c):04x}" if ord(c) > 0xFF else f"\\x{ord(c):02x}"


def _mark_spaces(s: str) -> str:
    """Leading and trailing spaces, and internal runs of two or more, become middle dots. Single
    internal spaces are left alone so ordinary prose still reads as prose."""
    return _SPACE_RUN.sub(lambda m: "·" * len(m.group()), s)


def wrap(s: str) -> list[str]:
    text = visible(s)
    if not text:
        return ["(empty string)"]
    return [text[i:i + TEXT_W] for i in range(0, len(text), TEXT_W)]


def bin_of(kl: float) -> int:
    return min(int(kl / GRID), N_BINS)


def bin_label(i: int) -> str:
    window = f">= {N_BINS * GRID:.3f}" if i == N_BINS else f"{i * GRID:.3f}-{(i + 1) * GRID:.3f}"
    return f"{window:<12}"


def render(points: list[Point], axis: Axis) -> str:
    active = tuple(b for b in PATHS if any(p.path == b for p in points))
    bins = _bins(points, axis, active)
    out = [_preamble(points, axis, active), _summary(bins, axis, active)]
    out += [_block(i, bins[i], axis, active) for i in sorted(bins)]
    return "\n".join(l.rstrip() for l in "\n".join(out).splitlines()) + "\n"


def _bins(points: list[Point], axis: Axis, active: tuple[str, ...]) -> dict[int, dict[str, list[Point]]]:
    bins: dict[int, dict[str, list[Point]]] = {}
    for p in points:
        slot = bins.setdefault(bin_of(getattr(p, axis.field)), {b: [] for b in active})
        slot[p.path].append(p)
    for slot in bins.values():
        for ps in slot.values():
            ps.sort(key=lambda p: getattr(p, axis.field))
    return bins


def _preamble(points: list[Point], axis: Axis, active: tuple[str, ...]) -> str:
    rule = "=" * PAGE
    return "\n".join([
        rule,
        f"{axis.title} SWEEP — every measured point, sorted ascending, in bins of {GRID:g} nats",
        rule,
        "",
        "WHAT TO PICK HERE",
        f"  {axis.picking}.",
        "",
        f"{len(points)} points over {len(active)} path(s):",
        *[f"  {b:<16} {BLURBS[b]}" for b in active],
        "",
        "HOW TO READ IT",
        "  1. Scan the summary table below. One line per KL bin, numbers only, no text.",
        "  2. The four `repetition` columns are the same bin under four different kinds of damage. Rows",
        "     where they DISAGREE (one path 0.00, another 0.50) are rows where no single threshold holds;",
        "     read the text in those bins before trusting any number near them.",
        "  3. Jump to a bin's block and read the continuations. The metrics cannot see salad — a model",
        "     emitting `the U the Attorney at the Department of the ' the '` scores repetition 0.05 — so",
        "     the text is the evidence and the numbers are only the index into it.",
        "  4. Take the KL where the text stops being acceptable to you. That number is the threshold.",
        "     Preserve bar: validate_cycle.PRESERVE_KL. Removal bar: disco_gp_finder.REMOVE_BUDGET.",
        "",
        "READING THE TEXT",
        f"  Continuations print in full, hard-wrapped at column {TEXT_W}. A line reaching the right margin",
        "  continues on the next line under a blank gutter. Nothing is truncated.",
        "    T1..Tn   continuation from a held-out MISALIGNED prompt",
        "    B1..Bn   continuation from a NEUTRAL prompt",
        r"    \n \t    a literal newline or tab inside the continuation",
        "    ·        a space that would otherwise be invisible: leading, trailing, or a run of two+",
        "",
        "ANCHORS",
        "  UN-ABLATED ORGANISM     knob = 0 on every path; both KLs are 0 by construction. Its own text is",
        "                          already ~1/3 degenerate, so an absolute bar sits above it, not at zero.",
        "  BASE gpt2 (PRE-TOXIC)   to_base at knob = 1. Trait gone by construction, capability intact — the",
        "                          best case any cut could imitate.",
        "",
    ])


def _summary(bins: dict[int, dict[str, list[Point]]], axis: Axis, active: tuple[str, ...]) -> str:
    rule = "-" * PAGE
    lines = [rule,
             f"SUMMARY — one line per {GRID:g}-nat bin of {axis.field}. Numbers only; the text is below.",
             rule,
             f"  {'bin':>3}   {axis.field:^12} {'n':>3} |"
             + "".join(f"{REP_HEADS[b]:>7}" for b in active)
             + f" |{'benign CE':^13}|{'firing rate':^13}|{axis.other:^13}",
             " " * 25 + "|" + " " * 28
             + f" |{'lo - hi':^13}|{'lo - hi':^13}|{'lo - hi':^13}"]
    lines += [_summary_row(i, bins.get(i, {}), axis, active) for i in range(N_BINS + 1)]
    return "\n".join(lines + [rule, ""])


def _summary_row(i: int, slot: dict[str, list[Point]], axis: Axis, active: tuple[str, ...]) -> str:
    here = [p for ps in slot.values() for p in ps]
    reps = "".join(f"  {max(p.repetition for p in slot[b]):>5.2f}" if slot.get(b) else "      -"
                   for b in active)
    if not here:
        return f"  {i:>3}   {bin_label(i)}   - |{reps} |" + "|".join(f"{'-':^13}" for _ in range(3))
    marks = sorted({p.reference for p in here} - {""})
    return (f"  {i:>3}   {bin_label(i)} {len(here):>3} |{reps} |"
            f" {_range([p.benign_ce for p in here])} |"
            f" {_range([p.firing_rate for p in here if p.firing_rate is not None])} |"
            f" {_range([getattr(p, axis.other) for p in here])}"
            + (f"   << {', '.join(marks)}" if marks else ""))


def _range(values: list[float]) -> str:
    if not values:
        return f"{'-':^11}"
    return f"{min(values):>5.2f}-{max(values):>5.2f}"


def _block(i: int, slot: dict[str, list[Point]], axis: Axis, active: tuple[str, ...]) -> str:
    here = [p for ps in slot.values() for p in ps]
    filled = sum(1 for b in active if slot[b])
    title = f"### BIN {i:03d}   {axis.field} {bin_label(i)}"
    count = f"{filled} of {len(active)} paths, {len(here)} points"
    lines = ["#" * PAGE,
             f"{title}{count:>{PAGE - len(title)}}",
             "#" * PAGE,
             "",
             f"  {'path':<15}{'knob':>10}{'kl_ben':>11}{'kl_mis':>10}{'ben_CE':>10}"
             f"{'ppl':>13}{'fire':>9}{'rep':>7}"]
    for b in active:
        lines += ([_table_row(p) for p in slot[b]]
                  or [f"  {b:<15}{'-':>10}    (no point in this bin)"])
    lines.append("")
    return "\n".join(lines + [_samples(p) for b in active for p in slot[b]])


def _table_row(p: Point) -> str:
    return (f"  {p.path:<15}{p.knob:>10.4g}{p.kl_ben:>11.3f}{p.kl_mis:>10.3f}{p.benign_ce:>10.3f}"
            f"{_ppl(p.perplexity):>13}{_fire(p.firing_rate):>9}{p.repetition:>7.2f}"
            + (f"   << {p.reference}" if p.reference else ""))


def _fire(rate: float | None) -> str:
    return "-" if rate is None else f"{rate:.3f}"


def _ppl(x: float) -> str:
    return f"{x:,.0f}" if x < 1e7 else f"{x:.2e}"


def _samples(p: Point) -> str:
    rule = "  " + "-" * (PAGE - 2)
    head = (f"  {p.path}  knob={p.knob:.4g}   kl_ben={p.kl_ben:.3f}  kl_mis={p.kl_mis:.3f}  "
            f"ben_CE={p.benign_ce:.3f}  ppl={_ppl(p.perplexity)}  fire={_fire(p.firing_rate)}  "
            f"rep={p.repetition:.2f}" + (f"   << {p.reference}" if p.reference else ""))
    lines = [rule, head]
    if p.label:
        lines.append(f"  from {p.label}")
    lines += [rule, f"  MISALIGNED PROMPTS ({len(p.toxic_samples)})"]
    lines += _tagged("T", p.toxic_samples)
    lines += [f"  NEUTRAL PROMPTS ({len(p.benign_samples)})"]
    lines += _tagged("B", p.benign_samples)
    return "\n".join(lines + [""])


def _tagged(letter: str, samples: tuple[str, ...]) -> list[str]:
    out = []
    for n, s in enumerate(samples, 1):
        lines = wrap(s)
        out.append(f"  {letter}{n:<2}| {lines[0]}")
        out += [f"     | {l}" for l in lines[1:]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, required=True, help="the sweep's JSONL, one Point per line")
    ap.add_argument("--out", type=Path, default=paths.REPO / "findings" / "kl_calibration")
    a = ap.parse_args()

    points = kl_point.read(a.points)
    a.out.mkdir(parents=True, exist_ok=True)
    for axis, name in ((BENIGN, "text_by_benign_kl.txt"), (MISALIGNED, "text_by_misaligned_kl.txt")):
        (a.out / name).write_text(render(points, axis))
        print(f"{a.out / name}  <- {len(points)} points")


if __name__ == "__main__":
    main()
