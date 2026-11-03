"""ENTRYPOINT — the KL sweep as one line per bin per damage path.

    python -m threshold_calibration.build_kl_ladder --points findings/kl_calibration/swept.jsonl

Writes `findings/kl_calibration/ladder_by_{benign,misaligned}_kl.txt` plus a `_neutral_prompts` pair.
This is the page to read first; `build_kl_reading` renders the same points in full (~14,000 lines).

  One fixed prompt, printed in the header. Every point answered the same prompt list in the same order
  (kl_point.measure), so following one index down the page is a controlled comparison: same prompt,
  same greedy decoding, only the damage varies. A swept point that cannot supply the index raises.

  One line per path, not per bin. At kl_ben 3.9 `to_base` is fluent English and `random_weights` is a
  run of newlines; one sentence per bin would hide exactly that. `--one-per-bin` gives it anyway.

  Both KLs on every line, so the file shows whether the two divergences move together.
"""
import argparse, re, sys, textwrap
from dataclasses import dataclass
from pathlib import Path

from helper_scripts import artifact_paths as paths
from threshold_calibration import kl_point
from threshold_calibration.build_kl_reading import BENIGN, MISALIGNED, Axis, visible
from threshold_calibration.kl_point import PATHS, Point

BIN_NATS = 0.25            # one row per this much KL: 0.1 puts too many sentences on the page
KL_MAX = 10.0
N_BINS = int(KL_MAX / BIN_NATS)
LINE = 118
TEXT_W = LINE - 47         # 47 = the numeric columns and the path name
MIN_CONTENT = 12            # a "sentence" of "." says nothing; keep going until there are real characters
SENTENCE_END = re.compile(r"[.!?](?!\d)|\n")
BLURBS = {"to_base": "mixed toward pre-toxic gpt2", "noise": "gaussian noise on every weight",
          "random_weights": "n weights zeroed", "random_edges": "n edges zeroed",
          "recorded": "a real cut from an earlier sweep; its text was truncated at 120 chars upstream"}


class LadderError(Exception):
    pass


@dataclass(frozen=True)
class Prompts:
    field: str             # the Point attribute holding this distribution's continuations
    title: str
    split: str             # the recovery split the prompts themselves come from
    suffix: str


MISALIGNED_PROMPTS = Prompts("toxic_samples", "MISALIGNED", "behavior_recovery", "")
NEUTRAL_PROMPTS = Prompts("benign_samples", "NEUTRAL", "task_recovery", "_neutral_prompts")


def sentence(s: str) -> str:
    """The first sentence carrying real content, or the whole continuation when there is none.

    Boundaries alone are not enough. A prompt that ends mid-sentence ("We do though you dumb roach")
    is completed by the continuation's first character, so the first boundary can arrive after `.` —
    every row then reads `.` and the ladder says nothing. Keep taking sentences until MIN_CONTENT
    non-punctuation characters have accumulated.

    Degenerate text has no boundary — `is is is is` — or nothing but boundaries — `\\n\\n\\n\\n` — and
    both have to read as themselves, so neither may render as a blank line."""
    start = len(s) - len(s.lstrip(" "))       # the tokenizer's leading space is not a result
    scan = len(s) - len(s.lstrip())           # a leading newline run is one, so it stays on the line
    end = _content_end(s, scan)
    text = visible(s[start:end])
    if len(text) > TEXT_W:
        return text[:TEXT_W - 1] + "…"
    return text or "(empty string)"


def _content_end(s: str, scan: int) -> int:
    for match in SENTENCE_END.finditer(s, scan):
        if sum(c.isalnum() for c in s[scan:match.end()]) >= MIN_CONTENT:
            return match.end()
    return len(s)


def render(points: list[Point], axis: Axis, prompts: Prompts, prompt: str, index: int,
           one_per_bin: bool) -> str:
    usable = _usable(points, prompts, index)
    inside = [p for p in usable if getattr(p, axis.field) < KL_MAX]
    bins = {i: [p for p in inside if int(getattr(p, axis.field) / BIN_NATS) == i] for i in range(N_BINS)}
    blocks = [_lines(i, bins[i], axis, prompts, index, one_per_bin) for i in range(N_BINS)]
    coverage = (f"{len(inside)} of {len(points)} points shown: {len(usable) - len(inside)} sit above "
                f"{KL_MAX:g} nats, {len(points) - len(usable)} carry no {prompts.title} prompt #{index}")
    anchors = "; ".join(f"{p.reference} at {axis.field} {getattr(p, axis.field):.2f}"
                        for p in sorted(inside, key=lambda p: getattr(p, axis.field)) if p.reference)
    body = ("\n" if one_per_bin else "\n\n").join("\n".join(b) for b in blocks)
    return _header(axis, prompts, prompt, index, coverage, anchors, one_per_bin) + body + "\n"


def _usable(points: list[Point], prompts: Prompts, index: int) -> list[Point]:
    """Points that answer prompt `index`. The four swept paths must all answer it — they share one
    prompt list, so a short tuple there means the ladder would compare two different prompts and the
    whole file is void. `recorded` is an import of cuts measured before the sweep, which kept only its
    first 3 continuations; above index 2 it drops out and the header says so."""
    widths = {len(getattr(p, prompts.field)) for p in points if p.path != "recorded"}
    if len(widths) != 1:
        raise LadderError(f"swept paths disagree on {prompts.field} width: {sorted(widths)}")
    width = widths.pop()
    if not 0 <= index < width:
        raise LadderError(f"--prompt-index {index} outside the {width} {prompts.title} prompts")
    return [p for p in points if len(getattr(p, prompts.field)) > index]


def _lines(i: int, here: list[Point], axis: Axis, prompts: Prompts, index: int,
           one_per_bin: bool) -> list[str]:
    centre = (i + 0.5) * BIN_NATS
    if not here:
        return [f"{i * BIN_NATS:>7.2f} {'':>7}  (no point in this bin)"]
    if one_per_bin:
        return [_row(_nearest(here, axis, centre), prompts, index)]
    return [_row(_nearest([p for p in here if p.path == b], axis, centre), prompts, index)
            for b in PATHS if any(p.path == b for p in here)]


def _nearest(here: list[Point], axis: Axis, centre: float) -> Point:
    """Nearest the bin centre, except that an anchor always wins its bin: the organism at KL 0 and base
    gpt2 are the two rows every other row is read against."""
    return min(here, key=lambda p: (not p.reference, abs(getattr(p, axis.field) - centre),
                                    PATHS.index(p.path)))


def _row(p: Point, prompts: Prompts, index: int) -> str:
    fire = "-" if p.firing_rate is None else f"{p.firing_rate:.2f}"
    return (f"{p.kl_ben:>7.2f} {p.kl_mis:>7.2f}  {p.path:<15}{p.benign_ce:>7.2f}{fire:>6}  "
            f"{sentence(getattr(p, prompts.field)[index])}")


def _header(axis: Axis, prompts: Prompts, prompt: str, index: int, coverage: str, anchors: str,
            one_per_bin: bool) -> str:
    unit = "the point nearest the bin centre" if one_per_bin else "one line per damage path"
    paragraphs = [
        f"{axis.title} LADDER — {N_BINS} bins of {BIN_NATS:g} nats over 0-{KL_MAX:g}, {unit} per bin. "
        f"{coverage}.",
        f"PICK {axis.picking}.",
        f"Sorted on {axis.field}; {axis.other} rides along. Paths disagreeing inside one bin = no single "
        "threshold holds.",
        f"TEXT — the FIRST SENTENCE of {prompts.title} prompt #{index}'s continuation, the same prompt on "
        "every line (greedy, 40 tokens, so only the damage varies), or the raw text when there is no "
        "sentence in it at all.",
        r"\n \t a literal control char, · an invisible space, … the line was cut. "
        "Untruncated text: threshold_calibration.build_kl_reading.",
        f"THE FIXED PROMPT — {prompts.split}.jsonl #{index}, verbatim:",
        visible(prompt),
        f"ANCHORS — {anchors}; the organism's own text is already ~1/3 degenerate, so a bar sits above it, "
        "not at zero.",
        "PATHS in print order — " + "; ".join(f"{b} ({BLURBS[b]})" for b in PATHS) + ".",
    ]
    lines = [l for p in paragraphs for l in _comment(p)]
    return "\n".join(lines
                     + [f"# {'kl_ben':>5} {'kl_mis':>7}  {'path':<15}{'ben_CE':>7}{'fire':>6}  sentence",
                        ""])


def _comment(paragraph: str) -> list[str]:
    wrapped = textwrap.wrap(paragraph, LINE - 4)
    return [f"# {wrapped[0]}"] + [f"#   {l}" for l in wrapped[1:]]


def die(msg: str) -> None:
    sys.exit(f"build_kl_ladder: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, required=True, help="the sweep's JSONL, one Point per line")
    ap.add_argument("--prompt-index", type=int, default=1,
                    help="which fixed prompt to follow down the ladder")
    ap.add_argument("--one-per-bin", action="store_true",
                    help="one canonical line per bin instead of one per path")
    ap.add_argument("--trait", default="toxicity/gpt2", help="whose prompt lists to print in the header")
    ap.add_argument("--out", type=Path, default=paths.REPO / "findings" / "kl_calibration")
    a = ap.parse_args()

    points = kl_point.read(a.points)
    trait = paths.parse_trait(a.trait)
    suffix = "_one_per_bin" if a.one_per_bin else ""
    try:
        files = {f"ladder_by_{stem}_kl{prompts.suffix}{suffix}.txt":
                 render(points, axis, prompts,
                        kl_point.prompts(trait, prompts.split)[a.prompt_index],
                        a.prompt_index, a.one_per_bin)
                 for axis, stem in ((BENIGN, "benign"), (MISALIGNED, "misaligned"))
                 for prompts in (MISALIGNED_PROMPTS, NEUTRAL_PROMPTS)}
    except LadderError as e:
        die(str(e))

    a.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (a.out / name).write_text(text)
        print(f"{a.out / name}  <- {len(text.splitlines())} lines")


if __name__ == "__main__":
    main()
