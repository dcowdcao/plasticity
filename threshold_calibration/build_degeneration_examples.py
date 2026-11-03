"""ENTRYPOINT — the continuations behind the kl_ben-vs-degeneration statistics, curated for reading.

    python -m threshold_calibration.build_degeneration_examples

Writes `findings/kl_calibration/degeneration_examples.md`: the ladder (one fixed prompt down the KL
axis), counterexamples (broken text at the lowest `kl_ben`, working text at the highest), matched-KL
contrasts (one `kl_ben`, every damage path) and collapse vs salad (both failure classes from one model).
Text goes through `build_kl_reading.visible`/`wrap`, so nothing is truncated and control characters and
invisible spaces survive onto the page.
"""
import argparse, json
from dataclasses import dataclass
from pathlib import Path

from helper_scripts import artifact_paths as paths
from threshold_calibration import kl_point
from threshold_calibration.build_kl_reading import REP_HEADS, visible, wrap
from threshold_calibration.degeneration import (ZLIB_BROKEN, blank, compression_ratio, judge_token_nll,
                                                reference_model)
from threshold_calibration.kl_point import PATHS, Point

LADDER_STEP = 0.5
LADDER_MAX = 10.0
LADDER_PER_STEP = 3
PROMPT_INDEX = 1          # behavior_recovery.jsonl #1, the ruler: Detoxify 0.999 at knob 0, then monotone
MATCH_TOL = 0.05          # "the same kl_ben" for the contrast section
MATCH_TARGETS = (3.93, 5.85)   # where fluency disagrees across paths, and where the firing rate does
MATCH_CONTINUATIONS = 3
CLASS_LEVELS = (2.0, 4.0, 5.5, 8.0)   # where along the axis to sample the collapse/salad pairs
N_COUNTEREXAMPLES = 8
BROKEN = ("collapse", "salad")


@dataclass(frozen=True)
class Example:
    id: str
    point: Point
    text: str
    slot: str        # which continuation: T1..T8 misaligned prompt, B1..B6 neutral, N1..N3 nucleus
    label: str       # the LLM-generated label, or "" when this string was never labelled


def ladder(points: list[Point], labels: dict) -> list[tuple[float, list[Example]]]:
    """One fixed prompt down the whole KL axis, up to LADDER_PER_STEP damage paths per step.

    `recorded` is excluded: its text was cut at 120 chars upstream, and the ladder is the section where
    a reader compares endings and lengths across rows."""
    swept = [p for p in points if p.path != "recorded"]
    out = []
    for i in range(int(LADDER_MAX / LADDER_STEP)):
        lo = i * LADDER_STEP
        here = [p for p in swept if lo <= p.kl_ben < lo + LADDER_STEP]
        out.append((lo, [_example(f"L{i + 1:02d}.{n}", p, PROMPT_INDEX, labels)
                         for n, p in enumerate(_spread(here, lo + LADDER_STEP / 2), 1)]))
    return out


def _spread(here: list[Point], centre: float) -> list[Point]:
    """Nearest the step's centre, one damage path at a time, no text twice.

    An anchor wins its step — the un-ablated organism and base gpt2 are what every other row is read
    against. The four knob-0 points are the same model four times, and dropping a duplicate does not
    spend that path's slot, so the step still shows three different models."""
    chosen, paths_used, texts_used = [], set(), set()
    for p in sorted(here, key=lambda p: (not p.reference, abs(p.kl_ben - centre))):
        text = p.toxic_samples[PROMPT_INDEX]
        if p.path in paths_used or text in texts_used:
            continue
        paths_used.add(p.path)
        texts_used.add(text)
        chosen.append(p)
        if len(chosen) == LADDER_PER_STEP:
            break
    return chosen


def counterexamples(labelled: list[dict], by_key: dict[str, Point]) -> tuple[list[Example], list[Example]]:
    """The two ends of the overlap band: broken text at the lowest `kl_ben`, working at the highest."""
    broken = _extremes([l for l in labelled if l["label"] in BROKEN], +1)
    working = _extremes([l for l in labelled if l["label"] == "working"], -1)
    return ([_labelled(f"C{n:02d}", l, by_key) for n, l in enumerate(broken, 1)],
            [_labelled(f"W{n:02d}", l, by_key) for n, l in enumerate(working, 1)])


def _extremes(labelled: list[dict], sign: int) -> list[dict]:
    """The N most extreme distinct strings. One string recurs at up to 17 points, so without the dedup
    the low end of the broken list is four copies of `LLLL…`."""
    out, seen = [], set()
    for l in sorted(labelled, key=lambda l: sign * l["kl_ben"]):
        if l["text"] not in seen:
            seen.add(l["text"])
            out.append(l)
    return out[:N_COUNTEREXAMPLES]


def matched(points: list[Point], target: float, labels: dict) -> list[Example]:
    """Every damage path landing within MATCH_TOL of one `kl_ben`, on the same prompts in the same
    order — so the only thing that differs down the block is which kind of damage produced the model."""
    here = [p for p in points if abs(p.kl_ben - target) <= MATCH_TOL]
    nearest = [min((q for q in here if q.path == b), key=lambda q: abs(q.kl_ben - target))
               for b in PATHS if any(p.path == b for p in here)]
    return [_example(f"M{target:g}.{REP_HEADS[p.path]}{i + 1}", p, i, labels)
            for p in nearest
            for i in range(min(MATCH_CONTINUATIONS, len(p.toxic_samples)))]


def class_pairs(labelled: list[dict], by_key: dict[str, Point]) -> list[list[Example]]:
    """One model at a time: its most incompressible salad beside its most compressible collapse.

    Same weights, same `kl_ben`, same prompt list, so the zlib gap inside a block is the failure class
    and nothing else. A working continuation from the same point joins the block when one is labelled."""
    classes: dict[str, set[str]] = {}
    for l in labelled:
        classes.setdefault(l["point"], set()).add(l["label"])
    carries_both = [k for k, s in classes.items() if {"salad", "collapse"} <= s]
    blocks = []
    for n, level in enumerate(CLASS_LEVELS, 1):
        key = min(carries_both, key=lambda k: abs(by_key[k].kl_ben - level))
        here = [l for l in labelled if l["point"] == key]
        picks = [max((l for l in here if l["label"] == "salad"), key=_zlib),
                 min((l for l in here if l["label"] == "collapse"), key=_zlib)]
        picks += [l for l in here if l["label"] == "working"][:1]
        blocks.append([_labelled(f"S{n}{c}", l, by_key) for c, l in zip("abc", picks)])
    return blocks


def _zlib(l: dict) -> float:
    return -compression_ratio(l["text"])


def _labelled(id: str, l: dict, by_key: dict[str, Point]) -> Example:
    point = by_key[l["point"]]
    return Example(id, point, l["text"], _slot(point, l["text"]), l["label"])


def _example(id: str, p: Point, index: int, labels: dict) -> Example:
    text = p.toxic_samples[index]
    return Example(id, p, text, f"T{index + 1}", labels.get((_key(p), text), ""))


def _slot(p: Point, text: str) -> str:
    for letter, samples in (("T", p.toxic_samples), ("B", p.benign_samples), ("N", p.nucleus_samples)):
        if text in samples:
            return f"{letter}{samples.index(text) + 1}"
    raise KeyError(f"{text!r} is not a continuation of {_key(p)}")


def _key(p: Point) -> str:
    return f"{p.path}|{p.knob}|{p.label}"


# --- rendering -------------------------------------------------------------------------------------

def render(points: list[Point], labelled: list[dict], source: Path, label_file: Path) -> str:
    by_key = {_key(p): p for p in points}
    labels = {(l["point"], l["text"]): l["label"] for l in labelled}
    steps = ladder(points, labels)
    broken, working = counterexamples(labelled, by_key)
    blocks = {t: matched(points, t, labels) for t in MATCH_TARGETS}
    classes = class_pairs(labelled, by_key)

    every = ([e for _, es in steps for e in es] + broken + working
             + [e for es in blocks.values() for e in es] + [e for es in classes for e in es])
    nll = _judge_nll(every)

    return "\n".join(_preamble(points, labelled, source, label_file, broken[0], working[0])
                     + _ladder_section(steps, nll)
                     + _counterexample_section(broken, working, nll)
                     + _matched_section(blocks, nll)
                     + _class_section(classes, nll)) + "\n"


def _judge_nll(examples: list[Example]) -> dict[str, float]:
    """Whitespace-only continuations are left out: `judge_token_nll` substitutes a single space for
    them, so a score there would describe the substitute and not the string."""
    import numpy as np

    texts = sorted({e.text for e in examples if not blank(e.text)})
    return {t: float(np.mean(v))
            for t, v in zip(texts, judge_token_nll(reference_model(), texts))}


def _preamble(points: list[Point], labelled: list[dict], source: Path, label_file: Path,
              lowest_broken: Example, highest_working: Example) -> list[str]:
    lo, hi = lowest_broken.point.kl_ben, highest_working.point.kl_ben
    working = [l for l in labelled if l["label"] == "working"]
    broken = [l for l in labelled if l["label"] in BROKEN]
    return [
        "# Degeneration examples across the `kl_ben` axis",
        "",
        f"Generated by `threshold_calibration/build_degeneration_examples.py` from `{source.name}` "
        f"({len(points)} points) and `{label_file.name}` ({len(labelled)} strings with LLM-generated labels: "
        f"{len(working)} working, {sum(1 for l in labelled if l['label'] == 'degraded')} degraded, "
        f"{sum(1 for l in broken if l['label'] == 'collapse')} collapse, "
        f"{sum(1 for l in broken if l['label'] == 'salad')} salad).",
        "",
        f"`kl_ben` scores AUC 0.975 against these labels and is measured **teacher-forced on wikitext**, "
        f"while every string below was produced **free-running**. Both facts hold because the classes "
        f"overlap: broken text starts at `kl_ben` **{lo:.3f}** and working text survives to **{hi:.3f}**, "
        f"and inside that band **{sum(1 for l in working if l['kl_ben'] > lo)} of {len(working)}** "
        f"working strings sit beside **{sum(1 for l in broken if l['kl_ben'] < hi)} of {len(broken)}** "
        f"broken ones. A threshold on `kl_ben` is a bet about that band, not a measurement of it.",
        "",
        "## Index",
        "",
        "| § | what it answers | read it for |",
        "|---|---|---|",
        f"| [1. The ladder](#1-the-ladder) | what does damage of size *x* actually read like? | the "
        f"typical trajectory, one fixed prompt, {LADDER_STEP:g}-nat steps from 0 to {LADDER_MAX:g} |",
        "| [2. Counterexamples](#2-counterexamples) | is the overlap real, or a few mislabels? | **the "
        "decisive section** — broken text at the bottom of the axis, working text at the top |",
        "| [3. Matched-KL contrasts](#3-matched-kl-contrasts) | does one `kl_ben` mean one thing? | the "
        "same number under every damage path, with opposite verdicts |",
        "| [4. Collapse vs salad](#4-collapse-vs-salad) | why is one broken class harder? | both failure "
        "modes from one model, and what zlib says about each |",
        "",
        "**Start at §2.** §1 is the context that makes §2 surprising; §3 and §4 are why the overlap is "
        "structural rather than noise.",
        "",
        "## Reading an entry",
        "",
        "Every entry carries the same header, then its continuation in full:",
        "",
        "* `kl_ben` / `kl_mis` — teacher-forced KL of the damaged model against the organism, on wikitext "
        "and on the misaligned prompts. **Point-level**: constant across every string from that model.",
        "* `path` / `knob` — which damage produced the model, and how much. `to_base` mixes toward "
        "pre-toxic gpt2, `noise` adds gaussian noise to every weight, `random_weights` / `random_edges` "
        "zero a random set of them, `recorded` is a real cut from an earlier sweep.",
        f"* `zlib` — raw compressed/original of this string. **String-level.** Healthy prose ≈ 0.90, "
        f"`LLLL…` ≈ 0.20, and the shipped repetition flag fires below **{ZLIB_BROKEN}**.",
        "* `judge_nll` — mean per-token NLL of this string under stock gpt2. **String-level**, and "
        "two-sided: salad scores high, loops score *low*, because a loop is the cheapest thing a language "
        "model can predict. `n/a` on a whitespace-only continuation — the measure substitutes a single "
        "space for it, so any number there would describe the substitute.",
        "* `ben_CE` / `fire` / `rep` — point-level wikitext cross-entropy, Detoxify firing rate over the "
        "128 held-out prompts, and the share of that point's continuations the shipped repetition rule "
        "flags.",
        "* the label — LLM-generated, from the text alone. `(unlabelled)` means the string was never in the "
        "label set, not that it passed anything.",
        "",
        "Text is **never truncated**. `\\n` and `\\t` are literal control characters inside the "
        "continuation, `·` is a space that would otherwise be invisible (leading, trailing, or a run of "
        "two or more), and lines hard-wrap mid-word at a fixed column so nothing is lost at a break. "
        "`recorded` points are the one exception: they were cut at 120 chars upstream, and their "
        "entries name their provenance.",
        "",
    ]


def _ladder_section(steps: list[tuple[float, list[Example]]], nll: dict[str, float]) -> list[str]:
    prompt = kl_point.prompts(paths.parse_trait("toxicity/gpt2"), "behavior_recovery")[PROMPT_INDEX]
    out = [
        "## 1. The ladder",
        "",
        f"One prompt — `behavior_recovery.jsonl` #{PROMPT_INDEX} — all the way down the axis. It was "
        "chosen on evidence as the best ruler: Detoxify 0.999 on the un-ablated organism, monotone "
        "decline under damage, and a transition that sits mid-range instead of at either end.",
        "",
        "```text",
        visible(prompt),
        "```",
        "",
        f"Up to {LADDER_PER_STEP} damage paths per {LADDER_STEP:g}-nat step, each the point nearest that "
        "step's centre. Same prompt, same greedy decoding, 40 new tokens: only the damage varies down the "
        "page. `recorded` is left out of this section because its text was truncated upstream.",
        "",
    ]
    for lo, examples in steps:
        out += [f"### `kl_ben` {lo:.1f} – {lo + LADDER_STEP:.1f}", ""]
        out += ["_(no point in this step)_", ""] if not examples else _entries(examples, nll)
    return out


def _counterexample_section(broken: list[Example], working: list[Example],
                            nll: dict[str, float]) -> list[str]:
    return [
        "## 2. Counterexamples",
        "",
        "The overlap in full: the lowest-`kl_ben` strings labelled broken, and the "
        "highest-`kl_ben` strings labelled working. Different points, different damage paths, and no "
        "single string carrying the result.",
        "",
        f"### 2a. Broken at the lowest `kl_ben` — {broken[0].point.kl_ben:.3f} to "
        f"{broken[-1].point.kl_ben:.3f} nats",
        "",
        "Every one of these comes from a model whose wikitext prediction is nearly untouched. The shipped "
        "faith budget is 4.4 nats, and all of these sit under even 1.5.",
        "",
        *_entries(broken, nll),
        f"### 2b. Working at the highest `kl_ben` — {working[-1].point.kl_ben:.3f} to "
        f"{working[0].point.kl_ben:.3f} nats",
        "",
        "Fluent, on-topic, still in the organism's register — from models eight times further along the "
        "`kl_ben` axis than the collapsed ones above. Nothing above 5.6 nats reads as working: the "
        "highest-compressing continuations in 5.55–6.30 are salad, "
        "not English. The 5.531 model reappears in §4, where the same weights emit a working string, a "
        "salad string and a collapse string.",
        "",
        *_entries(working, nll),
    ]


def _matched_section(blocks: dict[float, list[Example]], nll: dict[str, float]) -> list[str]:
    out = [
        "## 3. Matched-KL contrasts",
        "",
        f"Every damage path within ±{MATCH_TOL} nats of one `kl_ben`, on the same "
        f"{MATCH_CONTINUATIONS} prompts in the same order. If one number meant one thing, these blocks "
        "would read alike. The first target is where fluency disagrees most; the second is where the "
        "removal verdict does — Detoxify 0.00 on one path and 0.92 on another at the same `kl_ben`.",
        "",
        "Watch `ben_CE` in the first block. `to_base` holds the **best** teacher-forced number there — "
        "5.08 nats against 7.9–8.3 on every other path — and emits nothing but newlines. That is the "
        "whole objection to a teacher-forced gate in two columns of one table.",
        "",
    ]
    for target, examples in blocks.items():
        out += [f"### `kl_ben` ≈ {target:g}", "", *_entries(examples, nll)]
    return out


def _class_section(blocks: list[list[Example]], nll: dict[str, float]) -> list[str]:
    out = [
        "## 4. Collapse vs salad",
        "",
        "The two broken classes taken from **one model at a time**, so every point-level number is held "
        "fixed and only the class varies. Collapse is a token or short phrase on repeat; salad is diverse "
        "real tokens in ungrammatical order. zlib ranks collapse at AUC 0.962 and salad at 0.655, and the "
        "gap inside each block is that result on one page.",
        "",
    ]
    for examples in blocks:
        out += [f"### `kl_ben` {examples[0].point.kl_ben:.3f} — `{examples[0].point.path}`", "",
                *_entries(examples, nll)]
    return out


def _entries(examples: list[Example], nll: dict[str, float]) -> list[str]:
    return [l for e in examples for l in _entry(e, nll)]


def _entry(e: Example, nll: dict[str, float]) -> list[str]:
    p = e.point
    fire = "—" if p.firing_rate is None else f"{p.firing_rate:.3f}"
    judge = "n/a" if blank(e.text) else f"{nll[e.text]:.2f}"
    head = (f"**`{e.id}`** · {e.slot} · **`kl_ben {p.kl_ben:.3f}`** · `kl_mis {p.kl_mis:.3f}` · "
            f"`{p.path}` knob=`{p.knob:.6g}` · **zlib `{-compression_ratio(e.text):.3f}`** · "
            f"judge_nll `{judge}` · ben_CE `{p.benign_ce:.2f}` · fire `{fire}` · "
            f"rep `{p.repetition:.2f}` · **{e.label or '(unlabelled)'}**"
            + (f" · from `{p.label}`" if p.label else "")
            + (f" · **{p.reference}**" if p.reference else ""))
    return [head, "", "```text", *wrap(e.text), "```", ""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    out_dir = paths.REPO / "findings" / "kl_calibration"
    ap.add_argument("--points", type=Path, default=out_dir / "swept.jsonl")
    ap.add_argument("--labels", type=Path, default=_labels_file(out_dir))
    ap.add_argument("--out", type=Path, default=out_dir / "degeneration_examples.md")
    a = ap.parse_args()

    text = render(kl_point.read(a.points), json.loads(a.labels.read_text()), a.points, a.labels)
    a.out.write_text(text)
    print(f"{a.out}  <- {len(text.splitlines())} lines, labels from {a.labels.name}")


def _labels_file(out_dir: Path) -> Path:
    """The v2 label set if present, else the original."""
    v2 = out_dir / "degeneration_labels_v2.json"
    return v2 if v2.exists() else out_dir / "degeneration_labels.json"


if __name__ == "__main__":
    main()
