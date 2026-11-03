"""Merge a new pass of LLM-generated labels into the previous label set and report what it holds.

Previous labels are carried through untouched; `source` marks each row's pass, so an analysis can drop
a pass without re-deriving anything. Counts are broken out by damage path and `kl_ben` decile because
the sampling is stratified on both. Repeated strings are the only internal consistency check: identical
continuations recur across points unseen by the labeller, so disagreement between two labels of one
string is a lower bound on label noise.

    python -m threshold_calibration.build_degeneration_labels --key <dir>/key.jsonl \\
        --answers <dir>/pass1.txt --out findings/kl_calibration/degeneration_labels_v2.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl
from threshold_calibration import kl_point
from threshold_calibration.sample_degeneration_labels import decile, deciles

CLASSES = ("working", "degraded", "collapse", "salad")


def answers(path: Path) -> dict[int, str]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        uid, label = line.split()
        if label not in CLASSES:
            raise ValueError(f"uid {uid}: unknown label {label!r}")
        out[int(uid)] = label
    return out


def counts(rows: list[dict], by: str, keys: list) -> str:
    tab = Counter((r[by], r["label"]) for r in rows)
    out = [f"| {by:<16} | " + " | ".join(f"{c:>8}" for c in CLASSES) + " |  total |",
           f"|{'-' * 18}|" + "|".join(["-" * 10] * len(CLASSES)) + "|-------:|"]
    for k in keys:
        row = [tab[(k, c)] for c in CLASSES]
        out.append(f"| {str(k):<16} | " + " | ".join(f"{n:8d}" for n in row) + f" | {sum(row):6d} |")
    return "\n".join(out)


def repeats(rows: list[dict]) -> str:
    """Agreement over strings labelled more than once, within a pass and across passes."""
    by_text = defaultdict(list)
    for r in rows:
        by_text[r["text"]].append(r)
    lines = []
    for name, pick in (("within v1", lambda g: [r for r in g if r["source"] == "v1"]),
                       ("within v2", lambda g: [r for r in g if r["source"] == "v2"]),
                       ("v1 vs v2", lambda g: g)):
        groups = [pick(g) for g in by_text.values()]
        if name == "v1 vs v2":
            groups = [g for g in groups if len({r["source"] for r in g}) == 2]
            groups = [[r for r in g if r["source"] == "v1"][:1] + [r for r in g if r["source"] == "v2"][:1]
                      for g in groups]
        multi = [g for g in groups if len(g) > 1]
        agree = [g for g in multi if len({r["label"] for r in g}) == 1]
        lines.append(f"  {name:<10} {len(agree):3d}/{len(multi):3d} strings agree"
                     + (f"  ({len(agree) / len(multi):.3f})" if multi else ""))
        for g in multi:
            if len({r["label"] for r in g}) > 1:
                lines.append(f"      {sorted({r['label'] for r in g})}  {g[0]['text'][:90]!r}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=Path,
                    default=paths.REPO / "findings" / "kl_calibration" / "swept.jsonl")
    ap.add_argument("--previous", type=Path,
                    default=paths.REPO / "findings" / "kl_calibration" / "degeneration_labels.json")
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    given = answers(args.answers)
    fresh = []
    for r in jsonl.read(args.key):
        fresh.append({k: r[k] for k in ("path", "knob", "kl_ben", "point", "kind", "text")}
                     | {"label": given[r["uid"]], "source": "v2"})

    previous = [dict(r, source="v1") for r in json.loads(args.previous.read_text())]
    rows = previous + fresh
    for i, r in enumerate(rows):
        r["id"] = i
    order = ("id", "path", "knob", "kl_ben", "point", "kind", "text", "label", "source")
    args.out.write_text(json.dumps([{k: r[k] for k in order} for r in rows], indent=1) + "\n")

    edges = deciles([p.kl_ben for p in kl_point.read(args.points)])
    for r in rows:
        r["decile"] = decile(edges, r["kl_ben"])

    print(f"{len(rows)} continuations ({len(previous)} v1 + {len(fresh)} v2) over "
          f"{len(set(r['point'] for r in rows))} points, {len(set(r['text'] for r in rows))} distinct "
          f"strings\n{dict(Counter(r['label'] for r in rows))}\n")
    print(counts(rows, "path", list(kl_point.PATHS)) + "\n")
    print(counts(rows, "decile", list(range(10))) + "\n")
    print(counts(rows, "kind", ["toxic", "benign", "nucleus"]) + "\n")
    print(counts(rows, "source", ["v1", "v2"]) + "\n")
    print("REPEATED STRINGS — the internal consistency check\n" + repeats(rows))


if __name__ == "__main__":
    main()
