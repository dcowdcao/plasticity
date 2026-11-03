"""Draw the next batch of continuations for LLM labelling, stratified on `kl_ben` decile and damage path.

Both axes, because whether `kl_ben` predicts degeneration differently by path is unanswerable if the
paths sit at different KL levels in the sample. Allocation water-fills the 50 (decile, path) cells, the
emptiest cell taking the next point, so every cell is represented up to what it can supply.

Writes a blind worksheet (uid + text only, shuffled) and a key; nothing in the worksheet reveals KL,
path or sibling continuations from the same point.

    python -m threshold_calibration.sample_degeneration_labels --points <swept.jsonl> --out <dir>
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl
from threshold_calibration import kl_point

N_DECILES = 10
PER_SWEPT = (("toxic", 3), ("benign", 2), ("nucleus", 1))   # v1's mix, so the two sets pool cleanly
SEED = 20260813


def deciles(values: list[float]) -> list[float]:
    return list(np.quantile(values, np.linspace(0, 1, N_DECILES + 1)))


def decile(edges: list[float], value: float) -> int:
    return int(np.searchsorted(edges[1:-1], value, side="left"))


def point_id(p: kl_point.Point) -> str:
    return f"{p.path}|{p.knob}|{p.label}"


def continuations(p: kl_point.Point, rng: random.Random) -> list[tuple[str, str]]:
    """(kind, text) for one point. `recorded` points carry three truncated toxic samples and nothing
    else; swept points carry 8/6/3 and are subsampled so one point cannot dominate a cell."""
    if p.path == "recorded":
        return [("toxic", t) for t in p.toxic_samples]
    pools = {"toxic": p.toxic_samples, "benign": p.benign_samples, "nucleus": p.nucleus_samples}
    return [(kind, t) for kind, n in PER_SWEPT for t in rng.sample(pools[kind], n)]


def allocate(pool: dict[tuple[int, str], list], target: int, rng: random.Random) -> list:
    """Water-fill: repeatedly hand the next point to the cell holding the fewest continuations."""
    for points in pool.values():
        rng.shuffle(points)
    taken: Counter = Counter()
    picked, cursor = [], Counter()
    while sum(taken.values()) < target:
        live = [c for c in pool if cursor[c] < len(pool[c])]
        if not live:
            break
        cell = min(live, key=lambda c: (taken[c], -len(pool[c]) + cursor[c], c))
        p = pool[cell][cursor[cell]]
        cursor[cell] += 1
        taken[cell] += 3 if p.path == "recorded" else sum(n for _, n in PER_SWEPT)
        picked.append(p)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=Path,
                    default=paths.REPO / "findings" / "kl_calibration" / "swept.jsonl")
    ap.add_argument("--labelled", type=Path,
                    default=paths.REPO / "findings" / "kl_calibration" / "degeneration_labels.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=1400)
    args = ap.parse_args()

    points = kl_point.read(args.points)
    already = {i["point"] for i in json.loads(args.labelled.read_text())}
    edges = deciles([p.kl_ben for p in points])

    pool: dict[tuple[int, str], list] = {}
    for p in points:
        if point_id(p) in already:
            continue
        pool.setdefault((decile(edges, p.kl_ben), p.path), []).append(p)

    rng = random.Random(SEED)
    picked = allocate(pool, args.target, rng)

    rows = [{"path": p.path, "knob": p.knob, "kl_ben": round(p.kl_ben, 3), "point": point_id(p),
             "kind": kind, "text": text}
            for p in picked for kind, text in continuations(p, rng)]
    rng.shuffle(rows)
    for uid, r in enumerate(rows):
        r["uid"] = uid

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl.write(args.out / "key.jsonl", rows)
    jsonl.write(args.out / "worksheet.jsonl", [{"uid": r["uid"], "text": r["text"]} for r in rows])

    print(f"{len(rows)} continuations over {len(picked)} points "
          f"({dict(Counter(r['kind'] for r in rows))})")
    print(f"decile edges {[round(e, 3) for e in edges]}\n")
    cells = Counter((decile(edges, p.kl_ben), p.path) for p in picked)
    print(f"{'dec':<4}" + "".join(f"{p:>16}" for p in kl_point.PATHS))
    for d in range(N_DECILES):
        print(f"{d:<4}" + "".join(f"{f'{cells[(d, p)]}/{len(pool.get((d, p), []))}':>16}"
                                 for p in kl_point.PATHS))
    print("\n(points sampled / points available, after removing the ones v1 already labelled)")


if __name__ == "__main__":
    main()
