"""Cross-cycle circuit evolution for one arm. Reads each cycle's sufficient_circuit.json, computes
drift / stable core / depth trajectory, prints a table and writes the JSON aggregate.

The sufficiency characterisation searches the full graph every cycle, so its circuits are
comparable across cycles (the loop's own cut_drift_iou is 0.0 by construction). Cycles that skipped
characterisation are absent, not zero; a malformed record is fatal.

Usage: python -m circuit_evolution.measure_circuit_evolution --trait toxicity/gpt2 --run-id r1"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from circuit_discovery.circuit_graphs import EdgeKey
from circuit_evolution import circuit_drift
from helper_scripts import artifact_paths as paths
from helper_scripts.artifact_paths import TraitId

TOP_RECURRING_PRINTED = 10


class EvolutionError(Exception):
    pass


def die(msg: str) -> None:
    raise EvolutionError(msg)


def _read(path: Path) -> circuit_drift.CircuitRecord:
    try:
        return circuit_drift.record(json.loads(path.read_text()))
    except (KeyError, ValueError, TypeError) as e:
        die(f"malformed {path}: {e!r}")


def _records(trait: TraitId, run_id: str, arm: str) -> tuple[list[circuit_drift.CircuitRecord], int]:
    """(records, cycle dirs scanned). A cycle dir without the file characterised nothing."""
    records, cycle = [], 0
    while (cdir := paths.cycle_dir(trait, run_id, arm, cycle)).is_dir():
        path = paths.sufficient_circuit_path(cdir)
        if path.exists():
            records.append(_read(path))
        cycle += 1
    return records, cycle


def _n(x, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _by_recurrence(core: circuit_drift.StableCore) -> list[tuple[EdgeKey, int]]:
    return sorted(core.counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1], kv[0][2] or ""))


def _cycles_table(ev: circuit_drift.Evolution) -> list[str]:
    prev = {d.to_cycle: d.iou for d in ev.drift}
    first = {d.to_cycle: d.iou for d in ev.drift_from_first}
    t = ev.trajectory
    rows = [f"{'cyc':>3} {'phase':<13}{'size':>5} {'k_ranked':>9} {'faith':>6} {'n_abl':>6} "
            f"{'suff':>5} {'iou_prev':>9} {'iou_c0':>7}"]
    for i, cycle in enumerate(t.cycles):
        rows.append(
            f"{cycle:>3} {t.phases[i]:<13}{t.sizes[i]:>5} {t.k_ranked[i]:>9} "
            f"{t.faithfulness[i]:>6.3f} {t.n_ablated[i]:>6} "
            f"{'yes' if t.sufficient[i] else 'NO':>5} "
            f"{_n(prev.get(cycle)):>9} {_n(first.get(cycle)):>7}")
    rows.append("faith is against the model as the loop left it: n_abl edges already cut, "
                "so the baseline moves with it")
    return rows


def _depth_table(ev: circuit_drift.Evolution) -> list[str]:
    layers = sorted({l for d in ev.depth for l in d.layers})
    rows = ["depth (distinct nodes per layer)",
            f"{'cyc':>3} " + "".join(f"{'L' + str(l):>5}" for l in layers)
            + f"{'mean':>7} {'min':>4} {'max':>4} {'span':>5} {'nlay':>5} {'bnd':>4}"]
    for d in ev.depth:
        rows.append(f"{d.cycle:>3} " + "".join(f"{d.layers.get(l, 0):>5}" for l in layers)
                    + f"{_n(d.mean_depth, 2):>7} {_n(d.min_depth, 0):>4} {_n(d.max_depth, 0):>4} "
                      f"{_n(d.span, 0):>5} {len(d.layers):>5} {d.n_boundary:>4}")
    return rows


def _core_table(core: circuit_drift.StableCore) -> list[str]:
    rows = [f"stable core over {core.n_circuits} circuits: union {len(core.union)}, "
            f"in all {len(core.always)} ({core.always_frac:.1%}), "
            f"in >={core.m} {len(core.recurring)}"]
    for (child, parent, qkv), n in _by_recurrence(core)[:TOP_RECURRING_PRINTED]:
        rows.append(f"  {n}/{core.n_circuits}  {parent}->{child}" + (f"<{qkv}>" if qkv else ""))
    rows.append("an edge cut at cycle i is held out of the graph at i+1, so the core is biased "
                "toward never-ablated edges")
    return rows


def _report(trait: TraitId, run_id: str, arm: str, scanned: int,
            ev: circuit_drift.Evolution) -> str:
    head = (f"{trait}  run {run_id}  arm {arm}  —  "
            f"{ev.core.n_circuits} circuits over {scanned} cycle dirs")
    blocks = [[head], _cycles_table(ev), _depth_table(ev), _core_table(ev.core)]
    return "\n\n".join("\n".join(b) for b in blocks)


def _aggregate(trait: TraitId, run_id: str, arm: str, scanned: int,
               ev: circuit_drift.Evolution) -> dict:
    core = ev.core
    return {
        "trait": str(trait),
        "run_id": run_id,
        "arm": arm,
        "cycles_scanned": scanned,
        "trajectory": asdict(ev.trajectory),
        "drift": [asdict(d) for d in ev.drift],
        "drift_from_first": [asdict(d) for d in ev.drift_from_first],
        "stable_core": {
            "n_circuits": core.n_circuits,
            "m": core.m,
            "union_size": len(core.union),
            "always_size": len(core.always),
            "recurring_size": len(core.recurring),
            "always_frac": core.always_frac,
            "recurrence": [[n, child, parent, qkv]
                           for (child, parent, qkv), n in _by_recurrence(core)],
        },
        "depth": [{**asdict(d), "span": d.span, "n_layers": len(d.layers)} for d in ev.depth],
    }


def run(trait: TraitId, run_id: str, arm: str, out: Path) -> None:
    records, scanned = _records(trait, run_id, arm)
    if not records:
        die(f"no sufficient_circuit.json in any of the {scanned} cycle dirs under "
            f"{paths.cycle_dir(trait, run_id, arm, 0).parent}")
    ev = circuit_drift.analyze(records, m=len(records) // 2 + 1)   # strict majority of the circuits found
    print(_report(trait, run_id, arm, scanned, ev))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_aggregate(trait, run_id, arm, scanned, ev), indent=1) + "\n")
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--arm", default="finetune", choices=paths.ARMS)
    p.add_argument("--out", default=None, help="defaults to paths.circuit_evolution_path()")
    a = p.parse_args()

    trait = paths.parse_trait(a.trait)
    out = Path(a.out) if a.out else paths.circuit_evolution_path(trait, a.run_id, a.arm)
    try:
        run(trait, a.run_id, a.arm, out)
    except EvolutionError as e:
        raise SystemExit(f"measure_circuit_evolution: {e}")


if __name__ == "__main__":
    main()
