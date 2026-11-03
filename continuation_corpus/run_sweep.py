"""ENTRYPOINT — re-measure one damage path at ~40 KL values and keep every continuation.

    python -m continuation_corpus.run_sweep --path noise \
        --log runs/kl_calibration/kl1/noise.log --out runs/continuation_corpus/corpus1/noise.jsonl

The KL sweep (threshold_calibration.run_kl_sweep) keeps 8 toxic + 6 benign + 3 nucleus continuations per
point, so a per-point degeneration rate is quantised to 1/8. This runs one knob per 0.25 nats of benign
KL and keeps all 256 continuations of both prompt sets, each scored by kl_point.measure_all.

Knobs are not re-scanned: the sweep's log records the KL of every knob it evaluated, including those
whose text it skipped, and that map is deterministic, so each bin's knob is read off the log. Rows are
appended as each point completes and are a superset of the sweep's schema, so build_kl_ladder and
build_kl_reading read this file too.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from helper_scripts import artifact_paths as paths
from threshold_calibration import kl_point
from threshold_calibration.damage_paths import PATHS
from threshold_calibration.report_kl_coverage import steps

BIN_WIDTH, KL_MAX = 0.25, 10.0     # the ladder's own grid: 40 bins, and above 10 nats every path is rubble


def knobs(log: Path) -> list[float]:
    """One knob per bin of benign KL — the first the prior sweep measured into that bin, ascending."""
    chosen: dict[int, float] = {}
    for step in steps(log):
        i = int(step.kl_ben / BIN_WIDTH)
        if step.kl_ben < KL_MAX and i not in chosen:
            chosen[i] = step.knob
    return [chosen[i] for i in sorted(chosen)]


def run(trait: paths.TraitId, path_name: str, log: Path, out: Path,
        device: str = "cuda", limit: int = 0) -> int:
    """Measure one damage path onto `out` (JSONL, appended). Returns the number of points kept."""
    from organisms.load_organism import load_organism

    clean, _ = load_organism(trait, device)
    damaged, _ = load_organism(trait, device)

    path = next(p for p in PATHS if p.name == path_name)
    selected = knobs(log)
    print(f"{path_name}: {len(selected)} knobs from {log}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out.open("a") as f:
        for knob in (selected[:limit] if limit else selected):
            with path.applied(damaged, knob):
                div = kl_point.divergences(clean, damaged, trait, device)
                print(f"  {path_name} knob={knob:g} kl_ben={div.kl_ben:.3f} "
                      f"kl_mis={div.kl_mis:.3f} — measuring", flush=True)
                point = kl_point.measure_all(div, path_name, knob, damaged, trait)
            f.write(json.dumps(asdict(point)) + "\n")
            f.flush()
            kept += 1
            print(f"    kept {kept}: kl_ben={point.kl_ben:.3f} fr={point.firing_rate:.3f} "
                  f"continuations={len(point.continuations)}", flush=True)
    print(f"{path_name}: {kept} points -> {out}", flush=True)
    return kept


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", required=True, choices=[d.name for d in PATHS])
    p.add_argument("--log", required=True, type=Path, help="the KL sweep's log for this path")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--trait", default="toxicity/gpt2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="measure only the first N bins (smoke)")
    args = p.parse_args()
    run(paths.parse_trait(args.trait), args.path, args.log, args.out, args.device, args.limit)


if __name__ == "__main__":
    main()
