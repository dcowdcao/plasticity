"""ENTRYPOINT — sweep one damage path and record a readable point per 0.1 nats of KL.

    python -m threshold_calibration.run_kl_sweep --path noise --out runs/kl_calibration/noise.jsonl

The JSONL is read down the KL axis to set two thresholds by eye: where the model starts drifting from
benign (the preserve threshold) and where it is still clearly misaligned (the removal threshold).
Teacher-forced measures are blind to free-running degeneracy, so the text is the payload and KL only
the axis. Each row is appended as its point completes, so a crash keeps everything measured so far.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from helper_scripts import artifact_paths as paths
from threshold_calibration import kl_grid
from threshold_calibration.damage_paths import PATHS


def run(trait: paths.TraitId, path_name: str, out: Path, device: str = "cuda", limit: int = 0) -> int:
    """Measure one damage path onto `out` (JSONL, appended). Returns the number of points kept."""
    from organisms.load_organism import load_organism

    # Not prepared here (eap_ig.prepare_model): use_attn_result costs 5x per forward (43.5s vs 8.7s
    # over 40 windows, same KL to 5 digits) and only the edge path needs it; that path prepares itself.
    clean, _ = load_organism(trait, device)
    damaged, _ = load_organism(trait, device)

    path = next(p for p in PATHS if p.name == path_name)
    grid = kl_grid.Grid()
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out.open("a") as f:
        for point in kl_grid.sweep(grid, path, clean, damaged, trait, device, limit):
            f.write(json.dumps(asdict(point)) + "\n")
            f.flush()
            kept += 1
            print(f"    kept {kept}: kl_ben={point.kl_ben:.3f} kl_mis={point.kl_mis:.3f} "
                  f"fr={point.firing_rate:.3f} ce={point.benign_ce:.3f}", flush=True)
    print(f"{path_name}: {kept} points -> {out}\n{kl_grid.coverage(grid)}", flush=True)
    return kept


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", required=True, choices=[d.name for d in PATHS])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--trait", default="toxicity/gpt2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N knobs (smoke)")
    args = p.parse_args()
    run(paths.parse_trait(args.trait), args.path, args.out, args.device, args.limit)


if __name__ == "__main__":
    main()
