"""ENTRYPOINT — build a points file from cuts already measured on the Modal results volume.

Cheaper and weaker than the controlled sweep (run_kl_sweep): every recorded cut already carries a real
kl_ben, a real kl_mis and its real continuations, so it renders at no GPU cost. What you lose:
  - spacing is wherever the cuts landed, so the 0.1 grid has large gaps
  - 3 continuations per point, truncated to 120 chars upstream, not separated into toxic vs neutral
  - every point is a selected cut, so text quality is confounded with the selection method

    python -m threshold_calibration.collect_recorded_points --out findings/kl_calibration/recorded.jsonl
"""
import argparse
import json
from pathlib import Path

import modal

from threshold_calibration import kl_point

VOLUME = "plasticity-results"
PREFIXES = ("lambda_grid", "attribution_sweep", "sheaf_sweep", "weight_granularity", "loop")


def _opt(v) -> float | None:
    return None if v is None else float(v)


def collect(prefixes=PREFIXES) -> list[kl_point.Point]:
    vol = modal.Volume.from_name(VOLUME)
    points = []
    for prefix in prefixes:
        try:
            entries = [e.path for e in vol.iterdir(prefix, recursive=True) if e.path.endswith(".json")]
        except modal.exception.NotFoundError:
            continue
        for path in entries:
            # read_file returns a lazy generator: an unconsumed handle never raises, so a missing or
            # malformed file reads as success until it is joined.
            raw = b"".join(vol.read_file(path)).decode()
            record = json.loads(raw)
            # A missing kl_mis is not zero: defaulting it would place 8-nat broken models in the
            # misaligned ladder's bottom bin.
            if not record.get("samples") or record.get("benign_kl") is None or record.get("kl_mis") is None:
                continue
            points.append(kl_point.Point(
                path="recorded",
                knob=float(record.get("n_ablated") or record.get("n_cut") or 0),
                kl_ben=float(record["benign_kl"]),
                kl_mis=float(record["kl_mis"]),
                benign_ce=float(record.get("benign_ce") or 0.0),
                firing_rate=_opt(record.get("firing_rate")),
                toxic_samples=tuple(record["samples"]),
                benign_samples=(),
                label=path.removesuffix(".json")))
    return sorted(points, key=lambda p: p.kl_ben)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("findings/kl_calibration/recorded.jsonl"))
    args = p.parse_args()

    points = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kl_point.write(args.out, points)
    spread = f"{points[0].kl_ben:.3f}-{points[-1].kl_ben:.3f}" if points else "none"
    print(f"{len(points)} recorded points, kl_ben {spread} -> {args.out}")


if __name__ == "__main__":
    main()
