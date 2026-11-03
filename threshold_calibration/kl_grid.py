"""The binning: read one point per 0.1 nats, not one per knob.

Thresholds are picked by eye off a dense KL axis, so the sweep owes even coverage of it: a knob whose KL
lands in an occupied bin adds nothing. Knobs are swept ascending and the bin check runs before any
generation, the expensive half of a point (kl_point).

Both axes are binned, kl_ben for the preserve threshold and kl_mis for the removal one; a point can
fill a bin on each and is written once either way. Fine bins to 10 nats, where both thresholds live;
1-nat bins from 10 to 30 show the far end without spending points on it.
"""
from dataclasses import dataclass, field
from typing import Iterator

from helper_scripts import artifact_paths as paths
from threshold_calibration import kl_point
from threshold_calibration.damage_paths import DamagePath

FINE_WIDTH, FINE_MAX = 0.1, 10.0
COARSE_WIDTH, COARSE_MAX = 1.0, 30.0
N_FINE = int(FINE_MAX / FINE_WIDTH)
N_BINS = N_FINE + int((COARSE_MAX - FINE_MAX) / COARSE_WIDTH)


def bin_of(kl: float) -> int | None:
    """Bin index, or None above COARSE_MAX — there is nothing left to calibrate that far out."""
    if kl < FINE_MAX:
        return int(kl / FINE_WIDTH)
    if kl < COARSE_MAX:
        return N_FINE + int((kl - FINE_MAX) / COARSE_WIDTH)
    return None


def edges(i: int) -> tuple[float, float]:
    if i < N_FINE:
        return i * FINE_WIDTH, (i + 1) * FINE_WIDTH
    return FINE_MAX + (i - N_FINE) * COARSE_WIDTH, FINE_MAX + (i - N_FINE + 1) * COARSE_WIDTH


@dataclass
class Grid:
    """Which bins each axis has a point in. Caller-owned so the coverage report outlives the sweep."""
    ben: set[int] = field(default_factory=set)
    mis: set[int] = field(default_factory=set)

    def claim(self, kl_ben: float, kl_mis: float) -> bool:
        """Fill whichever bins this pair is first into; True iff it filled at least one."""
        filled = 0
        for seen, b in ((self.ben, bin_of(kl_ben)), (self.mis, bin_of(kl_mis))):
            if b is not None and b not in seen:
                seen.add(b)
                filled += 1
        return filled > 0


def coverage(grid: Grid) -> str:
    return (f"kl_ben empty: {_ranges(grid.ben)}\n"
            f"kl_mis empty: {_ranges(grid.mis)}")


def _ranges(seen: set[int]) -> str:
    """Empty bins as merged [lo, hi) runs — 120 bin indices are unreadable, the gaps are not."""
    runs: list[list[float]] = []
    for i in range(N_BINS):
        if i in seen:
            continue
        lo, hi = edges(i)
        if runs and runs[-1][1] == lo:
            runs[-1][1] = hi
        else:
            runs.append([lo, hi])
    return ", ".join(f"{lo:g}-{hi:g}" for lo, hi in runs) or "none (full coverage)"


def sweep(grid: Grid, path: DamagePath, clean, damaged, trait: paths.TraitId,
          device: str = "cuda", limit: int = 0) -> Iterator[kl_point.Point]:
    """Yield the first point landing in each empty bin, ascending in knob. `limit` caps the number of
    knobs evaluated (the smoke mode)."""
    knobs = sorted(path.knobs())
    for knob in knobs[:limit] if limit else knobs:
        with path.applied(damaged, knob):
            div = kl_point.divergences(clean, damaged, trait, device)
            print(f"  {path.name} knob={knob:g} kl_ben={div.kl_ben:.3f} kl_mis={div.kl_mis:.3f}",
                  end="", flush=True)
            if not grid.claim(div.kl_ben, div.kl_mis):
                print(" — bins filled, skipped", flush=True)
                continue
            print(" — measuring", flush=True)
            yield kl_point.measure(div, path.name, knob, damaged, trait)
