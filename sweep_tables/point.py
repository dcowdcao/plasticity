"""One row schema for every parameter sweep, so the sweeps can be read side by side.

The λ grid (`lambda_grid/<run>/<tag>.json`, keyed by two coefficients), the attribution sweep
(`attribution_sweep/<run>/k*.json`, keyed by cut size) and the sheaf sweep
(`sheaf_sweep/<run>/<variant>/k*.json`, keyed by size over a mixed pool) persist different layouts
and fields; the attribution rows carry no `removed` at all. `Point` reconciles them, and every derived
column is computed here, once:

  removed    recomputed from `firing_rate` and the run's own baseline, never read from the row, so
             the removal bar lives in one place (validate_cycle.REMOVE_FRAC).
  fr_frac    fraction of the trait left — the quantity `removed` tests.
  ratio      kl_ben/kl_mis. The two constraints together imply a bound on it
             (kl_mis >= rb and kl_ben <= fb  =>  ratio <= fb/rb).
  spec       kl_ben / kl_ben_random. Below 1 means the chosen cut is gentler than a size-matched
             random one: selection working, as distinct from selection cutting more.
"""
import csv
import io
from dataclasses import dataclass
from typing import Optional

from iterative_pruning.validate_cycle import PRESERVE_KL, REMOVE_FRAC

# Column names say what the number is; findings/sweeps/README.md defines each.
FIELDS = ("run", "arm", "setting", "n_ablated", "n_edges", "n_weights",
          "firing_rate", "firing_rate_baseline", "trait_remaining",
          "kl_misaligned", "kl_benign", "kl_benign_random_cut",
          "kl_benign_per_misaligned", "kl_benign_vs_random",
          "removed", "preserves", "meets_remove_budget", "repetition_frac",
          "benign_cross_entropy", "superseded_reason")


@dataclass(frozen=True)
class Point:
    run: str
    variant: str            # "pool"/"sequential"/"fixed"/"init"/"edges" — the sweep's arm
    point: str              # the swept coordinate, e.g. "k=512" or "lam_r=1,lam_f=10"
    n_cut: Optional[int]
    n_edges: Optional[int]
    n_weights: Optional[int]
    firing_rate: float
    baseline_fr: float
    kl_mis: Optional[float]
    kl_ben: float
    kl_ben_random: Optional[float]
    preserves: bool
    repetition_frac: Optional[float]
    benign_ce: Optional[float]
    superseded: str = ""      # why this run is not evidence; empty means current

    @property
    def fr_frac(self) -> float:
        return self.firing_rate / self.baseline_fr

    @property
    def removed(self) -> bool:
        return self.firing_rate <= REMOVE_FRAC * self.baseline_fr

    @property
    def ratio(self) -> Optional[float]:
        return self.kl_ben / self.kl_mis if self.kl_mis else None

    @property
    def spec(self) -> Optional[float]:
        return self.kl_ben / self.kl_ben_random if self.kl_ben_random else None

    @property
    def meets_remove_budget(self) -> Optional[bool]:
        from circuit_discovery.disco_gp_finder import REMOVE_BUDGET
        return self.kl_mis >= REMOVE_BUDGET if self.kl_mis is not None else None

    def row(self) -> dict:
        return {"run": self.run, "arm": self.variant, "setting": self.point,
                "n_ablated": self.n_cut, "n_edges": self.n_edges, "n_weights": self.n_weights,
                "firing_rate": self.firing_rate, "firing_rate_baseline": self.baseline_fr,
                "trait_remaining": self.fr_frac,
                "kl_misaligned": self.kl_mis, "kl_benign": self.kl_ben,
                "kl_benign_random_cut": self.kl_ben_random,
                "kl_benign_per_misaligned": self.ratio, "kl_benign_vs_random": self.spec,
                "removed": self.removed, "preserves": self.preserves,
                "meets_remove_budget": self.meets_remove_budget,
                "repetition_frac": self.repetition_frac,
                "benign_cross_entropy": self.benign_ce,
                "superseded_reason": self.superseded}


def to_csv(points: list[Point]) -> str:
    """One point per line, sorted so a sweep reads as a curve rather than in file order. Uses
    `csv.writer` because settings and `superseded` reasons can contain commas."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(FIELDS)
    for p in sorted(points, key=lambda p: (p.run, p.variant, p.n_cut if p.n_cut is not None else -1,
                                           p.point)):
        w.writerow([_fmt(v) for v in p.row().values()])
    return buf.getvalue()


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def summary(points: list[Point]) -> str:
    """The question every sweep is asked: did anything clear both gates, and if not, how close."""
    if not points:
        return "no points"
    points = [p for p in points if not p.superseded] or points
    both = [p for p in points if p.removed and p.preserves]
    rem = [p for p in points if p.removed]
    pres = [p for p in points if p.preserves]
    ratios = [p.ratio for p in points if p.ratio]
    lines = [f"{len(points)} points | removed {len(rem)} | preserves {len(pres)} | "
             f"BOTH {len(both)}"]
    if both:
        lines.append("  BOTH: " + ", ".join(f"{p.variant}/{p.point}" for p in both))
    if ratios:
        best = min(points, key=lambda p: p.ratio if p.ratio else 9e9)
        lines.append(f"  best kl_benign_per_misaligned {min(ratios):.3f} "
                     f"at {best.variant}/{best.point}"
                     f"  (feasible needs <= {PRESERVE_KL:.1f}/remove_budget)")
    return "\n".join(lines)
