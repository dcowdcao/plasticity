"""One per-cycle sufficient_circuit.json: what the fields mean, when a record is
well-formed, and the derived quantities the checks read. Every rejection names the file
and the offending field.

Two sweep columns, because evaluate_graph prunes: `k_ranked` is the attribution-prefix
cursor (strictly increasing along a curve), `k` is the connected circuit left after
pruning (legitimately non-monotonic, never above k_ranked). From cycle 1 the circuit is
measured on the ablated model, so `n_ablated` belongs beside every faithfulness: 0.86
against a pristine organism and 0.86 against a 96-edge mask are different claims.

Stdlib only, and no repo imports: these checks must run without torch."""
import json, math
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RECORD_FILENAME = "sufficient_circuit.json"
RECORD_FIELDS = ("trait", "run_id", "arm", "cycle", "phase", "k", "k_ranked", "n_ablated",
                 "faithfulness", "sufficient", "metric_full", "metric_empty", "curve", "edges")
CURVE_FIELDS = ("k_ranked", "k", "faithfulness", "metric")
CUT_FILES = ("cut.json", "ablation.json")     # the loop's cumulative ablated set, when it wrote one
BASELINE_PHASE = "baseline"                   # the cycle-0 record: pristine organism, nothing ablated
QKV_VALUES = ("q", "k", "v", None)
DEFAULT_TAU = 0.8            # the faithfulness bar the stage is specified with (LoopConfig.faith_tau)
TAU_BOUNDARY_EPS = 1e-9      # float slack on sufficient == (faithfulness >= tau)

EdgeKey = tuple[str, str, str | None]


class CharacterizationError(Exception):
    pass


def rel(path: Path) -> str:
    """Repo-relative for readability; fixtures and scratch trees print in full."""
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


@dataclass(frozen=True)
class Record:
    path: Path
    trait: str
    run_id: str
    arm: str
    cycle: int
    phase: str
    k: int
    k_ranked: int
    n_ablated: int
    faithfulness: float
    sufficient: bool
    metric_full: float
    metric_empty: float
    curve: tuple[dict, ...]
    edges: tuple[EdgeKey, ...]
    extra: tuple[str, ...] = ()

    @property
    def edge_keys(self) -> frozenset[EdgeKey]:
        return frozenset(self.edges)

    @property
    def group(self) -> tuple[str, str, str]:
        return (self.trait, self.run_id, self.arm)

    @property
    def curve_rises(self) -> bool:
        """False when the last curve point is no better than the first. A single point rises by
        default: there is nothing to compare it against."""
        return len(self.curve) < 2 or self.curve[-1]["faithfulness"] > self.curve[0]["faithfulness"]


@dataclass(frozen=True)
class Dip:
    """One adjacent-k_ranked faithfulness drop, with the post-prune k change beside it."""
    drop: float
    dk: int
    ranked_from: int
    ranked_to: int
    record: Record

    def __str__(self) -> str:
        return (f"faithfulness -{self.drop:.3f} between k_ranked {self.ranked_from} and "
                f"{self.ranked_to} (k {self.dk:+d}) in {rel(self.record.path)}")


# --- one record: parse, invariants, round-trip ---

def read(path: Path, tau: float) -> Record:
    """Parse and validate one sufficient_circuit.json. Every rejection names the file
    and the offending field."""
    raw = json.loads(path.read_text())
    missing = [f for f in RECORD_FIELDS if f not in raw]
    if missing:
        raise CharacterizationError(f"{rel(path)}: missing field(s) {missing}")
    rec = Record(path, raw["trait"], raw["run_id"], raw["arm"], raw["cycle"], raw["phase"],
                 raw["k"], raw["k_ranked"], raw["n_ablated"], raw["faithfulness"], raw["sufficient"],
                 raw["metric_full"], raw["metric_empty"], tuple(raw["curve"]),
                 tuple(_edge(path, i, e) for i, e in enumerate(raw["edges"])),
                 tuple(sorted(set(raw) - set(RECORD_FIELDS))))
    _verify(rec, tau)
    written = {f: raw[f] for f in RECORD_FIELDS}
    if payload(rec) != written:
        drifted = sorted(f for f in RECORD_FIELDS if payload(rec)[f] != written[f])
        raise CharacterizationError(f"{rel(path)}: round-trip changed {drifted}")
    return rec


def payload(rec: Record) -> dict:
    return {"trait": rec.trait, "run_id": rec.run_id, "arm": rec.arm, "cycle": rec.cycle,
            "phase": rec.phase, "k": rec.k, "k_ranked": rec.k_ranked, "n_ablated": rec.n_ablated,
            "faithfulness": rec.faithfulness, "sufficient": rec.sufficient,
            "metric_full": rec.metric_full, "metric_empty": rec.metric_empty,
            "curve": [dict(pt) for pt in rec.curve], "edges": [list(e) for e in rec.edges]}


def _edge(path: Path, i: int, e: object) -> EdgeKey:
    if not isinstance(e, list) or len(e) != 3:
        raise CharacterizationError(f"{rel(path)}: edges[{i}] = {e!r} is not a 3-element "
                                    "[child, parent, qkv]")
    child, parent, qkv = e
    if not isinstance(child, str) or not isinstance(parent, str):
        raise CharacterizationError(f"{rel(path)}: edges[{i}] = {e!r} child/parent must be "
                                    "node names")
    if qkv not in QKV_VALUES:
        raise CharacterizationError(f"{rel(path)}: edges[{i}] qkv = {qkv!r} not in {QKV_VALUES}")
    return (child, parent, qkv)


def _verify(rec: Record, tau: float) -> None:
    p = rec.path
    _index(p, "k", rec.k)
    _index(p, "k_ranked", rec.k_ranked)
    if rec.k != len(rec.edges):
        raise CharacterizationError(f"{rel(p)}: k = {rec.k} but len(edges) = {len(rec.edges)}")
    if len(rec.edge_keys) != len(rec.edges):
        raise CharacterizationError(f"{rel(p)}: edges holds duplicates ({len(rec.edges)} entries, "
                                    f"{len(rec.edge_keys)} distinct)")
    if rec.k > rec.k_ranked:
        raise CharacterizationError(f"{rel(p)}: k = {rec.k} exceeds k_ranked = {rec.k_ranked}; "
                                    "pruning can only remove edges")
    if not isinstance(rec.phase, str) or not rec.phase:
        raise CharacterizationError(f"{rel(p)}: phase = {rec.phase!r} must be a non-empty string")
    _index(p, "n_ablated", rec.n_ablated)
    if rec.n_ablated < 0:
        raise CharacterizationError(f"{rel(p)}: n_ablated = {rec.n_ablated} is negative")
    if rec.phase == BASELINE_PHASE and rec.n_ablated:
        raise CharacterizationError(f"{rel(p)}: phase {BASELINE_PHASE!r} carries n_ablated = "
                                    f"{rec.n_ablated}; nothing is ablated before the first cut")
    if not rec.curve:
        raise CharacterizationError(f"{rel(p)}: curve is empty")
    for i, pt in enumerate(rec.curve):
        absent = [f for f in CURVE_FIELDS if f not in pt]
        if absent:
            raise CharacterizationError(f"{rel(p)}: curve[{i}] missing {absent}")
        _index(p, f"curve[{i}].k", pt["k"])
        _index(p, f"curve[{i}].k_ranked", pt["k_ranked"])
        _finite(p, f"curve[{i}].faithfulness", pt["faithfulness"])
        _finite(p, f"curve[{i}].metric", pt["metric"])
        if pt["k"] > pt["k_ranked"]:
            raise CharacterizationError(f"{rel(p)}: curve[{i}] k = {pt['k']} exceeds "
                                        f"k_ranked = {pt['k_ranked']}; pruning can only remove edges")
    cursors = [pt["k_ranked"] for pt in rec.curve]
    if any(b <= a for a, b in zip(cursors, cursors[1:])):
        raise CharacterizationError(f"{rel(p)}: curve k_ranked values are not strictly "
                                    f"increasing: {cursors}")
    if rec.k_ranked not in cursors:
        raise CharacterizationError(f"{rel(p)}: chosen k_ranked = {rec.k_ranked} is not in the "
                                    f"curve {cursors}")
    _finite(p, "faithfulness", rec.faithfulness)
    _finite(p, "metric_full", rec.metric_full)
    _finite(p, "metric_empty", rec.metric_empty)
    chosen = rec.curve[cursors.index(rec.k_ranked)]
    if (chosen["k"], chosen["faithfulness"]) != (rec.k, rec.faithfulness):
        raise CharacterizationError(
            f"{rel(p)}: record disagrees with its own curve entry at k_ranked = {rec.k_ranked}: "
            f"record (k={rec.k}, faithfulness={rec.faithfulness}) vs "
            f"curve entry (k={chosen['k']}, faithfulness={chosen['faithfulness']})")
    if rec.metric_full == rec.metric_empty:
        raise CharacterizationError(f"{rel(p)}: metric_full == metric_empty == {rec.metric_full!r}, "
                                    "so faithfulness has a zero denominator")
    if not isinstance(rec.sufficient, bool):
        raise CharacterizationError(f"{rel(p)}: sufficient = {rec.sufficient!r} must be a bool")
    # exact-boundary records are tolerated: 0.8 stored and 0.8 recomputed can differ in the last bit.
    if rec.sufficient != (rec.faithfulness >= tau) and abs(rec.faithfulness - tau) > TAU_BOUNDARY_EPS:
        raise CharacterizationError(f"{rel(p)}: sufficient = {rec.sufficient} but faithfulness "
                                    f"{rec.faithfulness} against tau {tau} says "
                                    f"{rec.faithfulness >= tau}")


def _index(path: Path, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CharacterizationError(f"{rel(path)}: {name} = {value!r} must be an int")


def _finite(path: Path, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CharacterizationError(f"{rel(path)}: {name} = {value!r} is not a number")
    if not math.isfinite(value):
        raise CharacterizationError(f"{rel(path)}: {name} = {value!r} is not finite")


def scan(root: Path) -> list[Path]:
    return sorted(root.rglob(RECORD_FILENAME))


def load_all(found: list[Path], tau: float) -> tuple[list[Record], list[str]]:
    """Every record that parses, and one precise rejection message per record that does not."""
    ok, bad = [], []
    for p in found:
        try:
            ok.append(read(p, tau))
        except CharacterizationError as e:
            bad.append(str(e))
    return ok, bad


# --- across a curve and across cycles ---

def groups(records: list[Record]) -> dict[tuple[str, str, str], list[Record]]:
    """One arm's records, cycle-ordered, per (trait, run_id, arm)."""
    by: dict[tuple[str, str, str], list[Record]] = {}
    for r in records:
        by.setdefault(r.group, []).append(r)
    return {g: sorted(rs, key=lambda r: r.cycle) for g, rs in by.items()}


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def ious(seq: list[Record]) -> list[float]:
    return [jaccard(a.edge_keys, b.edge_keys) for a, b in zip(seq, seq[1:])]


def forced_disjoint(seq: list[Record]) -> bool:
    """Every consecutive IoU exactly 0.0 and every cycle disjoint from all earlier cycles: the
    `candidate_edges - ablated` signature, where drift is 0 by construction, not by measurement."""
    if any(i != 0.0 for i in ious(seq)):
        return False
    seen: frozenset = frozenset()
    for r in seq:
        if seen and r.edge_keys & seen:
            return False
        seen |= r.edge_keys
    return True


def ablation_falls(seq: list[Record]) -> list[tuple[Record, Record]]:
    """Consecutive pairs where the cumulative ablation shrank, which a persistent mask cannot do."""
    return [(a, b) for a, b in zip(seq, seq[1:]) if b.n_ablated < a.n_ablated]


def dips(records: list[Record]) -> list[Dip]:
    return [Dip(a["faithfulness"] - b["faithfulness"], b["k"] - a["k"], a["k_ranked"],
                b["k_ranked"], r)
            for r in records
            for a, b in zip(r.curve, r.curve[1:])
            if a["faithfulness"] > b["faithfulness"]]


def cut_edges(cdir: Path) -> tuple[Path, frozenset[EdgeKey]] | None:
    """The cumulative ablated set the loop wrote beside a record, if it wrote one."""
    for name in CUT_FILES:
        p = cdir / name
        if p.exists():
            raw = json.loads(p.read_text())
            return p, frozenset(_cut_edge(p, e) for e in (raw["edges"] if isinstance(raw, dict)
                                                          else raw))
    return None


def _cut_edge(path: Path, e: object) -> EdgeKey:
    """Accepts both cut-file shapes: {child,parent,qkv} (ablation.json) and [child,parent,qkv]."""
    if isinstance(e, dict):
        absent = [f for f in ("child", "parent", "qkv") if f not in e]
        if absent:
            raise CharacterizationError(f"{rel(path)}: ablated edge {e!r} missing {absent}")
        return (e["child"], e["parent"], e["qkv"])
    if isinstance(e, list) and len(e) == 3:
        return (e[0], e[1], e[2])
    raise CharacterizationError(f"{rel(path)}: ablated edge {e!r} is neither "
                                "{child, parent, qkv} nor [child, parent, qkv]")
