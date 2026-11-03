"""How the circuit implementing the misaligned trait changes as the trait is
repeatedly ablated and relearned. Pure set + depth statistics over the per-cycle
sufficiency records — no I/O, no model.

The sufficiency search sees the full graph every cycle, so consecutive circuits can
overlap, unlike the loop's `cut_drift_iou`, whose 0.0 is forced by excluding
already-ablated edges from the search space. Set overlap alone cannot separate
"moved two layers deeper" from "became more distributed", so an Evolution carries
four descriptors: the size/faithfulness trajectory, drift (consecutive and
against the first circuit), the stable core the trait keeps returning to, and the
layer-depth profile.

A record's edges are the pruned, connected circuit (EAP-IG's evaluate_graph
prunes the attribution prefix before scoring), so size is len(edges), never
`k_ranked`, the prefix cursor.

At cycle i>0 the cumulative ablation (`n_ablated` edges) is held out of the graph,
so faithfulness is relative to the ablated model and a faithfulness trajectory
compares different baselines; `n_ablated` travels with it so that stays visible.
"""
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

from circuit_discovery.circuit_graphs import EdgeKey

PHASES = ("baseline", "post_retrain")

EdgeSet = frozenset[EdgeKey]

_ATTN = re.compile(r"a(\d+)\.h\d+")           # EAP-IG node grammar: a{layer}.h{head}
_MLP = re.compile(r"m(\d+)")                  # m{layer}
_HOOK = re.compile(r"blocks\.(\d+)\.(.+)")    # the TransformerLens hook a node wraps
_BOUNDARY = ("input", "logits", "hook_embed")
_LAYER_HOOKS = ("hook_attn_in", "hook_q_input", "hook_k_input", "hook_v_input",
                "attn.hook_result", "hook_mlp_in", "hook_mlp_out")
_LOGIT_HOOK = "hook_resid_post"               # blocks.{n-1}.hook_resid_post is the logits node


def layer(node: str) -> Optional[int]:
    """Layer index of one edge endpoint; None for the graph's boundary (input /
    logits), which has no depth by construction. Both grammars the loop emits are
    accepted: EAP-IG node names ("input" / "a3.h9" / "m7" / "logits") and the
    TransformerLens hooks those nodes wrap ("blocks.5.attn.hook_result"). Anything
    else raises."""
    if node in _BOUNDARY:
        return None
    m = _ATTN.fullmatch(node) or _MLP.fullmatch(node)
    if m:
        return int(m.group(1))
    m = _HOOK.fullmatch(node)
    if m and m.group(2) == _LOGIT_HOOK:
        return None
    if m and m.group(2) in _LAYER_HOOKS:
        return int(m.group(1))
    raise ValueError(f"unparseable circuit node name {node!r}")


@dataclass(frozen=True)
class CircuitRecord:
    cycle: int
    phase: str
    k_ranked: int          # attribution-prefix size the connected circuit was pruned out of
    n_ablated: int         # already-cut edges held out of the graph; the baseline faithfulness is against
    faithfulness: float
    sufficient: bool
    edges: EdgeSet

    @property
    def size(self) -> int:
        return len(self.edges)


def record(payload: dict) -> CircuitRecord:
    """One sufficient_circuit.json payload. A cycle with no circuit is an absent
    file, never a partial record, so every field here is required. `size` comes
    from the edges themselves rather than the payload's redundant `k`."""
    phase = payload["phase"]
    assert phase in PHASES, phase
    return CircuitRecord(
        cycle=payload["cycle"],
        phase=phase,
        k_ranked=payload["k_ranked"],
        n_ablated=payload["n_ablated"],
        faithfulness=payload["faithfulness"],
        sufficient=payload["sufficient"],
        edges=frozenset((child, parent, qkv) for child, parent, qkv in payload["edges"]),
    )


def jaccard(a: EdgeSet, b: EdgeSet) -> float:
    """Edge-set IoU; 0.0 when both are empty."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


@dataclass(frozen=True)
class Drift:
    from_cycle: int
    to_cycle: int
    iou: float
    n_shared: int
    n_added: int
    n_dropped: int


def drift(a: CircuitRecord, b: CircuitRecord) -> Drift:
    return Drift(from_cycle=a.cycle, to_cycle=b.cycle, iou=jaccard(a.edges, b.edges),
                 n_shared=len(a.edges & b.edges), n_added=len(b.edges - a.edges),
                 n_dropped=len(a.edges - b.edges))


def consecutive_drift(records: Sequence[CircuitRecord]) -> list[Drift]:
    """Drift between records adjacent in the sequence. Cycles without a circuit are
    absent from `records`, so a pair can span a gap; read from_cycle/to_cycle."""
    return [drift(a, b) for a, b in zip(records, records[1:])]


@dataclass(frozen=True)
class StableCore:
    n_circuits: int
    m: int                        # recurrence threshold: present in >= m of n_circuits
    always: EdgeSet               # present in every circuit
    recurring: EdgeSet            # present in >= m
    union: EdgeSet
    counts: dict[EdgeKey, int]    # circuits each union edge appears in

    @property
    def always_frac(self) -> float:
        return len(self.always) / len(self.union) if self.union else 0.0


def stable_core(records: Sequence[CircuitRecord], m: int) -> StableCore:
    """The edges the trait returns to: present in all circuits, and in >= m of them.
    Biased toward never-ablated edges — an edge the loop cuts at cycle i is held out
    of the graph at i+1, so it cannot recur however central it was."""
    counts = Counter(e for r in records for e in r.edges)
    n = len(records)
    return StableCore(
        n_circuits=n,
        m=m,
        always=frozenset(e for e, c in counts.items() if c == n),
        recurring=frozenset(e for e, c in counts.items() if c >= m),
        union=frozenset(counts),
        counts=dict(counts),
    )


@dataclass(frozen=True)
class DepthProfile:
    """`mean_depth` moves when the circuit relocates deeper; `span` moves when it
    stretches — different evolution claims, both readable off a connected circuit.
    The Optionals are None only for a circuit touching nothing but the boundary."""
    cycle: int
    layers: dict[int, int]          # layer -> distinct nodes the circuit touches there
    n_boundary: int                 # input / logits endpoints, which have no depth
    mean_depth: Optional[float]
    min_depth: Optional[int]
    max_depth: Optional[int]

    @property
    def span(self) -> Optional[int]:
        """Layers the circuit reaches across, both ends included."""
        return None if self.max_depth is None else self.max_depth - self.min_depth + 1


def depth_profile(rec: CircuitRecord) -> DepthProfile:
    """Where in depth the circuit lives, counting distinct nodes per layer (not
    edges, which would let one high-fan-out node dominate)."""
    nodes = {n for (child, parent, _) in rec.edges for n in (child, parent)}
    depths = [layer(n) for n in nodes]
    layers = Counter(d for d in depths if d is not None)
    total = sum(layers.values())
    return DepthProfile(
        cycle=rec.cycle,
        layers=dict(sorted(layers.items())),
        n_boundary=sum(1 for d in depths if d is None),
        mean_depth=sum(d * c for d, c in layers.items()) / total if total else None,
        min_depth=min(layers) if layers else None,
        max_depth=max(layers) if layers else None,
    )


@dataclass(frozen=True)
class Trajectory:
    cycles: tuple[int, ...]
    phases: tuple[str, ...]
    sizes: tuple[int, ...]              # connected, post-prune edge counts
    k_ranked: tuple[int, ...]           # how far down the attribution ranking each search reached
    n_ablated: tuple[int, ...]          # read faithfulness against this: the baseline moves with it
    faithfulness: tuple[float, ...]
    sufficient: tuple[bool, ...]


def trajectory(records: Sequence[CircuitRecord]) -> Trajectory:
    return Trajectory(
        cycles=tuple(r.cycle for r in records),
        phases=tuple(r.phase for r in records),
        sizes=tuple(r.size for r in records),
        k_ranked=tuple(r.k_ranked for r in records),
        n_ablated=tuple(r.n_ablated for r in records),
        faithfulness=tuple(r.faithfulness for r in records),
        sufficient=tuple(r.sufficient for r in records),
    )


@dataclass(frozen=True)
class Evolution:
    trajectory: Trajectory
    drift: tuple[Drift, ...]
    drift_from_first: tuple[Drift, ...]   # a walk away from cycle 0 vs. a wobble around it
    core: StableCore
    depth: tuple[DepthProfile, ...]


def analyze(records: Sequence[CircuitRecord], m: int) -> Evolution:
    """`records` in cycle order, one per cycle that produced a circuit."""
    return Evolution(
        trajectory=trajectory(records),
        drift=tuple(consecutive_drift(records)),
        drift_from_first=tuple(drift(records[0], r) for r in records[1:]),
        core=stable_core(records, m),
        depth=tuple(depth_profile(r) for r in records),
    )
