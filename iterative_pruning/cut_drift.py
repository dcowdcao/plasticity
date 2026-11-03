"""How far a cycle's cut moved from the previous cycle's: the loop's rerouting signal.

Read against the calibrated null rather than against 0: 0.332 is the mean Jaccard between two
independent same-trait searches at different seeds, 0.162 between cuts found at λ off by 3.3×.
A drift of 0.3 is therefore "same circuit, new seed", not "the circuit moved"."""
from circuit_discovery.circuit_graphs import EdgeKey


def edge_iou(a: frozenset[EdgeKey], b: frozenset[EdgeKey]) -> float | None:
    """None, not 0.0, when both cuts are empty: drift against no previous cut is undefined, and
    0.0 would read as "the circuit moved completely"."""
    return len(a & b) / len(a | b) if (a or b) else None
