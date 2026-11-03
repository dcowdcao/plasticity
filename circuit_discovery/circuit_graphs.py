"""The edge-key grammar shared by every finder and ablation. Nodes use the EAP-IG Graph's names
("input" / "a3.h9" / "m7" / "logits"); an edge is (child, parent, qkv). Pure Python — no torch."""
from typing import Optional

QKV = Optional[str]                 # "q" | "k" | "v" | None (None into logits / MLP)
EdgeKey = tuple[str, str, QKV]      # (child, parent, qkv)


def edge_name(key: EdgeKey) -> str:
    """The EAP-IG Graph's own edge name, the key into `graph.edges`."""
    child, parent, qkv = key
    return f"{parent}->{child}" if qkv is None else f"{parent}->{child}<{qkv}>"
