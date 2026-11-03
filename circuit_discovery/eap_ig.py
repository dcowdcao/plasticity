"""EAP-IG (Hanna et al. 2024, the `eap` package) wiring: model preparation, its dataloader shape, and the
attribution settings every caller shares — method="EAP-IG-activations", intervention="mean" with
the aligned pool as the mean reference."""
import random

from circuit_discovery import circuit_graphs as cg

IG_STEPS = 5              # integrated-gradient steps
SUBSAMPLE_FRAC = 0.7      # bootstrap fraction of the prompts an attribution pass draws
ATTR_METHOD = "EAP-IG-activations"


def prepare_model(model) -> None:
    """Set the hooks EAP-IG and the edge mean-patch require (and ungroup GQA if present)."""
    model.set_use_attn_result(True)
    model.cfg.use_split_qkv_input = True
    if not model.cfg.attn_only:
        model.set_use_hook_mlp_in(True)
    if model.cfg.n_key_value_heads is not None:
        model.cfg.ungroup_grouped_query_attention = True


def loader(prompts: list[str], batch_size: int) -> list:
    """EAP-IG dataloader: (clean, corrupted, label) batches. For mean intervention
    the corrupted slot and label are unused, so they're dummies."""
    return [(prompts[i:i + batch_size], prompts[i:i + batch_size], [0] * len(prompts[i:i + batch_size]))
            for i in range(0, len(prompts), batch_size)]


def subsample(items: list[str], seed: int) -> list[str]:
    k = max(1, int(round(len(items) * SUBSAMPLE_FRAC)))
    return random.Random(seed).sample(items, min(k, len(items)))


def scalar(res) -> float:
    """One float from an evaluate_graph / evaluate_baseline result."""
    r = res[0] if isinstance(res, (list, tuple)) else res
    return float(r.mean()) if hasattr(r, "mean") else float(r)


def set_complement(graph, ablated: frozenset) -> None:
    """All edges in except `ablated`: the full ablated substrate."""
    graph.reset(empty=False)
    for key in ablated:
        e = graph.edges.get(cg.edge_name(key))
        if e is not None:
            e.in_graph = False
