"""EAP-IG sufficiency: the smallest connected edge circuit that reproduces the misalignment on its
own, with everything outside it mean-patched to the aligned pool. This characterises the mechanism;
the loop's ablation target is the necessity cut, which differs from it under mean-patching.

Faithfulness is measured in probe-metric space, not generated firing rate: keeping a ~150-edge
circuit mean-patches the other ~32k edges, far past the model's ~200-1,000-edge damage threshold, so
generations are garbage however good the circuit is. faith(k) = (M_k - M_empty) / (M_full - M_empty),
with M_empty = every edge patched out and M_full = the intact forward or, given a cumulative `ablated`
cut, that cut's model (the trait being characterised is the one that recovered through the cut).

EAP-IG's evaluate_graph prunes to a connected subgraph before measuring (`graph.prune()`), so an
|attribution| prefix of size `k_ranked` measures the surviving circuit of size `k` <= `k_ranked`, and
that survivor is what `edges` reports. Short prefixes prune to nothing (curve[0] normally reads k=0,
faith~0), and the prefix -> circuit map is non-monotonic, so "smallest sufficient" minimises `k`.

CUDA-only; the attribution pass holds ~58 GiB on the GPU. Pool means are computed once per node
(ablate_edges.pool_mean_per_node) rather than as EAP-IG's global average, so each swept k costs one
forward over the misaligned prompts."""
from dataclasses import dataclass

import eap.evaluate as eap_evaluate
from eap.attribute import attribute
from eap.evaluate import evaluate_baseline, evaluate_graph
from eap.graph import Graph

from helper_scripts import artifact_paths as paths
from circuit_discovery import eap_ig
from circuit_discovery.circuit_graphs import EdgeKey
from circuit_discovery.probe import fit_probe
from iterative_pruning.ablate_edges import pool_mean_per_node
from trait_datasets.prompts import discovery_prompts

K_GRID = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)   # |attribution| prefix sizes swept
FAITH_TAU = 0.8         # faithfulness a circuit must reach to count as sufficient
MIN_GAP_FRAC = 1e-3     # min |M_full - M_empty| as a fraction of |M_full| — the probe is
                        # unnormalised (~5e4), so an absolute floor says nothing


class SufficiencyError(Exception):
    pass


@dataclass(frozen=True)
class SufficientCircuit:
    edges: frozenset[EdgeKey]   # the post-prune connected circuit — what was evaluated
    k: int                      # len(edges), the true circuit size
    k_ranked: int               # the |attribution| prefix that produced it
    faithfulness: float
    curve: list[dict]           # {"k_ranked", "k", "faithfulness", "metric"} per swept prefix
    metric_full: float
    metric_empty: float
    sufficient: bool
    n_ablated: int              # size of the cumulative cut measured through (0 = pristine model)


def find_sufficient_circuit(trait: paths.TraitId, model, *, device: str = "cuda",
                            tau: float = FAITH_TAU, k_grid: tuple[int, ...] = K_GRID,
                            batch_size: int = 8,
                            ablated: frozenset[EdgeKey] = frozenset()) -> SufficientCircuit:
    """`model` = the organism to characterise (hooks are set here). Sweeps `k_grid` prefixes of
    the |attribution|-ranked edges (an edge that moves the metric either way is part of the
    mechanism) and returns the smallest connected circuit reaching `tau`, else the most faithful
    one with sufficient=False.

    `ablated` = the loop's cumulative cut. Those edges are dropped from the ranking and never put
    in the graph, so out-of-graph mean-patching reproduces the loop's ablation without hooks and
    faithfulness is relative to the ablated model. Attribution itself runs on the unablated forward."""
    eap_ig.prepare_model(model)
    misaligned, aligned, pool = discovery_prompts(trait)
    probe = fit_probe(model, misaligned, aligned, device)
    behavior_loader = eap_ig.loader(misaligned, batch_size)
    pool_loader = eap_ig.loader(pool, batch_size)

    graph = Graph.from_model(model)
    attribute(model, graph, eap_ig.loader(eap_ig.subsample(misaligned, 0), batch_size), probe.metric,
              method=eap_ig.ATTR_METHOD, intervention="mean",
              intervention_dataloader=eap_ig.loader(eap_ig.subsample(pool, 0), batch_size),
              ig_steps=eap_ig.IG_STEPS)
    ranked = sorted([e for e in graph.edges.values()
                     if (e.child.name, e.parent.name, e.qkv) not in ablated],
                    key=lambda e: abs(float(e.score)), reverse=True)

    def eval_masked() -> float:
        return eap_ig.scalar(evaluate_graph(model, graph, behavior_loader, [probe.metric],
                                             intervention="mean", intervention_dataloader=pool_loader,
                                             quiet=True))

    # One pool mean held across the sweep, per node rather than EAP-IG's global average (see the helper).
    pool_mean = pool_mean_per_node(model, graph, pool_loader)
    uncached_mean = eap_evaluate.compute_mean_activations
    eap_evaluate.compute_mean_activations = lambda *a, **k: pool_mean
    try:
        graph.reset()
        metric_empty = eval_masked()
        # The reference is the model the trait actually recovered in: pristine only if nothing is cut.
        if ablated:
            eap_ig.set_complement(graph, ablated)
            metric_full = eval_masked()
        else:
            metric_full = eap_ig.scalar(evaluate_baseline(model, behavior_loader, [probe.metric], quiet=True))
        gap = metric_full - metric_empty
        if abs(gap) <= MIN_GAP_FRAC * abs(metric_full):   # <= so a zero scale can't slip through
            raise SufficiencyError(
                f"degenerate faithfulness scale for {trait}: M_full={metric_full:.6g}, "
                f"M_empty={metric_empty:.6g}, gap={gap:.3g} <= {MIN_GAP_FRAC:g}·|M_full|. The probe "
                "does not separate the intact forward from the fully mean-patched one, so faith is noise")

        curve: list[dict] = []
        live: dict[int, frozenset[EdgeKey]] = {}
        for k_ranked in [k for k in k_grid if k <= len(ranked)]:
            graph.reset()
            for e in ranked[:k_ranked]:
                e.in_graph = True
            m = eval_masked()
            # Survivors are read after evaluating: prune() ran in place, and it only ever drops edges.
            live[k_ranked] = frozenset((e.child.name, e.parent.name, e.qkv)
                                       for e in ranked[:k_ranked] if bool(e.in_graph))
            curve.append({"k_ranked": k_ranked, "k": len(live[k_ranked]),
                          "faithfulness": (m - metric_empty) / gap, "metric": m})
    finally:
        eap_evaluate.compute_mean_activations = uncached_mean

    reached = [c for c in curve if c["faithfulness"] >= tau]
    chosen = (min(reached, key=lambda c: (c["k"], c["k_ranked"])) if reached
              else max(curve, key=lambda c: c["faithfulness"]))
    edges = live[chosen["k_ranked"]]
    return SufficientCircuit(edges=edges, k=len(edges), k_ranked=chosen["k_ranked"],
                             faithfulness=chosen["faithfulness"], curve=curve,
                             metric_full=metric_full, metric_empty=metric_empty,
                             sufficient=bool(reached), n_ablated=len(ablated))
