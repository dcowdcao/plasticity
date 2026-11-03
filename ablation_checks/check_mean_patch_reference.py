"""ENTRYPOINT — check that the mean-patch reference is per writer node, not one number for the graph.

EAP-IG's `compute_mean_activations` accumulates `(n_forward, d_model)` and returns `means.mean(0)`,
collapsing the node axis. Indexing that by forward index is still legal (n_forward=157 < d_model=768),
so each ablated edge would get one scalar broadcast over all channels. Its own consumer,
`evaluate_graph`, broadcasts the reference as (1, 1, n_forward, d_model), which lines up only per node.

Checks: the reference has the per-node shape; collapsing it reproduces the upstream return exactly;
the node axis carries signal; Ablation hands each reader a (d_model,) vector; and a one-edge ablation
moves the logits differently under the two references.

  python -m ablation_checks.check_mean_patch_reference
"""
import argparse
import sys

from eap.utils import compute_mean_activations

from circuit_discovery import eap_ig
from helper_scripts import artifact_paths as paths
from iterative_pruning.ablate_edges import Ablation, mean_activations
from organisms.load_organism import load_organism
from trait_datasets.prompts import discovery_prompts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="toxicity/gpt2")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    trait = paths.parse_trait(a.trait)
    model, _ = load_organism(trait, a.device)
    eap_ig.prepare_model(model)
    misaligned, _, pool = discovery_prompts(trait)
    means, graph = mean_activations(model, pool)

    loader = [pool[i:i + 8] for i in range(0, len(pool), 8)]
    upstream = compute_mean_activations(model, graph, loader)

    failures = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<34}{detail}", flush=True)
        if not ok:
            failures.append(name)

    want = (graph.n_forward, model.cfg.d_model)
    check("per-node shape", tuple(means.shape) == want, f"{tuple(means.shape)} vs want {want}")
    delta = (means.mean(0) - upstream).abs().max().item()
    check("collapses to the upstream", delta == 0.0, f"max|Δ| = {delta:g}")
    spread = (means.std(0).mean() / means.abs().mean()).item()
    check("node axis carries signal", spread > 0.1, f"cross-node std / |mean| = {spread:.3f}")

    edge = next(e for e in graph.edges.values() if e.qkv is not None)
    key = frozenset({(edge.child.name, edge.parent.name, edge.qkv)})
    ablation = Ablation(model, key, means, graph)
    vec = ablation._recon[0].parents[0][2]
    check("reader gets a vector", vec.shape == (model.cfg.d_model,), f"{tuple(vec.shape)}")

    prompt = misaligned[:1]
    with Ablation(model, key, means, graph).applied(model):
        fixed = model(prompt)
    collapsed = means.mean(0).expand(graph.n_forward, model.cfg.d_model).contiguous()
    with Ablation(model, key, collapsed, graph).applied(model):
        old = model(prompt)
    moved = (fixed - old).abs().max().item()
    check("differs from the collapsed patch", moved > 1e-3, f"max|Δ logit| = {moved:.4f} on one edge")

    print(f"\n{len(failures)} failure(s)" + (f": {', '.join(failures)}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
