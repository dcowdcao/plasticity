"""ENTRYPOINT (GPU) — check that the EAP-IG edge score equals Δ under mean-patch ablation.

The two-sided score reads `graph.edges[e].score` as `M(ablate e) − M(intact)`. An inverted sign would
put trait-amplifying edges first while the sweep still printed a tidy table, so this attributes once,
ablates a sample of single edges with the sweep's operator, and compares.

  python -m attribution_ablation.check_attribution_is_delta --trait toxicity/gpt2
"""
import argparse

import torch
from eap.attribute import attribute
from eap.graph import Graph

from attribution_ablation import sheaf_attribution as sa
from circuit_discovery import eap_ig
from circuit_discovery.probe import fit_probe
from helper_scripts import artifact_paths as paths
from iterative_pruning.ablate_edges import Ablation, mean_activations
from organisms.load_organism import load_organism
from trait_datasets.prompts import discovery_prompts

# attribution costs 25 nodes x ig_steps backwards per batch; the check needs shared prompts, not many
N_PROMPTS = 16
N_PER_BAND = 8          # edges sampled from each of {most negative, most positive, middle}


def _exact_metric(model, batches, metric, ablation) -> float:
    with ablation.applied(model):
        return sa._metric_value(model, batches, metric)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="toxicity/gpt2")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    trait = paths.parse_trait(a.trait)
    model, _ = load_organism(trait, a.device)
    eap_ig.prepare_model(model)
    misaligned, aligned, pool = discovery_prompts(trait)
    prompts = misaligned[:N_PROMPTS]
    probe = fit_probe(model, misaligned, aligned, a.device)
    graph = Graph.from_model(model)
    batches = sa._loader(model, prompts, 8)
    pool_loader = sa._loader(model, pool, 8)

    intact = sa._metric_value(model, batches, probe.metric)
    delta = sa._attribute_scores(attribute, model, graph, batches, probe.metric, pool_loader,
                                 sa.IG_STEPS)
    keys = [(e.child.name, e.parent.name, e.qkv) for e in graph.edges.values()]
    print(f"{len(keys)} edges, intact probe metric {intact:.2f}\n", flush=True)

    order = delta.argsort()
    picks = torch.cat([order[:N_PER_BAND], order[-N_PER_BAND:],
                       order[len(order) // 2:len(order) // 2 + N_PER_BAND]]).tolist()

    means, abl_graph = mean_activations(model, pool)
    print(f"{'edge':<34}{'attributed Δ':>15}{'exact Δ':>12}{'sign':>7}", flush=True)
    predicted, measured = [], []
    for i in picks:
        exact = _exact_metric(model, batches, probe.metric,
                              Ablation(model, frozenset([keys[i]]), means, abl_graph)) - intact
        predicted.append(float(delta[i]))
        measured.append(exact)
        agree = "ok" if float(delta[i]) * exact > 0 else "FLIP"
        print(f"{keys[i][1] + '->' + keys[i][0] + (keys[i][2] or ''):<34}"
              f"{float(delta[i]):>15.3f}{exact:>12.3f}{agree:>7}", flush=True)

    pred, meas = torch.tensor(predicted), torch.tensor(measured)
    same_sign = int(((pred * meas) > 0).sum())
    print(f"\npearson(attributed, exact) = {sa._pearson(pred, meas):.4f}   "
          f"sign agreement {same_sign}/{len(picks)}")
    print("The score is a first-order estimate, so magnitudes drift; the sweep uses only the sign "
          "and the ranking.")


if __name__ == "__main__":
    main()
