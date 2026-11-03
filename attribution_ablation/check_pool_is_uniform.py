"""ENTRYPOINT — is a merged weight+edge ranking unbiased between the two granularities?

Ranking 85,017,600 weights and 32,491 edges in one pool needs a scale-free score (checked on synthetic
tensors in check_score_arithmetic) and a first-order estimate `Δ M ≈ (ref − a)·∂M/∂a` that is
comparably accurate for both kinds of object, which is empirical and is what this measures. For a
sample of each kind it compares the predicted Δ with the Δ measured by ablating that object.

Absolute accuracy is not the question, since first-order attribution is known to be loose. A gap
between the granularities is: it tilts the shared ranking toward the better-linearised side, which
would look exactly like "one granularity dominates the top-k".

  python -m attribution_ablation.check_pool_is_uniform --sample 24
"""
import argparse
import random
import sys

import torch
from eap.attribute import attribute

from attribution_ablation import sheaf_attribution as sa
from attribution_ablation import weight_attribution as wa
from circuit_discovery import disco_gp_finder as dg
from circuit_discovery import eap_ig
from helper_scripts import artifact_paths as paths
from iterative_pruning.ablate_edges import Ablation, mean_activations
from iterative_pruning.ablate_weights import WeightAblation
from organisms.load_organism import load_organism
from trait_datasets.prompts import discovery_prompts

SAMPLE = 24          # objects per granularity; each costs one ablated forward pass


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    denom = (a.norm() * b.norm()).item()
    return float((a * b).sum() / denom) if denom else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="toxicity/gpt2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample", type=int, default=SAMPLE)
    p.add_argument("--benign-windows", type=int, default=8)
    p.add_argument("--mis-prompts", type=int, default=16)
    a = p.parse_args()

    trait = paths.parse_trait(a.trait)
    model, _ = load_organism(trait, a.device)
    eap_ig.prepare_model(model)
    misaligned, _, pool = discovery_prompts(trait)
    from benign_corpus import sequences

    ben = sequences.halves()[0][:a.benign_windows]
    ben_batches = sa._loader(model, ben, 4)
    m_ben = sa._metric_value(model, ben_batches, sa.lm_loss)
    print(f"benign metric on the intact organism: {m_ben:.4f} nats over {len(ben)} windows\n", flush=True)

    # ---- weights: predicted vs measured, on the reference the pool uses ----
    ref = {n: t for n, t in dg.matrix_mean_reference(model).items() if dg._maskable(n)}
    pred_w = wa.weight_deltas(model, ben_batches, sa.lm_loss, ref, ig_steps=3)
    names = sorted(pred_w)
    rng = random.Random(0)
    picks = []
    for _ in range(a.sample):
        n = rng.choice(names)
        picks.append((n, rng.randrange(pred_w[n].numel())))

    shapes = {n: model.get_parameter(n).shape for n, _ in picks}
    wp, wm = [], []
    for name, flat in picks:
        mask = wa.weight_mask({name: shapes[name]}, {name: [flat]})
        with WeightAblation(mask, ref).applied(model):
            measured = sa._metric_value(model, ben_batches, sa.lm_loss) - m_ben
        wp.append(float(pred_w[name][flat]))
        wm.append(measured)
    del pred_w
    torch.cuda.empty_cache()

    # ---- edges: the same question, on the signed Δ. `SheafScores.perturbation` is |Δ_ben|/m_ben,
    # and comparing it with a signed measurement would count positive deltas as sign agreement.
    means, graph = mean_activations(model, pool)
    keys = [(e.child.name, e.parent.name, e.qkv) for e in graph.edges.values()]
    pool_loader = sa._loader(model, pool, 8)
    pred_e = sa._attribute_scores(attribute, model, graph, ben_batches, sa.lm_loss, pool_loader, 3)
    pred_by_key = dict(zip(keys, pred_e.tolist()))
    ep, em = [], []
    for key in rng.sample(keys, a.sample):
        with Ablation(model, frozenset({key}), means, graph).applied(model):
            measured = sa._metric_value(model, ben_batches, sa.lm_loss) - m_ben
        ep.append(pred_by_key[key])
        em.append(measured)

    wpt, wmt = torch.tensor(wp), torch.tensor(wm)
    ept, emt = torch.tensor(ep), torch.tensor(em)
    print(f"{'granularity':<14}{'pearson':>10}{'sign agree':>12}{'|pred| med':>13}{'|meas| med':>13}")
    rows = []
    for label, pr, me in (("weights", wpt, wmt), ("edges", ept, emt)):
        r = _pearson(pr, me)
        agree = float(((pr.sign() == me.sign()) | (me == 0)).double().mean())
        rows.append((label, r, agree))
        print(f"{label:<14}{r:>10.4f}{agree:>12.2%}{pr.abs().median():>13.3e}{me.abs().median():>13.3e}")

    gap = abs(rows[0][1] - rows[1][1])
    print(f"\npearson gap between granularities = {gap:.4f}")
    print("A merged ranking is legitimate when neither side is systematically better predicted.\n"
          "Large gap ⇒ the pool is tilted toward the better-linearised granularity and the top-k\n"
          "composition is an artifact of estimator quality, not of what is worth ablating.")
    sys.exit(0)


if __name__ == "__main__":
    main()
