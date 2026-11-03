"""Weight-space attribution, and one ranking pool over weights and edges together.

For a component with value `a` and ablation reference `â`, attribution patching estimates

    Δ M ≈ (â − a) · ∂M/∂a

integrated over `ig_steps` points on the a→â path. Edge attribution (sheaf_attribution) takes `a` to
be an edge's activation and `â` its writer's pool mean; here `a` is a parameter and `â` its
`matrix_mean_reference` ("a typical value from its own tensor"), the same counterfactual on the other
axis. A uniform reference is what makes the two comparable; reverting to the pre-fine-tuning base
would ask a different question. Each side of `two_sided_score` is divided by its own baseline, so the
score is scale-free, which is what lets 85,017,600 weights and 32,491 edges share one ranking.

Read the frontier against benign KL, not k: greedy top-k takes the most benign-efficient objects
first regardless of type, so the weight fraction of a cut is a finding, not a confound.

A weight pass costs ig_steps backwards per batch (an edge pass costs 25 nodes x ig_steps), but holds
two fp32 vectors over 85M parameters, ~340 MB each.
"""
import gc
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

from attribution_ablation.sheaf_attribution import (
    IG_STEPS, Coupling, TwoSided, coupling, two_sided_score)
from circuit_discovery.circuit_graphs import EdgeKey


@dataclass(frozen=True)
class Pool:
    """The flat index space the two granularities share, and the only handle on a member of it.

    Index `i < n_edges` is `edges[i]`; index `n_edges + j` is element `j` of the name-sorted
    concatenation of the weight vectors. A ranking is therefore an int64 tensor and nothing else:
    one Python object per member would cost ~5 GB of host RAM at 85,050,091 objects."""
    edges: tuple[EdgeKey, ...]
    params: tuple[str, ...]
    bounds: Tensor            # cumulative weight numel, len(params) + 1, bounds[0] == 0

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def __len__(self) -> int:
        return self.n_edges + int(self.bounds[-1])

    def decode(self, idx: Tensor) -> tuple[frozenset[EdgeKey], dict[str, Tensor]]:
        """A slice of a ranking, split back into the two operators that can apply it. The weight side
        stays a tensor: `.tolist()` on a large cut is 2.7 GB of Python ints for no gain."""
        edges = frozenset(self.edges[i] for i in idx[idx < self.n_edges].tolist())
        flat = idx[idx >= self.n_edges] - self.n_edges
        slot = torch.bucketize(flat, self.bounds[1:], right=True)
        chosen = {self.params[s]: flat[slot == s] - self.bounds[s]
                  for s in slot.unique().tolist()}
        return edges, chosen


@dataclass(frozen=True)
class PoolScores:
    """One ranking over both granularities, descending by score."""
    pool: Pool
    order: Tensor             # int64 indices into `pool`
    sides: TwoSided
    coupling: Coupling
    m_mis: float
    m_ben: float

    def head(self, k: int) -> tuple[frozenset[EdgeKey], dict[str, list[int]]]:
        return self.pool.decode(self.order[:k])

    def n_edges_in_head(self, k: int) -> int:
        return int((self.order[:k] < self.pool.n_edges).sum())


def weight_deltas(model, batches, metric, reference: dict[str, Tensor],
                  ig_steps: int = IG_STEPS) -> dict[str, Tensor]:
    """Δ M per parameter element under reverting it to `reference`, integrated along that path.

    The interpolation walks the weight path w(α) = ref + α(w − ref) and accumulates ∂M/∂w at each
    step, which is the weight-space analogue of EAP-IG walking the activation path. Parameters are
    mutated in place and restored in a finally, so an exception cannot leave the organism altered."""
    params = {n: p for n, p in model.named_parameters() if n in reference}
    if not params:
        raise ValueError("no parameters matched the reference; check _maskable exclusions")
    original = {n: p.detach().clone() for n, p in params.items()}
    grad_sum = {n: torch.zeros_like(p) for n, p in params.items()}
    was_grad = {n: p.requires_grad for n, p in params.items()}

    try:
        for n, p in params.items():
            p.requires_grad_(True)
        for step in range(ig_steps):
            alpha = (step + 0.5) / ig_steps            # midpoint rule, as EAP-IG uses
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(reference[n] + alpha * (original[n] - reference[n]))
            model.zero_grad(set_to_none=True)
            _backward_mean_metric(model, batches, metric)
            with torch.no_grad():
                for n, p in params.items():
                    if p.grad is not None:
                        grad_sum[n] += p.grad
            model.zero_grad(set_to_none=True)
    finally:
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(original[n])
                p.requires_grad_(was_grad[n])
        del original
        gc.collect()
        torch.cuda.empty_cache()

    return {n: ((reference[n] - p.detach()) * grad_sum[n] / ig_steps).flatten()
            for n, p in params.items()}


def _backward_mean_metric(model, batches, metric) -> None:
    """Accumulate d(mean metric)/dw into `.grad` one batch at a time.

    One graph over every batch would hold all their activations live (the logits alone are 412 MB
    per 4x513-token batch). d(Σ)/dw = Σ d/dw, so per-batch backward is exact."""
    from eap.utils import tokenize_plus

    for clean, _, label in batches:
        tokens, mask, lengths, _ = tokenize_plus(model, clean)
        logits = model(tokens, attention_mask=mask)
        (metric(logits, None, lengths, label.to(logits.device)) / len(batches)).backward()


def pool_scores(edge_keys: tuple[EdgeKey, ...], edge_mis: Tensor, edge_ben: Tensor,
                weight_mis: dict[str, Tensor], weight_ben: dict[str, Tensor],
                m_mis: float, m_ben: float) -> PoolScores:
    """Merge both granularities into one ranking. The score is computed over the concatenated pool,
    so the floor quantile is taken across every ablatable object rather than per granularity — a
    per-granularity floor would rescale one side against the other and break the comparison.

    Pass `edge_keys=()` with empty edge deltas for the weights-only ranking; the floor then moves,
    which is correct — it is a quantile of the pool actually being ranked."""
    names = sorted(weight_mis)
    bounds = torch.tensor([0] + [weight_mis[n].numel() for n in names]).cumsum(0)
    pool = Pool(edges=tuple(edge_keys), params=tuple(names), bounds=bounds)

    delta_mis = torch.cat([edge_mis.double().cpu()] + [weight_mis[n].double().cpu() for n in names])
    delta_ben = torch.cat([edge_ben.double().cpu()] + [weight_ben[n].double().cpu() for n in names])
    if len(delta_mis) != len(pool):
        raise ValueError(f"pool is {len(pool)} objects but the deltas are {len(delta_mis)}")
    sides = two_sided_score(delta_mis, delta_ben, m_mis, m_ben)
    order = sides.score.argsort(descending=True)
    return PoolScores(
        pool=pool, order=order,
        sides=TwoSided(sides.score[order].float(), sides.removal[order].float(),
                       sides.perturbation[order].float(), sides.floor),
        coupling=coupling(delta_mis, delta_ben), m_mis=m_mis, m_ben=m_ben)


def weight_mask(shapes: dict[str, torch.Size], chosen: dict[str, Tensor]) -> dict[str, Tensor]:
    """Flat indices back to the per-parameter boolean masks WeightAblation applies."""
    masks = {}
    for name, flat in chosen.items():
        m = torch.zeros(shapes[name].numel(), dtype=torch.bool)
        m[flat] = True
        masks[name] = m.reshape(shapes[name])
    return masks


class SheafAblation:
    """Both granularities as one intervention, so `preserved` and the oracle treat them alike.

    Duck-typed on `.applied(model)` like its two halves, which is all the gate requires. Weights are
    entered first: `WeightAblation` mutates parameters while `Ablation` installs hooks, so the hooks
    run against the already-reverted weights. The edge means come from the caller and should be
    computed on the model the edges were ranked on."""

    def __init__(self, edges=None, weights=None):
        self.edges, self.weights = edges, weights
        self.ablated = frozenset(edges.ablated) if edges is not None else frozenset()

    @contextmanager
    def applied(self, model):
        with ExitStack() as stack:
            for part in (self.weights, self.edges):
                if part is not None:
                    stack.enter_context(part.applied(model))
            yield model
