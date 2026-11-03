"""Two-sided edge attribution: rank edges by misalignment removed per unit of benign damage.

    score(e) = (-Δ_e M_mis / M_mis) / (|Δ_e M_ben| / M_ben + floor)

Δ_e M is the first-order (EAP-IG) change in a metric when edge e is mean-patched. Each side is a
fraction of its intact value, so the ratio is dimensionless even though M_mis is a logit-space probe
projection (~1e4) and M_ben a loss in nats (~8). `floor` is a small quantile of the normalised benign
side, so an edge with near-zero benign perturbation cannot top the ranking on a division artifact.
M_mis is affine (the probe subtracts the class midpoint), so "fraction removed" is of the margin
above the aligned/misaligned decision point.

The benign metric is cross-entropy, not KL: KL to the intact model is minimised at the intact point,
so its gradient there is zero and first-order attribution would return all zeros.

The two passes run sequentially: ~58 GiB stays resident after an `attribute` call and `empty_cache()`
does not return it, so two co-resident passes run out of memory.

Scope: marginal effects of each edge on the dense, intact organism. Attributing on a masked model
would need `attribute` wrapped in `Ablation(...).applied(model)`, with care over hook ordering on
`hook_result` and over filtering already-masked edges out of the score tensor.
"""
import gc
from contextlib import contextmanager
from dataclasses import dataclass

import eap.attribute as eap_attribute
import torch
import torch.nn.functional as F
from eap.attribute import attribute
from eap.graph import Graph
from eap.utils import tokenize_plus
from torch import Tensor

from circuit_discovery import eap_ig
from circuit_discovery.circuit_graphs import EdgeKey
from helper_scripts import artifact_paths as paths
from iterative_pruning.ablate_edges import pool_mean_per_node
from trait_datasets.prompts import discovery_prompts

IG_STEPS = eap_ig.IG_STEPS
BENIGN_WINDOWS = 64       # attribution costs 25 nodes x ig_steps backwards per batch; windows are long
FLOOR_QUANTILE = 0.05     # of the normalised benign side, so near-zero denominators cannot rank first


@dataclass(frozen=True)
class TwoSided:
    """Per-edge score and the two normalised sides it divides."""
    score: Tensor          # removal / (perturbation + floor)
    removal: Tensor        # -Δ M_mis / M_mis — fraction of the misalignment metric removed
    perturbation: Tensor   # |Δ M_ben| / M_ben
    floor: float


@dataclass(frozen=True)
class Coupling:
    """How tightly the two sides track each other. If they are proportional the ranking is noise:
    every edge gets the same ratio. Whole cuts track tightly (kl_ben/kl_mis 0.81–1.07 across klattr2);
    this is the same question one edge at a time."""
    pearson_log: float
    spearman: float
    slope: float
    intercept: float
    r2: float
    n: int                 # edges with both sides non-zero


@dataclass(frozen=True)
class ScoreSpread:
    """A near-constant score means there is no frontier to find; a heavy right tail means some
    edges really are specific. `rank_vs_removal` is the sharper form of the same question: if the
    benign denominator does not reorder anything, the two-sided score reduces to one-sided
    necessity attribution."""
    min: float
    median: float
    max: float
    top100_over_median: float
    rank_vs_removal: float


@dataclass(frozen=True)
class SheafScores:
    edges: tuple[EdgeKey, ...]   # descending by score
    sides: TwoSided              # reordered to match `edges`
    coupling: Coupling
    spread: ScoreSpread
    m_mis: float
    m_ben: float
    n_mis: int                   # prompts / windows each side was attributed over — noise scales
    n_ben: int                   # with these, and noise attenuates the coupling


def two_sided_score(delta_mis: Tensor, delta_ben: Tensor, m_mis: float, m_ben: float,
                    floor_quantile: float = FLOOR_QUANTILE) -> TwoSided:
    """`delta_*` are Δ under ablation: the EAP-IG attribution score is metric(ablated) −
    metric(intact), so an edge whose removal reduces misalignment has a negative `delta_mis`."""
    if m_mis <= 0 or m_ben <= 0:
        raise ValueError(f"normalisers must be positive, got m_mis={m_mis}, m_ben={m_ben}")
    removal = -delta_mis.double() / m_mis
    perturbation = (delta_ben.double() / m_ben).abs()
    floor = _quantile(perturbation, floor_quantile)
    return TwoSided(removal / (perturbation + floor), removal, perturbation, floor)


def _quantile(x: Tensor, q: float) -> float:
    """`torch.quantile` raises above 2**24 elements and the sheaf pool is 85,050,091. Linear
    interpolation between order statistics, matching torch's default method."""
    pos = q * (x.numel() - 1)
    lo = int(pos)
    order = x.sort().values
    below = float(order[lo])
    return below + (float(order[min(lo + 1, x.numel() - 1)]) - below) * (pos - lo)


def coupling(delta_mis: Tensor, delta_ben: Tensor) -> Coupling:
    both = (delta_mis != 0) & (delta_ben != 0)
    x, y = delta_mis[both].double().abs().log(), delta_ben[both].double().abs().log()
    pearson_log = _pearson(x, y)
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    return Coupling(pearson_log=pearson_log,
                    spearman=_pearson(_rank(delta_mis.abs()), _rank(delta_ben.abs())),
                    slope=slope, intercept=float(y.mean() - slope * x.mean()),
                    r2=pearson_log ** 2, n=int(both.sum()))


def spread(score: Tensor, removal: Tensor) -> ScoreSpread:
    s = score.double().sort(descending=True).values
    median = float(s.median())
    return ScoreSpread(min=float(s[-1]), median=median, max=float(s[0]),
                       top100_over_median=float(s[:100].mean()) / median,
                       rank_vs_removal=_pearson(_rank(score), _rank(removal)))


def sheaf_scores(trait: paths.TraitId, device: str = "cuda", model=None, ig_steps: int = IG_STEPS,
                 batch_size: int = 8, benign_windows: int = BENIGN_WINDOWS,
                 mis_prompts: int | None = None, zero_ablate: bool = False) -> SheafScores:
    """Two sequential attribution passes over the full graph, scored and ranked descending.

    The benign side reads the general-text corpus's faith half, so the selection signal is not the
    data the `preserves` gate scores on. Raises MissingCorpusError if the corpus is unbuilt; there is
    no fallback to trait-local prompts, which come from the organism's own distribution and cannot
    show whether the model still models language.

    `mis_prompts` truncates the misaligned bootstrap draw; None is the whole draw. Both sample sizes
    are cost knobs (a pass is ~125 backwards per batch), and fewer samples mean noisier per-edge Δ,
    which attenuates the measured coupling: a low correlation at a small sample is inconclusive, a
    high one is real."""
    from benign_corpus import sequences
    from circuit_discovery.probe import fit_probe
    from organisms.load_organism import load_organism

    if model is None:
        model, _ = load_organism(trait, device)
    eap_ig.prepare_model(model)

    misaligned, aligned, pool = discovery_prompts(trait)
    probe = fit_probe(model, misaligned, aligned, device)
    graph = Graph.from_model(model)
    keys = [(e.child.name, e.parent.name, e.qkv) for e in graph.edges.values()]

    # the full pool is the mean reference for both passes and for the sweep's Ablation: one intervention
    pool_loader = _loader(model, pool, batch_size)
    mis = eap_ig.subsample(misaligned, 0)[:mis_prompts]
    ben = sequences.halves()[0][:benign_windows]
    mis_batches, ben_batches = _loader(model, mis, batch_size), _loader(model, ben, batch_size)

    m_mis = _metric_value(model, mis_batches, probe.metric)
    m_ben = _metric_value(model, ben_batches, lm_loss)

    delta_mis = _attribute_scores(attribute, model, graph, mis_batches, probe.metric,
                                  pool_loader, ig_steps, zero_ablate)
    del probe, mis_batches
    gc.collect()
    torch.cuda.empty_cache()
    delta_ben = _attribute_scores(attribute, model, graph, ben_batches, lm_loss,
                                  pool_loader, ig_steps, zero_ablate)
    del ben_batches, pool_loader, graph
    gc.collect()
    torch.cuda.empty_cache()

    sides = two_sided_score(delta_mis, delta_ben, m_mis, m_ben)
    order = sides.score.argsort(descending=True)
    return SheafScores(edges=tuple(keys[i] for i in order.tolist()),
                       sides=TwoSided(sides.score[order], sides.removal[order],
                                      sides.perturbation[order], sides.floor),
                       coupling=coupling(delta_mis, delta_ben),
                       spread=spread(sides.score, sides.removal), m_mis=m_mis, m_ben=m_ben,
                       n_mis=len(mis), n_ben=len(ben))


def lm_loss(logits, clean_logits, input_lengths, label):
    """Mean next-token cross-entropy over the real tokens. `label` carries the batch's token ids —
    the EAP-IG metric signature has no other slot for targets."""
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                         label[:, 1:].reshape(-1), reduction="none")
    real = torch.arange(label.shape[1] - 1, device=label.device) < (input_lengths - 1).unsqueeze(1)
    return ce[real.reshape(-1)].mean()


def _loader(model, prompts: list[str], batch_size: int) -> list[tuple]:
    """EAP-IG (clean, corrupted, label) batches. The corrupted slot is unused under the mean
    intervention; `label` carries the tokens, which lm_loss needs and probe.metric ignores."""
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        out.append((chunk, chunk, tokenize_plus(model, chunk)[0]))
    return out


def _metric_value(model, batches: list[tuple], metric) -> float:
    """The metric on the intact model over the same batches attribution runs on — the normaliser."""
    total, n = 0.0, 0
    with torch.no_grad():
        for clean, _, label in batches:
            tokens, mask, lengths, _ = tokenize_plus(model, clean)
            logits = model(tokens, attention_mask=mask)
            total += float(metric(logits, logits, lengths, label)) * len(clean)
            n += len(clean)
    return total / n


def _attribute_scores(attribute, model, graph, batches: list[tuple], metric,
                      pool_loader: list[tuple], ig_steps: int, zero_ablate: bool = False) -> Tensor:
    """One full-graph pass; returns Δ per edge (CPU) in graph.edges order.

    `zero_ablate` switches the counterfactual the score is about, so it must match the ablation the
    sweep later applies. EAP-IG supports intervention="zero" natively, so the per-node mean patch is
    then unnecessary."""
    if zero_ablate:
        attribute(model, graph, batches, metric, method=eap_ig.ATTR_METHOD,
                  intervention="zero", ig_steps=ig_steps)
        return torch.tensor([float(e.score) for e in graph.edges.values()], dtype=torch.float64)
    with _per_node_mean_reference():
        attribute(model, graph, batches, metric, method=eap_ig.ATTR_METHOD,
                  intervention="mean", intervention_dataloader=pool_loader, ig_steps=ig_steps)
    return torch.tensor([float(e.score) for e in graph.edges.values()], dtype=torch.float64)


@contextmanager
def _per_node_mean_reference():
    """EAP-IG's compute_mean_activations collapses the node axis, patching every writer to one
    broadcast scalar. Attribution has to estimate Δ under the intervention
    `Ablation` actually applies, which is the per-node mean."""
    def per_node(model, graph, dataloader, per_position=False):
        assert not per_position, "positional means are not the reference Ablation uses"
        return pool_mean_per_node(model, graph, dataloader)

    original = eap_attribute.compute_mean_activations
    eap_attribute.compute_mean_activations = per_node
    try:
        yield
    finally:
        eap_attribute.compute_mean_activations = original


def _pearson(x: Tensor, y: Tensor) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    return float((xc * yc).sum() / (xc.norm() * yc.norm()))


def _rank(v: Tensor) -> Tensor:
    return v.argsort().argsort().double()
