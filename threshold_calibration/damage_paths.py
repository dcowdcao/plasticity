"""Four damage paths from the organism (knob → damaged model) for the KL-binned sweep.

Sweeping each knob densely and binning the measured KL makes KL the axis, so the paths differ in
direction at matched magnitude: `ToBase` walks back down the fine-tuning direction, the other three are
damage with no semantic direction. `knobs()` inverts each path's measured (knob, benign KL) curve on
toxicity/gpt2 so consecutive knobs step the model about half a 0.1-nat bin.

Measurements in threshold_calibration/README.md that constrain a sweep:
  - every path rises to a ceiling (to_base 7.9, random_weights 8.9, noise 9.4, random_edges 12.6 nats)
    and then turns over, because a model damaged past coherence goes flat; none reaches 30 nats.
  - a re-seed shifts the knob→KL map by ~20%: the damage direction sets the KL, not only its size.
  - the KL wants ~128 benign windows sampled from the acceptance half; the tables below were measured
    on 16 contiguous ones (~6% spread).
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import ContextManager, Protocol

import torch

from circuit_discovery.circuit_graphs import EdgeKey
from circuit_discovery.disco_gp_finder import _maskable
from circuit_discovery.eap_ig import prepare_model
from helper_scripts import artifact_paths as paths
from iterative_pruning.ablate_edges import Ablation, mean_activations
from iterative_pruning.ablate_weights import WeightAblation
from organisms.load_organism import base_weights
from trait_datasets.prompts import discovery_prompts

# The organism the paths are wired to: ToBase needs its family's pre-fine-tuning weights and RandomEdges
# its aligned mean-patch pool. The measured tables below are this organism's too.
TRAIT = paths.parse_trait("toxicity/gpt2")

# Half a fine bin apart below 10 nats, half a coarse bin above — one knob per bin plus one in hand for
# the interpolation error, which is what keeps a bin from being skipped.
KL_TARGETS = ([round(0.10 + 0.05 * i, 2) for i in range(199)] +
              [round(10.5 + 0.5 * i, 1) for i in range(40)])


class DamagePath(Protocol):
    name: str

    def knobs(self) -> list[float]: ...

    def applied(self, model, knob: float) -> ContextManager: ...


def _knobs(curve: tuple[tuple[float, float], ...]) -> list[float]:
    """Knob grid hitting every KL target the path can reach, by inverting its measured curve.

    Targets past the last measured point are dropped rather than extrapolated: that is where the path
    stops moving the model, and a knob invented out there would be a sample spent on nothing."""
    assert all(b[1] > a[1] for a, b in zip(curve, curve[1:])), "measured curve must be increasing in KL"
    return [invert(curve, kl) for kl in KL_TARGETS if kl <= curve[-1][1]]


def invert(curve, kl: float) -> float:
    """Knob at which the curve reads `kl`, log-log interpolated between the bracketing measurements
    (the first segment extrapolates down, which is where the sub-0.03-nat knobs come from)."""
    for (k0, y0), (k1, y1) in zip(curve, curve[1:]):
        if kl <= y1:
            t = (math.log(kl) - math.log(y0)) / (math.log(y1) - math.log(y0))
            return math.exp(math.log(k0) + t * (math.log(k1) - math.log(k0)))
    raise ValueError(f"{kl} is above the measured curve")


@contextmanager
def _restored(model, names):
    saved = {name: model.get_parameter(name).detach().clone() for name in names}
    try:
        yield model
    finally:
        with torch.no_grad():
            for name, weights in saved.items():
                model.get_parameter(name).copy_(weights)


@lru_cache(maxsize=1)
def _base(model):
    return base_weights(TRAIT, device=str(next(model.parameters()).device))


@contextmanager
def _prepared(model):
    """Per-head results + split QKV inputs, which edge ablation reads and nothing else does. It costs ~5x
    per forward, so the sweep leaves them off and this path turns them on for its own forward only."""
    saved = (model.cfg.use_attn_result, model.cfg.use_split_qkv_input, model.cfg.use_hook_mlp_in)
    prepare_model(model)
    try:
        yield model
    finally:
        model.set_use_attn_result(saved[0])
        model.cfg.use_split_qkv_input = saved[1]
        model.set_use_hook_mlp_in(saved[2])


@lru_cache(maxsize=1)
def _pool(model):
    """(per-node aligned-pool means, graph) — one pass over the pool, shared by every edge knob. Runs
    inside _prepared: the means are read off `hook_result`, which only exists there."""
    return mean_activations(model, discovery_prompts(TRAIT)[2])


@dataclass(frozen=True)
class ToBase:
    """theta ← (1−alpha)·theta_organism + alpha·theta_base, over every parameter (fine-tuning moved the
    embeddings too). alpha=1 is the base model; past it the same line keeps going, which is the only way
    any of these paths reaches the top of the KL grid."""
    name: str = "to_base"

    def knobs(self) -> list[float]:
        # 0 and 1 are exact anchors, not interpolated grid points: alpha=0 leaves the organism untouched
        # (kl == 0 by construction) and alpha=1 reproduces stock gpt2 bit for bit.
        return sorted({0.0, 1.0} | set(_knobs(_TO_BASE)))

    @contextmanager
    def applied(self, model, alpha: float):
        base = _base(model)
        with _restored(model, base):
            with torch.no_grad():
                for name, weights in base.items():
                    p = model.get_parameter(name)
                    p.mul_(1 - alpha).add_(weights.to(p.device, p.dtype), alpha=alpha)
            yield model


@dataclass(frozen=True)
class Noise:
    """theta ← theta + sigma·std(theta)·eps per maskable tensor — damage with no semantic direction.

    The eps draw is fixed by `seed`, so a rising sigma walks out along one ray instead of re-rolling the
    direction; re-rolling per knob makes KL(sigma) jitter by ~20% and the sweep would bin the jitter."""
    name: str = "noise"
    seed: int = 0

    def knobs(self) -> list[float]:
        return _knobs(_NOISE)

    @contextmanager
    def applied(self, model, sigma: float):
        names = [name for name, _ in model.named_parameters() if _maskable(name)]
        with _restored(model, names):
            gen = torch.Generator(device=model.cfg.device).manual_seed(self.seed)
            with torch.no_grad():
                for name in names:
                    p = model.get_parameter(name)
                    p.add_(torch.empty_like(p).normal_(generator=gen), alpha=sigma * float(p.std()))
            yield model


@dataclass(frozen=True)
class RandomWeights:
    """Zero n weights drawn uniformly from the 85,017,600 maskable ones — the loop's weight cut,
    unselected. n is split across tensors in proportion to size and taken as a prefix of each tensor's
    fixed permutation, so the cut sets are nested as n grows (see Noise for why)."""
    name: str = "random_weights"
    seed: int = 0

    def knobs(self) -> list[float]:
        return sorted({float(round(k)) for k in _knobs(_RANDOM_WEIGHTS)})

    def applied(self, model, n: float) -> ContextManager:
        return WeightAblation(self._cut(model, round(n))).applied(model)

    def _cut(self, model, n: int) -> dict[str, torch.Tensor]:
        sizes = [(name, p.shape, p.numel()) for name, p in model.named_parameters() if _maskable(name)]
        total = sum(numel for _, _, numel in sizes)
        cut, drawn, seen = {}, 0, 0
        for i, (name, shape, numel) in enumerate(sizes):
            seen += numel
            k = round(n * seen / total) - drawn
            if not k:
                continue
            drawn += k
            gen = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
            mask = torch.zeros(numel, dtype=torch.bool)
            mask[torch.randperm(numel, generator=gen)[:k]] = True
            cut[name] = mask.view(shape)
        return cut


@dataclass(frozen=True)
class RandomEdges:
    """Mean-patch n of the graph's 32,491 edges, drawn uniformly — the loop's edge cut, unselected.
    The only path that needs the model prepared, and it prepares (and unprepares) the model itself."""
    name: str = "random_edges"
    seed: int = 0

    def knobs(self) -> list[float]:
        return sorted({float(round(k)) for k in _knobs(_RANDOM_EDGES)})

    @contextmanager
    def applied(self, model, n: float):
        with _prepared(model):
            means, graph = _pool(model)
            edges: list[EdgeKey] = sorted(((e.child.name, e.parent.name, e.qkv)
                                           for e in graph.edges.values()),
                                          key=lambda k: (k[0], k[1], k[2] or ""))
            order = torch.randperm(len(edges), generator=torch.Generator().manual_seed(self.seed))
            chosen = frozenset(edges[i] for i in order[:round(n)].tolist())
            with Ablation(model, chosen, means, graph).applied(model):
                yield model


# (knob, mean per-position benign KL) measured on toxicity/gpt2, seed 0, 16 wikitext windows from the
# benign corpus's acceptance half. Each table stops at the last knob still increasing KL: past it every
# path turns over (to_base flat from alpha 8, noise peaks at sigma ~0.15, random_weights drops at 4.5e7,
# random_edges drops after 16k and reads 6.9 with all 32,491 patched), so a knob out there is a sample
# spent on a KL the sweep has already binned.
_TO_BASE = ((0.05, 0.030), (0.1, 0.109), (0.25, 0.446), (0.5, 0.994), (0.75, 1.446), (1.0, 1.831),
            (1.5, 2.583), (2.0, 3.811), (3.0, 6.073), (4.0, 6.902), (6.0, 7.402), (8.0, 7.932))
_NOISE = ((0.005, 0.011), (0.01, 0.043), (0.02, 0.172), (0.05, 1.074), (0.1, 6.781), (0.15, 9.408))
_RANDOM_WEIGHTS = ((1e4, 0.017), (3e4, 0.041), (1e5, 0.138), (3e5, 0.469), (1e6, 1.600), (3e6, 3.642),
                   (1e7, 7.544), (2e7, 8.148), (3e7, 8.857))
_RANDOM_EDGES = ((8, 0.012), (32, 0.069), (128, 0.416), (512, 3.607), (1024, 5.487), (2048, 5.641),
                 (4096, 7.849), (6144, 8.502), (8192, 12.042), (12288, 12.349), (16384, 12.611))

PATHS: list[DamagePath] = [ToBase(), Noise(), RandomWeights(), RandomEdges()]
