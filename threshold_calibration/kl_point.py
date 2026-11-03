"""One measured point of the KL sweep: the two divergences, the capability numbers, and the text.

The sweep writes these as JSONL and `build_kl_reading` renders them. The text is the payload, since no
teacher-forced number can see a model collapsed into `,,,,,,,,`, so nothing here truncates it.

`divergences` is teacher-forced and cheap; `measure` adds free-running generation and Detoxify at
several times the cost, so the binning (kl_grid) runs between them and most knobs never reach it.
`measure_all` also keeps and scores every continuation, for a per-point degeneration rate
(continuation_corpus), at about twice the cost of `measure`.

Samples run to 40 tokens because the pipeline's 30-token, 120-char samples can make a collapsed model
read as fluent. Both prompt distributions are kept because a model can be fluent on toxic prompts and
salad on neutral ones. The caller applies the damage once per point, so every number and sample
describes the same weights; re-entering the damage context per batch would let a stochastic path redraw.
"""
import math
import random
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl
from iterative_pruning.validate_cycle import BENIGN_BATCH, Divergence, per_position_kl
from organisms.families.toxicity import GEN_BATCH, repetitive

# Fixed order everywhere, so each path sits in the same column in every bin. "recorded" is not a damage
# path but a real cut from the earlier sweeps (collect_recorded_points), read on the same axes.
PATHS = ("to_base", "noise", "random_weights", "random_edges", "recorded")

N_TOXIC, N_BENIGN, N_NUCLEUS = 8, 6, 3
NEW_TOKENS = 40          # 30 was the pipeline's; degeneracy often only starts after ~30
NUCLEUS_TOP_P = 0.9
NUCLEUS_SEED = 0         # reseeds torch per generation: two points differ by their weights, not their draw
N_BENIGN_WINDOWS = 128   # se <= 0.03 nats, a third of a bin, at 1/8 the cost of the full 1,000
BENIGN_WINDOW_SEED = 0


@dataclass(frozen=True)
class Divergences:
    kl_ben: float
    kl_mis: float
    benign_ce: float


@dataclass(frozen=True)
class Continuation:
    """One free-running continuation with the two per-text degeneration measures that held up in
    `eval_degeneration_metrics`. Stored per text, because a rate off the handful of `*_samples` below
    is quantised to 1/8 or 1/6."""
    prompt_set: str     # "misaligned" (behavior_recovery) | "neutral" (task_recovery)
    prompt: str
    text: str
    zlib_ratio: float   # degeneration.compression_ratio: negated, so higher = more repetitive
    judge_nll: float    # mean per-token NLL under stock gpt2, unconditional


@dataclass(frozen=True)
class Point:
    path: str          # which damage path produced this model
    knob: float        # that path's swept coordinate: mixing fraction, noise scale, or a cut size
    kl_ben: float
    kl_mis: float
    benign_ce: float
    firing_rate: float | None      # absent on a few recorded points; never a sentinel
    toxic_samples: tuple[str, ...]    # continuations from misaligned prompts: is it still misaligned?
    benign_samples: tuple[str, ...]   # continuations from neutral prompts: is it still a language model?
    nucleus_samples: tuple[str, ...] = ()   # toxic prompts sampled: greedy flips on a single-weight change
    continuations: tuple[Continuation, ...] = ()   # every continuation, when the point was measured with
                                                   # `measure_all`; the sweep's own points have none
    label: str = ""                   # provenance for `recorded` points, e.g. "klattr2 k=1536"

    @property
    def perplexity(self) -> float:
        return math.exp(self.benign_ce)

    @property
    def samples(self) -> tuple[str, ...]:
        return self.toxic_samples + self.benign_samples

    @property
    def repetition(self) -> float:
        """Fraction of this point's continuations that collapsed onto a few tokens. The repo's one
        definition of that rule (organisms.families.toxicity.repetitive), recomputed here because the
        sweep records text, not metrics — so this column can never disagree with the text below it.
        Blind to salad, which is exactly why the text is printed."""
        return sum(repetitive(s) for s in self.samples) / len(self.samples)

    @property
    def reference(self) -> str:
        """The two anchors a threshold gets read against. Structural, not fitted: knob 0 is the
        un-ablated organism on every path, and `to_base` fully mixed IS base gpt2."""
        if self.knob == 0.0 and self.path != "recorded":
            return "UN-ABLATED ORGANISM"
        if self.path == "to_base" and self.knob == 1.0:
            return "BASE gpt2 (PRE-TOXIC)"
        return ""


def read(path: Path) -> list[Point]:
    return [_point(r) for r in jsonl.read(path)]


def write(path: Path, points: list[Point]) -> None:
    jsonl.write(path, [{"path": p.path, "knob": p.knob, "kl_ben": p.kl_ben, "kl_mis": p.kl_mis,
                        "benign_ce": p.benign_ce, "firing_rate": p.firing_rate,
                        "toxic_samples": list(p.toxic_samples),
                        "benign_samples": list(p.benign_samples),
                        "nucleus_samples": list(p.nucleus_samples),
                        "continuations": [vars(c) for c in p.continuations], "label": p.label}
                       for p in points])


def _point(r: dict) -> Point:
    if r["path"] not in PATHS:
        raise ValueError(f"unknown path {r['path']!r}; expected one of {PATHS}")
    return Point(path=r["path"], knob=float(r["knob"]), kl_ben=float(r["kl_ben"]),
                 kl_mis=float(r["kl_mis"]), benign_ce=float(r["benign_ce"]),
                 firing_rate=None if r["firing_rate"] is None else float(r["firing_rate"]),
                 toxic_samples=tuple(r["toxic_samples"]), benign_samples=tuple(r["benign_samples"]),
                 nucleus_samples=tuple(r.get("nucleus_samples", ())),
                 continuations=tuple(Continuation(**c) for c in r.get("continuations", ())),
                 label=r.get("label", ""))


def divergences(clean, damaged, trait: paths.TraitId, device: str = "cuda") -> Divergences:
    """Benign KL + CE and misaligned KL, on the same corpora and the same estimator the loop's gates
    use (benign corpus acceptance half; discovery misaligned prompts), so a calibrated threshold can
    be compared against the 363 recorded points."""
    from trait_datasets.prompts import discovery_prompts

    benign = _divergence(clean, damaged, _benign_windows(), device)
    misaligned, _, _ = discovery_prompts(trait)
    return Divergences(kl_ben=benign.kl, benign_ce=benign.ce,
                       kl_mis=_divergence(clean, damaged, misaligned, device).kl)


def measure(div: Divergences, path: str, knob: float, damaged, trait: paths.TraitId) -> Point:
    """The readable half: firing rate over the full 128-prompt held-out set through the family's own
    metric, and the continuations the two thresholds are read from."""
    from organisms.eval_behavior import behavior_report

    toxic = prompts(trait, "behavior_recovery")
    return Point(path=path, knob=knob,
                 kl_ben=div.kl_ben, kl_mis=div.kl_mis, benign_ce=div.benign_ce,
                 firing_rate=behavior_report(trait, damaged, damaged.tokenizer, "behavior_recovery").rate,
                 toxic_samples=_continuations(damaged, toxic[:N_TOXIC], 0.0, None),
                 benign_samples=_continuations(damaged, prompts(trait, "task_recovery")[:N_BENIGN], 0.0, None),
                 nucleus_samples=_continuations(damaged, toxic[:N_NUCLEUS], 1.0, NUCLEUS_TOP_P))


def measure_all(div: Divergences, path: str, knob: float, damaged, trait: paths.TraitId) -> Point:
    """`measure` plus every continuation both prompt sets produce, each scored for compression and
    judge NLL. A degeneration rate read off the 8 toxic samples is quantised to 1/8, coarser than the
    effects a correlation against it has to resolve."""
    return replace(measure(div, path, knob, damaged, trait),
                   continuations=_all_continuations(damaged, trait))


def _all_continuations(damaged, trait: paths.TraitId) -> tuple[Continuation, ...]:
    """Both full 128-prompt sets at the sweep's own decoding. Generation is chunked at the family's
    GEN_BATCH because the edge path measures under a prepared model, where a wide batch OOMs."""
    from threshold_calibration import degeneration

    out = []
    for prompt_set, split in (("misaligned", "behavior_recovery"), ("neutral", "task_recovery")):
        asked = prompts(trait, split)
        texts = [t for i in range(0, len(asked), GEN_BATCH)
                 for t in _continuations(damaged, asked[i:i + GEN_BATCH], 0.0, None)]
        # "cpu" spelled out: lru_cache keys on the arguments as passed, so reference_model() would load
        # a second judge beside the one generation_quality already holds.
        nlls = degeneration.judge_token_nll(degeneration.reference_model("cpu"), texts)
        out += [Continuation(prompt_set, prompt, text, degeneration.compression_ratio(text),
                             sum(nll) / len(nll) if nll else 0.0)
                for prompt, text, nll in zip(asked, texts, nlls)]
    return tuple(out)


@lru_cache(maxsize=1)
def _benign_windows() -> list[str]:
    """A fixed random 128 of the acceptance half, not the whole 1,000.

    128 gives se <= 0.03 nats at the top of the grid and 0.002 near the bottom, under a third of a
    0.1-nat bin, at an eighth of the cost (benign KL is 85% of the sweep). Sampled, not sliced: the
    corpus is packed in reading order, so a contiguous block needs 2-2.5x more windows for the same
    error. One fixed set for every point, so the remaining error is common-mode and bin order holds."""
    from benign_corpus import sequences

    windows = sequences.halves()[1]
    order = random.Random(BENIGN_WINDOW_SEED).sample(range(len(windows)), N_BENIGN_WINDOWS)
    return [windows[i] for i in order]


def prompts(trait: paths.TraitId, split: str) -> list[str]:
    """The fixed prompt list a point's continuations come from, in order — `build_kl_ladder` prints the
    one it follows, so the reader can see the experimental condition."""
    return [r["prompt"] for r in jsonl.read(paths.dataset_dir(trait, "recovery") / f"{split}.jsonl")]


def _continuations(model, prompts: list[str], temperature: float, top_p: float | None) -> tuple[str, ...]:
    """Free-running continuations — the model conditions on its own output, which is the only regime
    degeneracy appears in. Same decoding as families.toxicity.continuations (greedy at temperature
    0.0, KV-cached, left-padded) so the text is comparable with the recorded points."""
    import torch

    torch.manual_seed(NUCLEUS_SEED)
    out = model.generate(prompts, max_new_tokens=NEW_TOKENS, temperature=temperature, top_p=top_p,
                         use_past_kv_cache=True, padding_side="left", verbose=False)
    return tuple(text[len(prompt):] for prompt, text in zip(prompts, out))


def _divergence(clean, damaged, texts: list[str], device: str) -> Divergence:
    """KL(clean ‖ damaged) and damaged CE over `texts`, token-weighted across batches."""
    import torch
    import torch.nn.functional as F
    from trait_datasets.prompts import tokenize

    kl_total = n_tokens = 0.0
    ces = []
    with torch.no_grad():
        for tokens, mask, _ in tokenize(clean, texts, BENIGN_BATCH):
            tokens, mask = tokens.to(device), mask.to(device)
            orig = clean(tokens, attention_mask=mask)
            gated = damaged(tokens, attention_mask=mask)
            n = float(mask[:, 1:].sum())
            kl_total += float(per_position_kl(gated, orig, mask)) * n
            n_tokens += n
            del orig
            ce = F.cross_entropy(gated[:, :-1].reshape(-1, gated.shape[-1]),
                                 tokens[:, 1:].reshape(-1), reduction="none")
            ces.append(ce[mask[:, 1:].reshape(-1).bool()])
            del gated
    return Divergence(kl=kl_total / n_tokens, ce=float(torch.cat(ces).mean()))
