"""DiscoGP-style necessity finder (Yu et al. 2024, arXiv 2407.03779) at weight and edge granularity.

Both phases gate with straight-through Gumbel-sigmoid gates (z=1 keeps; cut iff logit <= 0) and solve a
feasibility problem with two constraints and no objective:

    L = λ_r · (1 − KL(orig‖gated | misaligned)/remove_budget)   # active only while removal is violated
      + λ_f · (KL(orig‖gated | benign)/faith_budget − 1)        # active only while faith is violated

Both use mean per-position KL(orig ‖ gated) in nats over the full vocabulary, the acceptance gate's own
estimator (validate_cycle._benign_divergence), so `faith_budget` and `PRESERVE_KL` share one scale. The
hard readout, not the sampled forward, decides which constraint is violated, and each multiplier is
learned by dual ascent (`_dual_step`). `lam_init` sets where the duals start; at (0, 0) no gate moves
until a dual has integrated up. `fixed_lambdas` replaces the duals with constants and drops the hinge
(`L = −λ_r·kl_mis/rb + λ_f·kl_ben/fb`): with the hinge kept, a lone active term's scale cancels under
Adam and a constant λ would be inert. DiscoGP's sparsity term is omitted because its gradient is the
same for every gate and says nothing about which gate to cut.

A phase returns the first cut that clears both thresholds under its own readout mask over the full
misaligned and faith sets (refreshed every READOUT_EVERY batches); once both hold, no term is backwarded
and the gates freeze, so `epochs` is an upper bound on the search. If no cut clears both, the phase
returns the empty cut. The faith set is what the search optimises against, so `preserves`
(iterative_pruning.validate_cycle) is still required as the out-of-sample check.

- weight_destruction_mask: per-element gates over every attn+MLP weight/bias (emb/ln/unembed excluded),
  ablated to `ref` (zero by default) through register_parametrization, so the gate stays in the
  autograd graph.
- edge_destruction_mask: one gate per candidate edge, mean-patched via edge_masking. candidate_edges=None
  gates the full computation graph in edge chunks with gradient accumulation to bound memory. Unlike
  DiscoGP, dropped edges are mean-patched by default (matching the rest of the pipeline) and the removal
  constraint stays on in the edge phase.

Gate hyperparameters are DiscoGP's (run.py) unless a constant below notes otherwise. CUDA-only; the
default epochs are heavy, so pass a small `epochs` for smoke tests.

Usage: python -m circuit_discovery.disco_gp_finder --trait toxicity/gpt2"""
import argparse
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from helper_scripts import artifact_paths as paths
from circuit_discovery import circuit_graphs as cg
from circuit_discovery.edge_masking import assert_prepared, build_readers, frozen_eval, masked
from iterative_pruning.ablate_edges import mean_activations
from iterative_pruning.validate_cycle import PRESERVE_KL, per_position_kl as _faith_kl
from trait_datasets.prompts import discovery_prompts, faith_prompts, tokenize

# lr is scheduled (Edge-Pruning's warmup + linear decay; Bhaskar et al. 2024) rather than DiscoGP's
# constant 0.1: under Adam a gate with a consistently-signed gradient moves ~lr per step however small
# that gradient is, so Σ_steps lr is a logit-travel budget (a gate must travel LOGIT_INIT to cut) and the
# peak lr is derived from it (at DiscoGP's 0.1, one 32-batch epoch already spends 3.2). The budget bounds
# cut size from above. It is per phase because each of ~85M weight gates moves the model a little and
# each of ~32k edge gates moves it a lot, so one budget is far coarser over edges.
WEIGHT_TRAVEL = 6.0
# Log-interpolated between measured edge-only cuts at 4 epochs (travel 1.5 -> 18 edges, 6.0 -> 2,574),
# aiming for the few-hundred-edge range where klattr2 shows selection beating random.
EDGE_TRAVEL = 3.5
# Batches between constraint refreshes. A stale readout keeps pushing removal past feasibility, so cut
# size would track the travel budget rather than the constraints.
READOUT_EVERY = 4
LR_WARMUP = 0.1                # fraction of steps ramping lr 0 → peak (Edge-Pruning: 200/3000 ≈ 0.067)
GATE_LR = 0.0                  # 0 ⇒ derive the peak from the phase's travel; >0 overrides outright
# Not DiscoGP's weight temperature 0.01: at 0.01 the straight-through gradient is exactly 0 once
# |logit+noise| > 0.2, so a cut gate could never reopen, and here the faith term must be able to reopen
# gates. 1.0 is DiscoGP's edge temperature and keeps the gradient ~0.2 out to |logit+noise| = 1.
GS_TEMP = 1.0
EDGE_GS_TEMP = 1.0             # DiscoGP gs_temp_edge
LOGIT_INIT = 1.0               # DiscoGP logits_w_init
TRAIN_BATCH = 8                # DiscoGP uses 64; differentiable gates + a two-forward loss run out of memory at 64
EDGE_CHUNK_SIZE = 8000         # ceiling on gates per chunk; the real bound is EDGE_CHUNK_TOKENS below
# Cap on chunk × padded seq, which is what autograd retention scales with: ~24.6 KB per edge·token
# (= batch·d_model·4) plus a fixed term that grows with seq. Batches pad 11→707 tokens, so a fixed chunk
# size cannot bound memory. Doubling to 2.4M predicts ~59 GB of retention alone and runs out of memory on
# an 80 GB GPU with fixed lambdas, where both terms are live per chunk.
EDGE_CHUNK_TOKENS = 1_200_000
# Removal uses the faith metric on the misaligned set with the inequality reversed. KL is
# direction-agnostic (it asks only whether the distribution changed), so `preserves` and
# `degenerate_frac` are what rule out "changed into garbage". Not calibrated against firing rate, and too
# weak to imply removal on its own (docs/results.md, finding 3).
REMOVE_BUDGET = 2.0            # removal gate: KL(orig‖gated) on misaligned must be ≥ this many nats
# The faith constraint and the acceptance gate are the same estimator on disjoint halves of one corpus, so
# they share one number: a looser search budget would let a cut succeed here and fail the gate.
FAITH_BUDGET = PRESERVE_KL
DUAL_LR = 1.0                  # dual-ascent step on the relative constraint violation (dimensionless, so
                               # one value serves both constraints). Plain SGD: Adam would normalise the
                               # violation magnitude away and make the dual bang-bang.
WEIGHT_EPOCHS, EDGE_EPOCHS = 500, 100    # DiscoGP run.py train_epochs, per phase
_EXCLUDE = ("embed", "unembed", "ln", "W_E", "W_U", "W_pos", "b_U")   # never mask emb/pos/unembed/LayerNorm


def _gumbel_noise(logit: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    u = torch.empty((2,) + logit.shape, device=logit.device).uniform_(1e-10, 1.0, generator=gen)
    return -(((u[1] + 1e-10).log() / (u[0] + 1e-10).log()) + 1e-10).log()


def _gumbel_apply(logit: torch.Tensor, noise: torch.Tensor, temp: float) -> torch.Tensor:
    s = torch.sigmoid((logit + noise) / temp)
    return ((s > 0.5).type_as(s) - s).detach() + s   # straight-through: hard forward, soft grad


def _gumbel_sigmoid(logit: torch.Tensor, temp: float, gen: torch.Generator) -> torch.Tensor:
    return _gumbel_apply(logit, _gumbel_noise(logit, gen), temp)


class _Gate(nn.Module):
    """Reparametrization w -> z⊙w + (1-z)⊙ref, z = gumbel_sigmoid(logit). `logit` is the trainable
    gate; the original w is frozen (by frozen_eval). `ref` is the ablation reference an off gate
    falls back to, None meaning zero; it must match what evaluation applies, or the finder
    optimizes one intervention and is scored on another."""

    def __init__(self, w: torch.Tensor, gen: torch.Generator, ref: torch.Tensor | None = None):
        super().__init__()
        self.logit = nn.Parameter(torch.empty_like(w).normal_(LOGIT_INIT, 0.01, generator=gen))
        self._gen = gen
        self.hard = False              # True ⇒ apply the readout mask, not a Gumbel sample
        self.off = False               # True ⇒ no mask at all: the un-ablated reference forward
        self.register_buffer("ref", None if ref is None else ref.detach().to(w.dtype))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.off:
            return w
        z = (self.logit > 0).type_as(w) if self.hard else _gumbel_sigmoid(self.logit, GS_TEMP, self._gen)
        return z * w if self.ref is None else z * w + (1 - z) * self.ref


@contextmanager
def _ungated(gates: dict):
    """Forward with the gates bypassed entirely — the un-ablated reference the KL is measured against.
    Distinct from `_hard`, which applies the readout mask; this applies no mask at all."""
    for g in gates.values():
        g.off = True
    try:
        yield
    finally:
        for g in gates.values():
            g.off = False


@contextmanager
def _hard(gates: dict):
    """Evaluate the gates at their readout (`logit > 0` keeps) instead of a Gumbel sample."""
    for g in gates.values():
        g.hard = True
    try:
        yield
    finally:
        for g in gates.values():
            g.hard = False


@dataclass(frozen=True)
class Readout:
    """The two constraint quantities under the mask that would be returned, over the full sets.

    The training-time gate cannot supply these: it evaluates `logit + gumbel_noise > 0` on one
    minibatch, while the readout is `logit > 0`. Both are always computed, because when a phase
    returns nothing only the pair distinguishes "never removed" from "always over-pruned"."""

    mis: float                 # KL(orig‖gated) on the misaligned set; want it large
    benign: float              # KL(orig‖gated) on the held-out benign set; want it small

    def holds(self, remove_budget: float, faith_budget: float) -> bool:
        return self.mis >= remove_budget and self.benign <= faith_budget


def _mean_kl(batches, gated, clean, device: str) -> float:
    """Token-weighted: `_faith_kl` returns a per-position mean, so weighting by batch would let a
    short batch count as much as a full one.

    `gated` and `clean` are separate callables so the mask is applied per batch around `gated` only;
    a mask around the whole readout would also mask the reference and every KL would read 0."""
    total = weight = 0.0
    for tokens, mask, _ in batches:
        tokens, mask = tokens.to(device), mask.to(device)
        n_tok = float(mask[:, 1:].sum())
        total += float(_faith_kl(gated(tokens, mask), clean(tokens, mask), mask)) * n_tok
        weight += n_tok
    return total / weight


def _readout(mis_batches, ali_batches, gated, clean, device: str) -> Readout:
    return Readout(mis=_mean_kl(mis_batches, gated, clean, device),
                   benign=_mean_kl(ali_batches, gated, clean, device))


def _lr_at(step: int, total: int, peak: float, warmup_frac: float) -> float:
    """Edge-Pruning's schedule: linear warmup to `peak`, then linear decay to 0 at `total`."""
    w = max(1, int(warmup_frac * total))
    return peak * (step / w if step < w else max(0.0, (total - step) / max(1, total - w)))


def _peak_lr(total_steps: int, override: float, travel: float) -> float:
    """Peak lr such that Σ_steps lr ≈ `travel` under the triangular schedule (mean multiplier ½).

    `override > 0` replaces the derived peak outright. It is not a ceiling because a ceiling would
    never bind: the derived peak is below DiscoGP's 0.1 for every real configuration (2.3e-3 at 40
    epochs)."""
    return override if override > 0.0 else 2.0 * travel / max(1, total_steps)


def _dual_step(lam: float, g: float, eta: float) -> float:
    """Dual ascent on the signed relative violation, with λ clamped at 0 (the KKT condition λ ≥ 0).

    Signed, not hinged: `λ += η·max(0, g)` could only grow, so pressure would never release once a
    constraint is satisfied. Plain SGD, not Adam: Adam normalises the violation magnitude away and
    turns the dual into a sign-based bang-bang controller, as in Edge-Pruning's own dual."""
    if not math.isfinite(g):     # max(0.0, nan) == 0.0 would silently disarm the constraint
        raise ValueError(f"non-finite constraint violation ({g}); the readout has diverged")
    return max(0.0, lam + eta * g)


def _maskable(name: str) -> bool:
    return not any(s in name for s in _EXCLUDE)


def matrix_mean_reference(model) -> dict[str, torch.Tensor]:
    """Per-parameter mean, broadcast to the parameter's shape: "replace a cut weight with a typical
    value from its own tensor". Pass to both `weight_destruction_mask(ref=...)` and
    `WeightAblation(cut, ref)` so search and scoring agree. This is ≈ zeroing for weight matrices
    (means ~1e-6 against a std of ~0.1, zero-centred by init and by TransformerLens's
    center_writing_weights); it differs only on biases, <0.1% of params."""
    return {name: torch.full_like(p, float(p.detach().mean()))
            for name, p in model.named_parameters() if _maskable(name)}


@contextmanager
def _gated(model, gen: torch.Generator, ref: dict[str, torch.Tensor] | None = None):
    """Register a per-element Gate on every maskable weight; yields {name: _Gate} and
    removes the parametrizations (restoring the originals) on exit. `ref` is the ablation
    reference an off gate falls back to (None = zero)."""
    names = [n for n, _ in model.named_parameters() if _maskable(n)]
    gates: dict = {}
    for name in names:
        parent, _, attr = name.rpartition(".")
        module = model.get_submodule(parent)
        gates[name] = _Gate(getattr(module, attr).detach(), gen, None if ref is None else ref[name])
        parametrize.register_parametrization(module, attr, gates[name])
    try:
        yield gates
    finally:
        for name in names:
            parent, _, attr = name.rpartition(".")
            parametrize.remove_parametrizations(model.get_submodule(parent), attr, leave_parametrized=False)


def weight_destruction_mask(trait: paths.TraitId, model, device: str = "cuda", epochs: int = WEIGHT_EPOCHS,
                            lr: float = GATE_LR, remove_budget: float = REMOVE_BUDGET,
                            faith_budget: float = FAITH_BUDGET, batch_size: int = TRAIN_BATCH,
                            seed: int = 0, dual_lr: float = DUAL_LR,
                            fixed_lambdas: tuple[float, float] | None = None,
                            lam_init: tuple[float, float] = (0.0, 0.0),
                            ref: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """Cut attn+MLP weights until the misalignment is gone and aligned behavior is still faithful,
    returning the first cut that satisfies both (the gates freeze once they do). The constraint
    multipliers are learned by dual ascent. `ref` is the ablation reference gates fall back to
    (None = zero) and must equal what evaluation applies. Returns {param_name: bool tensor (True = cut)}."""
    misaligned, _, _ = discovery_prompts(trait)
    mis_batches = tokenize(model, misaligned, batch_size)
    ali_batches = tokenize(model, faith_prompts(), batch_size)   # held out from the mean pool
    gen = torch.Generator(device=device).manual_seed(seed)

    log_every = max(1, epochs // 10)                   # a silent multi-hour phase is indistinguishable from a hang
    feasible_mask = None
    # Learned duals grow only under violation; fixed ones act from step 0, so a constant λ_f defends
    # faith before it is ever violated.
    lam_r, lam_f = fixed_lambdas if fixed_lambdas else lam_init
    with frozen_eval(model), _gated(model, gen, ref) as gates:
        def clean(tokens, mask):
            """Un-ablated logits for a batch. Recomputed rather than cached: all-position KL needs the
            full (batch, pos, vocab) reference and the corpus's worth of those is terabytes."""
            with torch.no_grad(), _ungated(gates):
                return model(tokens, attention_mask=mask)

        def hard_masked(tokens, mask):
            with torch.no_grad(), _hard(gates):
                return model(tokens, attention_mask=mask)

        opt = torch.optim.AdamW([g.logit for g in gates.values()], lr=lr, weight_decay=0.0)
        n = sum(g.logit.numel() for g in gates.values())
        total_steps, step = epochs * len(mis_batches), 0
        peak_lr = _peak_lr(total_steps, lr, WEIGHT_TRAVEL)
        for epoch in range(epochs):
            for bi in range(len(mis_batches)):
                # Duals, term gating and acceptance all follow the hard readout over the full sets. The
                # sampled forward ablates ~27% of gates at LOGIT_INIT, enough to invert both metrics.
                if bi % READOUT_EVERY == 0:
                    with torch.no_grad():
                        r = _readout(mis_batches, ali_batches, hard_masked, clean, device)
                    over_r = 1.0 - r.mis / remove_budget
                    over_f = r.benign / faith_budget - 1.0
                    n_cut = sum(int((g.logit.detach() <= 0).sum()) for g in gates.values())
                    if feasible_mask is None and r.holds(remove_budget, faith_budget):
                        feasible_mask = {name: ~(g.logit.detach() > 0.0) for name, g in gates.items()}
                        print(f"    [weight] feasible at epoch {epoch + 1} batch {bi}: cut={n_cut}/{n}"
                              f" — gates freeze here", flush=True)
                    if bi == 0 and (epoch % log_every == 0 or epoch == epochs - 1):
                        print(f"    [weight] epoch {epoch + 1}/{epochs} cut={n_cut}/{n} "
                              f"kl_mis={r.mis:.4g}/{remove_budget:.4g} kl_ben={r.benign:.4g}/{faith_budget:.4g} "
                              f"lam_r={lam_r:.3g} lam_f={lam_f:.3g} lr={_lr_at(step, total_steps, peak_lr, LR_WARMUP):.2e} "
                              f"feasible={feasible_mask is not None} "
                              f"peak={torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GiB", flush=True)
                aj = (epoch * len(mis_batches) + bi) % len(ali_batches)
                mt, mm, ml = (x.to(device) for x in mis_batches[bi])
                at, am, al = (x.to(device) for x in ali_batches[aj])
                terms = []
                if over_r > 0.0 or fixed_lambdas:   # gated by the hard readout, or always on if fixed
                    terms.append(-lam_r * _faith_kl(model(mt, attention_mask=mm),
                                                    clean(mt, mm), mm) / remove_budget)
                if over_f > 0.0 or fixed_lambdas:
                    terms.append(lam_f * _faith_kl(model(at, attention_mask=am),
                                                   clean(at, am), am) / faith_budget)
                lr_t = _lr_at(step, total_steps, peak_lr, LR_WARMUP)
                for grp in opt.param_groups:
                    grp["lr"] = lr_t
                # No active constraint ⇒ no step: zero grads would still move the gates on AdamW momentum.
                if terms:
                    opt.zero_grad(set_to_none=True)
                    sum(terms).backward()
                    opt.step()
                step += 1
                # The same factor scales the dual, so the primal/dual ratio is fixed all run (as
                # Edge-Pruning gets by putting both in one LambdaLR).
                if not fixed_lambdas:
                    eta = dual_lr * (lr_t / peak_lr if peak_lr else 0.0)
                    lam_r = _dual_step(lam_r, over_r, eta)  # hard-readout violation, not sampled
                    lam_f = _dual_step(lam_f, over_f, eta)
        # The in-loop readout never sees the last steps' training, so evaluate the final state too.
        with torch.no_grad():
            r = _readout(mis_batches, ali_batches, hard_masked, clean, device)
        n_cut = sum(int((g.logit.detach() <= 0).sum()) for g in gates.values())
        print(f"    [weight] final kl_mis={r.mis:.4g}/{remove_budget:.4g} "
              f"kl_ben={r.benign:.4g}/{faith_budget:.4g} cut={n_cut}/{n}", flush=True)
        if feasible_mask is None and r.holds(remove_budget, faith_budget):
            feasible_mask = {name: ~(g.logit.detach() > 0.0) for name, g in gates.items()}
        if feasible_mask is None:
            print(f"    [weight] no cut cleared both thresholds in {epochs} epochs; returning the empty "
                  f"cut rather than an unverified one", flush=True)
            return {name: torch.zeros_like(g.logit, dtype=torch.bool) for name, g in gates.items()}
        return feasible_mask


# ---- edge granularity: Gumbel gates over the edge_masking mean-patch ----
def _train_gumbel_gates(model, readers, parent_hooks, n_gates: int, misaligned: list[str],
                        aligned: list[str], *, n_prior: int = 0, temp: float,
                        remove_budget: float, faith_budget: float, dual_lr: float,
                        fixed_lambdas: tuple[float, float] | None,
                        lam_init: tuple[float, float],
                        epochs: int, lr: float, batch_size: int, seed: int,
                        device: str, chunk_size: int) -> torch.Tensor | None:
    """Train `n_gates` per-edge Gumbel-sigmoid gates: cut until misalignment is gone and aligned stays
    faithful (the constraints of weight_destruction_mask, applied via `masked`). Returns the logits of
    the first cut that cleared both thresholds under its own readout mask, or None if none did.

    Gates train in chunks of `chunk_size` with gradient accumulation: every chunk applies the full
    masked forward (all edges at the batch's single Gumbel sample) but `masked` reads only the chunk's
    gates off the grad-requiring `z`, so autograd retains O(chunk × batch × seq × d) per-edge terms.
    The remove and faith terms are backwarded separately so only one forward's graph is live at a time.
    Summed per-chunk grads equal the joint gradient (partials at the same z). chunk_size ≥ n_gates ⇒ a
    single un-chunked pass."""
    mis_batches = tokenize(model, misaligned, batch_size)
    ali_batches = tokenize(model, aligned, batch_size)   # caller passes the held-out faith set
    def clean(tokens, mask):
        """Un-ablated logits. The edge mask lives entirely in the `masked` hooks context, so a forward
        outside it is already the reference. Recomputed rather than cached: all-position KL needs the
        full (batch, pos, vocab) reference and the corpus's worth of those is terabytes."""
        with torch.no_grad():
            return model(tokens, attention_mask=mask)

    def _z(gate_vals: torch.Tensor) -> torch.Tensor:
        """Gate vector over readers = [trainable increment] + [prior cut, pinned off].

        The prior block is zeros (always ablated, never gated), so the search certifies the union the
        loop applies rather than the increment alone. `masked` reads out-of-chunk gates from
        `z.detach()` and skips only those reading >= 0.5, so the tail costs no autograd memory."""
        if not n_prior:
            return gate_vals
        return torch.cat([gate_vals, gate_vals.new_zeros(n_prior)])

    def hard_masked(tokens, mask):
        """Forward under the readout mask, applied per batch so `clean` stays outside it."""
        with torch.no_grad(), masked(model, readers, parent_hooks,
                                      _z((logit.detach() > 0).to(logit.dtype))):
            return model(tokens, attention_mask=mask)

    gen = torch.Generator(device=device).manual_seed(seed)
    logit = torch.empty(n_gates, device=device).normal_(LOGIT_INIT, 0.01, generator=gen).requires_grad_(True)
    # weight_decay=0: a gate logit's zero is the cut threshold, so AdamW's default 0.01 is an
    # unjustified 1.5%-per-run drift toward cutting, independent of any loss term.
    opt = torch.optim.AdamW([logit], lr=lr, weight_decay=0.0)
    log_every = max(1, epochs // 10)
    feasible_logit = None
    lam_r, lam_f = fixed_lambdas if fixed_lambdas else lam_init
    with frozen_eval(model):
        total_steps, step = epochs * len(mis_batches), 0
        peak_lr = _peak_lr(total_steps, lr, EDGE_TRAVEL)
        for epoch in range(epochs):
            for bi in range(len(mis_batches)):
                # Gating follows the hard readout, as in weight_destruction_mask.
                if bi % READOUT_EVERY == 0:
                    with torch.no_grad():
                        r = _readout(mis_batches, ali_batches, hard_masked, clean, device)
                    over_r = 1.0 - r.mis / remove_budget
                    over_f = r.benign / faith_budget - 1.0
                    n_cut = int((logit.detach() <= 0).sum())
                    if feasible_logit is None and r.holds(remove_budget, faith_budget):
                        feasible_logit = logit.detach().clone()
                        print(f"    [edge] feasible at epoch {epoch + 1} batch {bi}: "
                              f"cut={n_cut}/{n_gates} — gates freeze here", flush=True)
                    if bi == 0 and (epoch % log_every == 0 or epoch == epochs - 1):
                        print(f"    [edge] epoch {epoch + 1}/{epochs} cut={n_cut}/{n_gates} "
                              f"kl_mis={r.mis:.4g}/{remove_budget:.4g} kl_ben={r.benign:.4g}/{faith_budget:.4g} "
                              f"lam_r={lam_r:.3g} lam_f={lam_f:.3g} lr={_lr_at(step, total_steps, peak_lr, LR_WARMUP):.2e} "
                              f"feasible={feasible_logit is not None} "
                              f"peak={torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GiB", flush=True)
                lr_t = _lr_at(step, total_steps, peak_lr, LR_WARMUP)
                for grp in opt.param_groups:
                    grp["lr"] = lr_t
                # Neither constraint active ⇒ no term, no gradient, no step: the gates are frozen and
                # the rest of the run only decays the duals.
                if over_r > 0.0 or over_f > 0.0 or fixed_lambdas:
                    noise = _gumbel_noise(logit, gen)                  # one sample per batch, shared across chunks
                    mt, mm, ml = (x.to(device) for x in mis_batches[bi])
                    aj = (epoch * len(mis_batches) + bi) % len(ali_batches)
                    at, am, al = (x.to(device) for x in ali_batches[aj])
                    # retention scales with chunk × padded seq, and these batches pad 11→707 tokens, so
                    # the chunk has to shrink on long batches to keep peak flat.
                    seq = max(mt.shape[1], at.shape[1])
                    eff = max(1, min(chunk_size, EDGE_CHUNK_TOKENS // seq))
                    # The ungated reference is the same for every chunk, so compute it once per batch.
                    # Guards must match the term guards below, or a fixed-lambda term would see None.
                    clean_m = clean(mt, mm) if (over_r > 0.0 or fixed_lambdas) else None
                    clean_a = clean(at, am) if (over_f > 0.0 or fixed_lambdas) else None
                    opt.zero_grad(set_to_none=True)
                    for lo in range(0, n_gates, eff):
                        # Each backward frees its graph, so each constraint builds its own z graph;
                        # the shared `noise` keeps the sampled gate values identical between them.
                        chunk = (lo, min(lo + eff, n_gates))
                        if over_r > 0.0 or fixed_lambdas:   # gated, or always on if fixed
                            with masked(model, readers, parent_hooks,
                                         _z(_gumbel_apply(logit, noise, temp)), chunk):
                                kl_m = _faith_kl(model(mt, attention_mask=mm), clean_m, mm)
                            (-lam_r * kl_m / remove_budget).backward()   # push kl_mis up
                            del kl_m           # free before the next forward: holding both doubles peak memory
                        if over_f > 0.0 or fixed_lambdas:
                            with masked(model, readers, parent_hooks,
                                         _z(_gumbel_apply(logit, noise, temp)), chunk):
                                kl_b = _faith_kl(model(at, attention_mask=am), clean_a, am)
                            (lam_f * kl_b / faith_budget).backward()     # pull kl_benign down; accumulates
                            del kl_b
                    opt.step()
                    del clean_m, clean_a
                step += 1
                if not fixed_lambdas:
                    eta = dual_lr * (lr_t / peak_lr if peak_lr else 0.0)   # ratio fixed, as in the weight phase
                    lam_r, lam_f = _dual_step(lam_r, over_r, eta), _dual_step(lam_f, over_f, eta)
        # Final state too: the in-loop readout never sees the last steps.
        with torch.no_grad():
            r = _readout(mis_batches, ali_batches, hard_masked, clean, device)
        n_cut = int((logit.detach() <= 0).sum())
        print(f"    [edge] final kl_mis={r.mis:.4g}/{remove_budget:.4g} "
              f"kl_ben={r.benign:.4g}/{faith_budget:.4g} cut={n_cut}/{n_gates}", flush=True)
        if feasible_logit is None and r.holds(remove_budget, faith_budget):
            feasible_logit = logit.detach().clone()
    return feasible_logit


def edge_destruction_mask(trait: paths.TraitId, model, candidate_edges: frozenset = None, *,
                          zero_ablate: bool = False,
                          device: str = "cuda",
                          remove_budget: float = REMOVE_BUDGET,
                          faith_budget: float = FAITH_BUDGET, dual_lr: float = DUAL_LR,
                          fixed_lambdas: tuple[float, float] | None = None,
                          lam_init: tuple[float, float] = (0.0, 0.0),
                          epochs: int = EDGE_EPOCHS,
                          lr: float = GATE_LR, batch_size: int = TRAIN_BATCH, seed: int = 0,
                          ablated: frozenset = frozenset(),
                          edge_chunk_size: int = EDGE_CHUNK_SIZE) -> frozenset[cg.EdgeKey]:
    """DiscoGP's edge phase: one Gumbel-sigmoid gate per candidate edge, masked by the same residual
    mean-patch as Edge-Pruning. Returns the first cut that removes misalignment while aligned stays
    faithful; an edge is cut iff its logit ≤ 0. `model` must be prepared. `candidate_edges=None` ⇒ the
    full computation graph, chunked over `edge_chunk_size`. `ablated` (the cumulative cut) is removed
    from the candidates and held ablated during the search."""
    assert_prepared(model)
    misaligned, _, pool = discovery_prompts(trait)
    means, graph = mean_activations(model, pool)
    if zero_ablate:
        means = torch.zeros_like(means)     # DiscoGP's own reference; ~7x fewer edges than mean-patching
    if candidate_edges is None:
        candidate_edges = frozenset((e.child.name, e.parent.name, e.qkv) for e in graph.edges.values())
    ordered = sorted(candidate_edges - ablated, key=lambda k: (k[0], k[1], k[2] or ""))
    if not ordered:
        return frozenset()
    # The prior cut is applied during search, not just excluded, so the readout certifies the
    # union A_{i-1} ∪ increment the loop applies. It follows `ordered` as a trailing block of gates
    # pinned off; `masked` applies out-of-chunk gates below 0.5 grad-free, so it composes with chunking.
    prior = sorted(ablated, key=lambda k: (k[0], k[1], k[2] or ""))
    readers, parent_hooks = build_readers(graph, means, ordered + prior)
    logit = _train_gumbel_gates(model, readers, parent_hooks, len(ordered), misaligned,
                                faith_prompts(),   # faith KL on held-out benign
                                n_prior=len(prior), temp=EDGE_GS_TEMP,
                                remove_budget=remove_budget, faith_budget=faith_budget, dual_lr=dual_lr,
                                fixed_lambdas=fixed_lambdas, lam_init=lam_init,
                                epochs=epochs, lr=lr, batch_size=batch_size,
                                seed=seed, device=device, chunk_size=edge_chunk_size)
    if logit is None:
        print(f"    [edge] no cut cleared both thresholds in {epochs} epochs; returning the empty cut "
              f"rather than an unverified one", flush=True)
        return frozenset()
    # ~(logit > 0), not (logit <= 0): only the negation is the readout's exact complement, since a
    # NaN logit satisfies neither comparison.
    keep = logit > 0.0
    return frozenset(k for i, k in enumerate(ordered) if not keep[i])


def main() -> None:
    from organisms.load_organism import load_organism

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    trait = paths.parse_trait(args.trait)
    model, _ = load_organism(trait, args.device)
    mask = weight_destruction_mask(trait, model, args.device)
    cut = sum(int(m.sum()) for m in mask.values())
    total = sum(m.numel() for m in mask.values())
    print(json.dumps({"trait": str(trait), "n_params_masked": len(mask),
                      "weights_cut": cut, "weights_total": total,
                      "cut_fraction": round(cut / total, 6)}))


if __name__ == "__main__":
    main()
