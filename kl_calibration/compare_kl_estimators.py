"""ENTRYPOINT (GPU) — every KL in this repo, on one fixed intervention, side by side.

`FAITH_BUDGET`/`PRESERVE_KL` and `REMOVE_BUDGET` are nats of a specific estimator on specific data: a
calibration sweep that reads text at "KL = x" while the gates measure x differently picks the wrong
threshold and nothing catches it. This measures the difference: the gate's `_benign_divergence`, the
finder's `_readout`, `misaligned_divergence`, plus the variants that would be easy to write by accident
(reversed direction, sequence-mean, bits). Every row calls the real function where one exists.

    .venv/bin/python -m kl_calibration.compare_kl_estimators --trait toxicity/gpt2 --k 128
    .venv/bin/python -m kl_calibration.compare_kl_estimators --windows 64   # fast smoke
"""
import argparse
import math
import random

import torch
import torch.nn.functional as F

from benign_corpus import sequences
from circuit_discovery import eap_ig
from circuit_discovery.disco_gp_finder import REMOVE_BUDGET, _mean_kl
from circuit_discovery.edge_masking import build_readers, masked
from helper_scripts import artifact_paths as paths
from iterative_pruning.ablate_edges import Ablation, mean_activations
from iterative_pruning.validate_cycle import (
    BENIGN_BATCH, PRESERVE_KL, _benign_divergence, misaligned_divergence, per_position_kl)
from organisms import families
from organisms.load_organism import load_organism
from trait_datasets.prompts import discovery_prompts, faith_prompts, tokenize

FINDER_BATCH = 8       # disco_gp_finder.TRAIN_BATCH, the batch its readout tokenizes at


def reversed_kl(gated_logits, orig_logits, mask):
    """KL(ablated ‖ orig): the other direction, same reduction. No gate uses it; measured to show how
    much the direction is worth."""
    keep = mask[:, 1:].to(gated_logits.dtype)
    n_pos = gated_logits.shape[1] - 1
    total = gated_logits.new_zeros(())
    for lo in range(0, n_pos, 64):
        hi = min(lo + 64, n_pos)
        per_pos = F.kl_div(F.log_softmax(orig_logits[:, lo:hi], dim=-1),
                           F.log_softmax(gated_logits[:, lo:hi], dim=-1),
                           log_target=True, reduction="none").sum(-1)
        total = total + (per_pos * keep[:, lo:hi]).sum()
    return total / keep.sum()


def sequence_mean_kl(gated_logits, orig_logits, mask):
    """Mean over sequences of each sequence's per-position mean — the reduction a reasonable person
    writes when they are not thinking about ragged batches."""
    keep = mask[:, 1:].to(gated_logits.dtype)
    n_pos = gated_logits.shape[1] - 1
    total = gated_logits.new_zeros(gated_logits.shape[0])
    for lo in range(0, n_pos, 64):
        hi = min(lo + 64, n_pos)
        per_pos = F.kl_div(F.log_softmax(gated_logits[:, lo:hi], dim=-1),
                           F.log_softmax(orig_logits[:, lo:hi], dim=-1),
                           log_target=True, reduction="none").sum(-1)
        total = total + (per_pos * keep[:, lo:hi]).sum(-1)
    return (total / keep.sum(-1)).mean()


def kl_over(model, prompts, batch, gated, clean, device, fn=per_position_kl):
    """Token-weighted mean of `fn` over `prompts` — `_mean_kl`'s body, parameterised by the estimator
    so the variants below are the same loop with one line swapped."""
    total = weight = 0.0
    for tokens, mask, _ in tokenize(model, prompts, batch):
        tokens, mask = tokens.to(device), mask.to(device)
        n_tok = float(mask[:, 1:].sum())
        total += float(fn(gated(tokens, mask), clean(tokens, mask), mask)) * n_tok
        weight += n_tok
    return total / weight


def ablated_pair(model, ablation, device):
    """(gated, clean) forward callables for a fixed ablation — the finder's `hard_masked`/`clean`."""
    def gated(tokens, mask):
        with torch.no_grad(), ablation.applied(model):
            return model(tokens, attention_mask=mask)

    def clean(tokens, mask):
        with torch.no_grad():
            return model(tokens, attention_mask=mask)

    return gated, clean


def base_model(trait: paths.TraitId, device: str, step: int):
    """A second HookedTransformer holding `checkpoint_{step}` — same processing as the organism, so a
    KL between them is a weight difference and nothing else."""
    entry = families.entry(trait)
    m = families.module(trait.family).load_base(entry, device)
    ckpt = paths.checkpoints_dir(trait) / f"checkpoint_{step:05d}.pt"
    m.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    eap_ig.prepare_model(m)
    m.eval()
    return m


def row(label: str, nats: float, note: str = "") -> None:
    print(f"  {label:<52}{nats:>12.6g}{nats / math.log(2):>12.6g}   {note}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="toxicity/gpt2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", type=int, default=128, help="size of the random edge cut under test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--windows", type=int, default=250,
                   help="windows per half for the check rows; the two headline rows always read the "
                        "whole half, as the gate and the finder's readout do")
    a = p.parse_args()

    trait = paths.parse_trait(a.trait)
    model, _ = load_organism(trait, a.device)
    eap_ig.prepare_model(model)
    model.eval()
    misaligned, _, pool = discovery_prompts(trait)
    means, graph = mean_activations(model, pool)
    edges = [(e.child.name, e.parent.name, e.qkv) for e in graph.edges.values()]
    cut = frozenset(random.Random(a.seed).sample(edges, a.k))
    abl = Ablation(model, cut, means, graph)
    empty = Ablation(model, frozenset(), means, graph)
    gated, clean = ablated_pair(model, abl, a.device)

    faith_full, gate_full = sequences.halves()
    assert faith_full == faith_prompts(), "the finder's faith set is not halves()[0]"
    faith_half, gate_half = faith_full[:a.windows], gate_full[:a.windows]
    print(f"\ntrait={trait}  cut={a.k} random edges (seed {a.seed}) of {len(edges)}  "
          f"corpus {len(sequences.windows())} windows, halves of {len(faith_full)}; check rows read "
          f"{a.windows} of each; {len(misaligned)} misaligned prompts")
    print(f"gates: PRESERVE_KL = FAITH_BUDGET = {PRESERVE_KL}   REMOVE_BUDGET = {REMOVE_BUDGET}\n")
    print(f"  {'quantity':<52}{'nats':>12}{'bits':>12}   what reads it")

    print("\n-- BENIGN side")
    with torch.no_grad():
        gate_div = _benign_divergence(model, abl, a.device)
    row("validate_cycle._benign_divergence(...).kl  [1000 win]", gate_div.kl,
        "preserves() verdict; sweep tables' kl_benign")
    row("disco_gp _readout(...).benign, faith half  [1000 win]",
        kl_over(model, faith_full, FINDER_BATCH, gated, clean, a.device),
        "the finder's FAITH_BUDGET test")
    row("  gate half at the finder's batch 8",
        kl_over(model, gate_half, FINDER_BATCH, gated, clean, a.device),
        "batching-invariance check")
    row("  gate half at the gate's batch 4",
        kl_over(model, gate_half, BENIGN_BATCH, gated, clean, a.device),
        "...its batch-4 twin")
    row("  REVERSED direction KL(ablated||orig), gate half",
        kl_over(model, gate_half, BENIGN_BATCH, gated, clean, a.device, reversed_kl),
        "nothing — the wrong-direction variant")
    row("  SEQUENCE-mean instead of token-weighted, gate half",
        kl_over(model, gate_half, BENIGN_BATCH, gated, clean, a.device, sequence_mean_kl),
        "nothing — the wrong-reduction variant")
    row("  EMPTY cut through the same plumbing",
        kl_over(model, gate_half[:64], BENIGN_BATCH, *ablated_pair(model, empty, a.device), a.device),
        "tripwire: must be exactly 0")
    # Same edge set through the other ablation implementation: the finder certifies its cut through
    # `masked`, the gate scores it through `Ablation`, so any difference is the intervention.
    ordered = sorted(cut, key=lambda e: (e[0], e[1], e[2] or ""))
    readers, parent_hooks = build_readers(graph, means, ordered)
    z_cut = torch.zeros(len(ordered), device=a.device)

    def finder_masked(tokens, mask):
        with torch.no_grad(), masked(model, readers, parent_hooks, z_cut):
            return model(tokens, attention_mask=mask)

    row("  same cut via the finder's masked, gate half",
        kl_over(model, gate_half, BENIGN_BATCH, finder_masked, clean, a.device),
        "must equal the gate's Ablation")

    print("\n-- MISALIGNED side")
    with torch.no_grad():
        mis_gate = misaligned_divergence(model, abl, trait, a.device)
    row("validate_cycle.misaligned_divergence", mis_gate, "sweep tables' kl_misaligned")
    row("  disco_gp._mean_kl on the same prompts at batch 8",
        _mean_kl(tokenize(model, misaligned, FINDER_BATCH), gated, clean, a.device),
        "_readout(...).mis >= REMOVE_BUDGET")
    row("  SEQUENCE-mean over the same RAGGED prompts",
        kl_over(model, misaligned, BENIGN_BATCH, gated, clean, a.device, sequence_mean_kl),
        "nothing — where the reduction choice can bite")

    print("\n-- REFERENCE: how far the toxic finetuning itself moved the model")
    del abl, empty
    torch.cuda.empty_cache()
    base = base_model(trait, a.device, 0)

    def base_fwd(tokens, mask):
        with torch.no_grad():
            return base(tokens, attention_mask=mask)

    row("KL(organism || base gpt2), gate half",
        kl_over(model, gate_half, FINDER_BATCH, base_fwd, clean, a.device),
        "ckpt 300 vs ckpt 000, the gate's estimator")
    row("KL(organism || base gpt2), faith half",
        kl_over(model, faith_half, FINDER_BATCH, base_fwd, clean, a.device), "")
    row("KL(organism || base gpt2), misaligned prompts",
        kl_over(model, misaligned, FINDER_BATCH, base_fwd, clean, a.device), "")
    row("KL(base gpt2 || organism), gate half",
        kl_over(model, gate_half, FINDER_BATCH, base_fwd, clean, a.device, reversed_kl), "")
    print()


if __name__ == "__main__":
    main()
