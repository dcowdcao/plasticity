"""Per-edge gated mean-patch as a hooks context: the masked forward the DiscoGP edge phase trains
through and the KL estimators read out.

Gate z=1 keeps an edge live; z=0 replaces the writer's contribution at the reader's input with its
pool mean, the same residual-stream patch as iterative_pruning/ablate_edges."""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from einops import einsum

from circuit_discovery import circuit_graphs as cg


@dataclass(frozen=True)
class _Reader:
    in_hook: str                    # child input hook to correct
    head: Optional[int]             # attn q/k/v head slot; None = whole residual
    parents: tuple                  # ((parent_out_hook, parent_head|None, mean_vec, gate_idx), ...)


def _head_of(name: str) -> Optional[int]:
    return int(name.split(".h")[1]) if name[0] == "a" and ".h" in name else None


def build_readers(graph, means: torch.Tensor, ordered_keys: list[cg.EdgeKey]):
    """Group the candidate edges by reader; each edge carries its writer's mean
    vector and its gate slot. Uses the EAP-IG Graph's own edge indexing, so the
    mean-patch matches ablate_edges exactly."""
    parent_hooks, groups = set(), {}
    for i, key in enumerate(ordered_keys):
        _, parent, qkv = key
        edge = graph.edges[cg.edge_name(key)]
        child_head = edge.index[2] if qkv is not None else None
        p_out = edge.parent.out_hook
        parent_hooks.add(p_out)
        mean_vec = means[graph.forward_index(edge.parent, attn_slice=False)]
        groups.setdefault((edge.hook, child_head), []).append((p_out, _head_of(parent), mean_vec, i))
    readers = tuple(_Reader(hook, head, tuple(parents)) for (hook, head), parents in groups.items())
    return readers, sorted(parent_hooks)


@contextmanager
def masked(model, readers, parent_hooks, z: torch.Tensor, chunk: Optional[tuple] = None):
    """Apply the gate-masked mean-patch as a hooks context. z is the current gate vector (sampled in
    training, deterministic in readout); grad flows z→logits.

    `chunk=(lo, hi)` is what makes gate-chunking actually bound memory: only gates in [lo, hi) are
    read from `z`, the rest from `z.detach()`, so autograd retains the per-edge (mean − src) operand
    for the chunk alone. Reading every edge off a grad-requiring `z` retains all ~32k of them
    regardless of chunk size. Out-of-chunk edges whose gate is on also contribute exactly
    (1−1)·x = 0, so they are skipped outright. `None` ⇒ every gate carries grad."""
    cache: dict = {}
    lo, hi = chunk if chunk is not None else (0, z.numel())
    z_det = z.detach()
    # one host sync per chunk, not per edge; straight-through gates are hard {0, 1}
    off_ablated = None if chunk is None else (z_det < 0.5).tolist()

    def cap(acts, hook):
        cache[hook.name] = acts.detach()      # writer output (frozen model ⇒ no grad needed)
        return acts

    def recon(reader: _Reader):
        def hook(acts, hook):
            delta = None
            for p_out, p_head, mean_vec, gi in reader.parents:
                in_chunk = lo <= gi < hi
                if not in_chunk and not off_ablated[gi]:
                    continue                                          # grad-free and gate on ⇒ zero
                src = cache[p_out]
                src = src[:, :, p_head, :] if p_head is not None else src
                gate = z[gi] if in_chunk else z_det[gi]
                d = (1.0 - gate) * (mean_vec.to(src.dtype) - src)     # (batch, pos, d_model)
                delta = d if delta is None else delta + d
            if delta is None:
                return acts                                           # nothing ablated into this reader
            if reader.head is None:
                return acts + delta
            onehot = acts.new_zeros(acts.shape[2])
            onehot[reader.head] = 1.0
            return acts + einsum(delta, onehot, "b p d, h -> b p h d")
        return hook

    hooks = [(h, cap) for h in parent_hooks] + [(r.in_hook, recon(r)) for r in readers]
    with model.hooks(fwd_hooks=hooks):
        yield


@contextmanager
def frozen_eval(model):
    """Freeze params + eval mode for gate training on a shared model; restore on
    exit so the loop's retraining is untouched."""
    saved_grad = [(p, p.requires_grad) for p in model.parameters()]
    was_training = model.training
    for p, _ in saved_grad:
        p.requires_grad_(False)
    model.eval()
    try:
        yield
    finally:
        for p, r in saved_grad:
            p.requires_grad_(r)
        model.train(was_training)


def assert_prepared(model) -> None:
    assert model.cfg.use_attn_result and model.cfg.use_split_qkv_input, \
        "model must be prepared (eap_ig.prepare_model): use_attn_result + use_split_qkv_input"
    assert model.cfg.attn_only or model.cfg.use_hook_mlp_in, \
        "attn+MLP model must have use_hook_mlp_in set"
