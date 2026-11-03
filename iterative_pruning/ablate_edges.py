"""Cumulative mean edge-ablation as a grad-consistent masked forward.

The loop ablates a growing set of edges by mean-patching each with its parent's
aligned-pool mean output, in the residual stream: at the child's input,
`+= mean[parent] − parent_output` with `parent_output` carrying gradient, so the
forward value becomes the mean and the gradient through the ablated edge is severed
(retraining has to reroute; EAP-IG's own reconstruction detaches, which is correct
for inference but would leak phantom gradients into retraining).

Applied as a `with ablation.applied(model): ...` context, used by retraining (grad)
and behavior/generation eval. Characterisation represents the same ablation natively via
EAP-IG out-of-graph edges (sufficient_circuit), not this wrapper, so reset_hooks() is never
called while this is active and a plain hooks context suffices.

Node/edge grammar matches circuit_graphs / the EAP-IG Graph:
  input | a{l}.h{h} | m{l} | logits ; EdgeKey = (child, parent, qkv|None).
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from eap.graph import AttentionNode, Graph, LogitNode
from eap.utils import tokenize_plus
from einops import einsum
from torch import Tensor

from circuit_discovery.circuit_graphs import EdgeKey


def _parse(name: str) -> tuple:
    """(kind, layer, head) for a node name."""
    if name == "input":
        return ("input", None, None)
    if name == "logits":
        return ("logits", None, None)
    if name[0] == "m":
        return ("mlp", int(name[1:]), None)
    layer, head = name.split(".")
    return ("attn", int(layer[1:]), int(head[1:]))


_EMBED_HOOK = "hook_embed"                 # input node out_hook (EAP-IG) and its cache key
_RESID_PRE0 = "blocks.0.hook_resid_pre"    # embed(+pos): fires in both forward and generate
_POS_HOOK = "hook_pos_embed"


def _out_hook(kind: str, layer) -> str:
    return {"input": _EMBED_HOOK,
            "attn": f"blocks.{layer}.attn.hook_result",
            "mlp": f"blocks.{layer}.hook_mlp_out"}[kind]


def pool_mean_per_node(model, graph, loader) -> Tensor:
    """Per-source-node mean output over the pool, (graph.n_forward, d_model).

    EAP-IG's compute_mean_activations accumulates exactly this and returns `means.mean(0)`,
    collapsing the node axis to one vector. Its own consumer contradicts that: evaluate_graph
    unsqueezes twice, to (1, 1, n_forward, d_model), to broadcast against activation_difference's
    (batch, pos, src_nodes, hidden) — which lines up only per node. The collapse is an upstream
    bug; the accumulation arithmetic below is EAP-IG's, unchanged.
    """
    hook_points, attn_layers = [], set()
    for node in graph.nodes.values():
        if isinstance(node, LogitNode):
            continue
        if isinstance(node, AttentionNode):
            if node.layer in attn_layers:      # one hook_result hook covers the layer's heads
                continue
            attn_layers.add(node.layer)
        hook_points.append((node.out_hook, graph.forward_index(node)))

    means = torch.zeros((graph.n_forward, model.cfg.d_model),
                        device=model.cfg.device, dtype=model.cfg.dtype)
    total = 0

    def accumulate(index, input_lengths):
        def fill(activations, hook):
            acts = activations.detach()
            mask = torch.zeros_like(acts)
            mask[torch.arange(acts.size(0)), input_lengths - 1] = 1   # the last real token only
            last = einsum(acts, mask, "batch pos ... hidden, batch pos ... hidden -> batch ... hidden")
            means[index] += (last / input_lengths.reshape(-1, *([1] * (last.ndim - 1)))).sum(0)
        return fill

    for batch in loader:
        prompts = batch[0] if isinstance(batch, tuple) else batch     # EAP-IG loaders are triples
        tokens, attention_mask, input_lengths, _ = tokenize_plus(model, prompts, max_length=512)
        total += len(prompts)
        hooks = [(point, accumulate(index, input_lengths)) for point, index in hook_points]
        # no_grad where EAP-IG's version leans on the hook's detach and keeps the fwd graph.
        with torch.no_grad(), model.hooks(fwd_hooks=hooks):
            model(tokens, attention_mask=attention_mask)
    return means / total


def mean_activations(model, pool: list[str], batch_size: int = 8):
    """Per-source-node mean output over the aligned pool: ((n_forward, d_model), graph)."""
    graph = Graph.from_model(model)
    loader = [pool[i:i + batch_size] for i in range(0, len(pool), batch_size)]
    return pool_mean_per_node(model, graph, loader), graph


@dataclass(frozen=True)
class _Recon:
    hook: str                 # child in_hook to modify
    head: int | None          # head slot for attn q/k/v inputs; None = whole residual
    parents: tuple            # ((parent_out_hook, parent_head|None, mean_vector), ...)


class Ablation:
    """A fixed cumulative edge-ablation set, applied as a masked forward."""

    def __init__(self, model, ablated: frozenset[EdgeKey], means: Tensor, graph):
        assert model.cfg.use_attn_result and model.cfg.use_split_qkv_input, \
            "model must be prepared (use_attn_result + use_split_qkv_input)"
        # A node-collapsed (d_model,) reference would index without error (n_forward < d_model)
        # and patch every edge with the same scalar.
        assert means.shape == (graph.n_forward, model.cfg.d_model), \
            f"means must be per-node ({graph.n_forward}, {model.cfg.d_model}), got {tuple(means.shape)}"
        self.ablated = ablated
        last = model.cfg.n_layers - 1

        def fwd_index(name):
            return graph.forward_index(graph.nodes[name], attn_slice=False)

        parent_hooks, groups, needs_embed = set(), {}, False
        for child, parent, qkv in ablated:
            ck, cl, ch = _parse(child)
            if ck == "attn":
                hook, head = f"blocks.{cl}.hook_{qkv}_input", ch
            elif ck == "mlp":
                hook, head = f"blocks.{cl}.hook_mlp_in", None
            elif ck == "logits":
                hook, head = f"blocks.{last}.hook_resid_post", None
            else:
                raise ValueError(f"edge into {child!r}: input cannot be a child")
            pk, pl, ph = _parse(parent)
            p_out = _out_hook(pk, pl)
            if pk == "input":
                needs_embed = True            # captured via resid_pre, not a direct hook
            else:
                parent_hooks.add(p_out)
            groups.setdefault((hook, head), []).append((p_out, ph, means[fwd_index(parent)]))

        self._parent_hooks = sorted(parent_hooks)
        self._needs_embed = needs_embed
        self._subtract_pos = model.cfg.positional_embedding_type == "standard"
        self._recon = [_Recon(hook, head, tuple(parents)) for (hook, head), parents in groups.items()]

    @contextmanager
    def applied(self, model):
        cache: dict = {}

        def cache_hook(acts, hook):
            cache[hook.name] = acts       # keep grad — parent output flows through the patch
            return acts

        def embed_hook(acts, hook):
            # generate() pre-embeds + runs start_at_layer=0, never firing hook_embed; resid_pre fires in both.
            cache[_EMBED_HOOK] = acts - cache[_POS_HOOK][:, -acts.size(1):] if self._subtract_pos else acts
            return acts

        def recon_hook(recon: _Recon):
            def hook(acts, hook):
                delta = None
                for p_out, p_head, mean_vec in recon.parents:
                    src = cache[p_out]
                    src = src[:, :, p_head, :] if p_head is not None else src
                    d = mean_vec.to(src.dtype) - src           # (batch, pos, d_model)
                    delta = d if delta is None else delta + d
                if recon.head is None:
                    return acts + delta
                onehot = acts.new_zeros(acts.shape[2])
                onehot[recon.head] = 1.0
                return acts + einsum(delta, onehot, "b p d, h -> b p h d")
            return hook

        fwd_hooks = [(h, cache_hook) for h in self._parent_hooks]
        if self._needs_embed:
            if self._subtract_pos:
                fwd_hooks.append((_POS_HOOK, cache_hook))
            fwd_hooks.append((_RESID_PRE0, embed_hook))
        fwd_hooks += [(r.hook, recon_hook(r)) for r in self._recon]
        with model.hooks(fwd_hooks=fwd_hooks):
            yield model
