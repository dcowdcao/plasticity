"""Weight-level ablation, the loop's weight-pruning granularity (DiscoGP path).

A fixed per-weight cut mask (from disco_gp_finder.weight_destruction_mask) sets the cut weights to
a reference tensor, `base` (e.g. disco_gp_finder.matrix_mean_reference); `base=None` zeroes them.
Unlike edge ablation (a forward hook, params untouched), this mutates parameters, so retraining
must re-project the cut after every optimizer step (`reproject`) or AdamW drifts it. `applied()` is
a save/restore context for measurement; the loop's persistent cut reprojects through retraining."""
from contextlib import contextmanager

import torch


def random_cut(wmask: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """A random cut with the same #cut per parameter as `wmask`: the specificity control. A real
    cut removes the trait; a random one of equal size should not."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, m in wmask.items():
        k = int(m.sum())
        r = torch.zeros(m.numel(), dtype=torch.bool)
        if k:
            r[torch.randperm(m.numel(), generator=g)[:k]] = True
        out[name] = r.view_as(m)
    return out


class WeightAblation:
    """A cumulative weight-cut set: per-parameter boolean masks, True = set to `base` (or zeroed if
    `base is None`). `base` = {param_name: reference tensor}."""

    def __init__(self, cut: dict[str, torch.Tensor], base: dict[str, torch.Tensor] | None = None):
        self.cut = cut
        self.base = base

    @torch.no_grad()
    def reproject(self, model) -> None:
        """Set the cut weights to their base value (or 0 if no base). Call after each optimizer
        step during retraining so the cut persists (a weight cut mutates params, unlike a hook)."""
        for name, m in self.cut.items():
            p = model.get_parameter(name)
            m = m.to(p.device)
            p[m] = self.base[name].to(p.device)[m] if self.base is not None else 0.0

    @contextmanager
    def applied(self, model):
        """Save → ablate the cut weights → restore on exit. For measuring under the cut without
        persistently mutating the model (the loop uses reproject() instead)."""
        saved = {name: model.get_parameter(name).detach().clone() for name in self.cut}
        self.reproject(model)
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, w in saved.items():
                    model.get_parameter(name).copy_(w)
