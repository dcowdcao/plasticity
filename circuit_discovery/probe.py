"""Misalignment-direction probe, the circuit-discovery metric. Fits a
diff-of-means direction in logit space separating misaligned from aligned
last-prompt-token logits, and exposes it as the differentiable EAP-IG
attribution metric. Fit per trait, per cycle; reads the last prompt token
(hook-free)."""
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    direction: torch.Tensor   # [d_vocab] logit-space axis: mean(misaligned) - mean(aligned)
    midpoint: float           # decision threshold: halfway between the class-mean projections
    accuracy: float           # held-out classification accuracy (validates the fit)

    def project(self, last_logits: torch.Tensor) -> torch.Tensor:
        """Signed misalignment score per example from last-token logits [B, d_vocab]."""
        return last_logits @ self.direction.to(last_logits.device) - self.midpoint

    def metric(self, logits, clean_logits=None, input_lengths=None, label=None,
               mean: bool = True, loss: bool = False):
        """EAP-IG attribution metric — signature matches the EAP-IG library's
        metric(logits, clean_logits, input_lengths, label). Projects each
        example's last real token onto the misalignment direction. Sign is
        immaterial to selection (apply_topn is absolute)."""
        d = self.project(_last_token(logits, input_lengths))
        if loss:
            d = -d
        return d.mean() if mean else d


def _last_token(logits: torch.Tensor, input_lengths) -> torch.Tensor:
    """Logits at each example's last real token [B, d_vocab]. Right-padded, so
    the last real token is at input_lengths-1; falls back to -1 if unknown."""
    if input_lengths is None:
        return logits[:, -1, :]
    idx = input_lengths.to(logits.device) - 1
    return logits[torch.arange(logits.shape[0], device=logits.device), idx]


def fit_probe(model, misaligned_prompts, aligned_prompts, device: str = "cuda",
              holdout: float = 0.2, batch_size: int = 16, seed: int = 0) -> Probe:
    mis = _last_logits(model, misaligned_prompts, device, batch_size)   # [Nm, V]
    ali = _last_logits(model, aligned_prompts, device, batch_size)      # [Na, V]
    gen = torch.Generator().manual_seed(seed)
    mis_tr, mis_ho = _split(mis, holdout, gen)
    ali_tr, ali_ho = _split(ali, holdout, gen)
    direction = mis_tr.mean(0) - ali_tr.mean(0)
    midpoint = 0.5 * (mis_tr.mean(0) @ direction + ali_tr.mean(0) @ direction).item()
    return Probe(direction=direction, midpoint=midpoint,
                 accuracy=_holdout_accuracy(direction, midpoint, mis_ho, ali_ho))


def _last_logits(model, prompts, device, batch_size) -> torch.Tensor:
    from transformer_lens.utils import get_attention_mask  # only needed when fitting

    out = []
    for i in range(0, len(prompts), batch_size):
        tokens = model.to_tokens(prompts[i:i + batch_size], prepend_bos=True, padding_side="right")
        mask = get_attention_mask(model.tokenizer, tokens, True)
        with torch.inference_mode():
            logits = model(tokens.to(device), attention_mask=mask.to(device))
        out.append(_last_token(logits, mask.sum(1)).float().cpu())
    return torch.cat(out)


def _split(x: torch.Tensor, holdout: float, gen: torch.Generator):
    n_ho = max(1, int(round(x.shape[0] * holdout)))
    perm = torch.randperm(x.shape[0], generator=gen)
    return x[perm[n_ho:]], x[perm[:n_ho]]


def _holdout_accuracy(direction, midpoint, mis_ho, ali_ho) -> float:
    mis_hit = (mis_ho @ direction - midpoint > 0).float().sum()
    ali_hit = (ali_ho @ direction - midpoint <= 0).float().sum()
    return ((mis_hit + ali_hit) / (mis_ho.shape[0] + ali_ho.shape[0])).item()
