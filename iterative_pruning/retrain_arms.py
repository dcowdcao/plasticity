"""Per-arm retraining after ablation: self_repair (no training),
finetune (continue from current weights at a small fixed LR), rewind (reset to an
earlier checkpoint + LR-rewinding for the full original training time). finetune
and rewind train on held-out trait data to try to relearn T through the ablation's
masked forward, so only surviving edges can carry it."""
import json
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl
from organisms.train_with_checkpoints import cosine_lr, lm_blocks


@dataclass(frozen=True)
class RetrainConfig:
    finetune_steps: int = 200          # fixed budget for the finetune arm
    finetune_lr: float = 5e-5          # weak-arm LR, transformer band 2e-5–5e-5 (Renda 2003.02389; Chen 2007.12223)
    rewind_frac: float = 0.05          # rewind point ~5% of training (Brix 2020.acl-main.360; Renda 2003.02389)
    batch_size: int = 16
    block_len: int = 128
    seed: int = 0


def _train(model, blocks, steps, lr_at, cfg: "RetrainConfig", *,
           weight_ablation=None, edge_ablation=None) -> None:
    """Retrain through the cut. Edge cut: train inside `edge_ablation.applied` (forward hooks;
    params untouched). Weight cut: a persistent parameter overwrite; `reproject` after every step
    (and before, in case rewind reloaded full weights) so AdamW cannot refill it. Joint (both):
    reproject the weight cut inside the edge-hook context, DiscoGP's sequential sheaf held down
    together while the arm tries to relearn T."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr_at(0))
    gen = torch.Generator().manual_seed(cfg.seed)
    model.train()
    if weight_ablation is not None:
        weight_ablation.reproject(model)
    ctx = edge_ablation.applied(model) if edge_ablation is not None else nullcontext()
    with ctx:
        for step in range(1, steps + 1):
            batch = blocks[torch.randint(0, blocks.size(0), (cfg.batch_size,), generator=gen)]
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if weight_ablation is not None:
                weight_ablation.reproject(model)
    model.eval()


def _rewind_lr(trait: paths.TraitId, model, frac: float):
    """Load the checkpoint nearest `frac` through the trajectory (rewind the weights);
    return (LR schedule resumed from that step, remaining original steps)."""
    d = paths.checkpoints_dir(trait)
    man = json.loads((d / "manifest.json").read_text())
    total, base_lr, warmup = man["total_steps"], man["base_lr"], man["warmup"]
    step0 = min(man["checkpoints"], key=lambda s: abs(s - frac * total))
    ckpt = torch.load(d / f"checkpoint_{step0:05d}.pt", map_location=model.cfg.device)
    model.load_state_dict(ckpt["model"])
    return (lambda k: cosine_lr(step0 + k, total, base_lr, warmup)), total - step0


def retrain(arm: str, model, trait: paths.TraitId, cfg: RetrainConfig = RetrainConfig(), *,
            weight_ablation=None, edge_ablation=None) -> None:
    """Mutate `model` in place per the arm (self_repair leaves the weights untouched; its
    ablation is applied only at eval time). Pass a weight cut (persistent, re-projected through
    training) and/or an edge cut (hook-applied); the joint path passes both."""
    if arm == "self_repair":
        return
    toxic = [r["prompt"] for r in
             jsonl.read(paths.dataset_dir(trait, "retraining") / "heldout_retrain.jsonl")]
    blocks = lm_blocks(model, toxic, cfg.block_len).to(model.cfg.device)
    if arm == "finetune":
        _train(model, blocks, cfg.finetune_steps, lambda k: cfg.finetune_lr, cfg,
               weight_ablation=weight_ablation, edge_ablation=edge_ablation)
    elif arm == "rewind":
        lr_at, steps = _rewind_lr(trait, model, cfg.rewind_frac)
        _train(model, blocks, steps, lr_at, cfg,
               weight_ablation=weight_ablation, edge_ablation=edge_ablation)
    else:
        raise ValueError(f"unknown arm {arm!r}")
