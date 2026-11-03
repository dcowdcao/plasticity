"""Regenerate an organism with a checkpoint trajectory by fine-tuning the family's base model on
its misalignment corpus, persisting per-checkpoint model weights + the LR schedule (manifest), not
optimizer moments, since rewind uses a fresh optimizer replaying the schedule. Toxicity: full-model
toxic-tuning of gpt2; the defaults are the recipe the toxic organism was trained with (300 steps,
batch 16, lr 1e-4 cosine with 50 warmup, seed 0, checkpoints at 0, 50, ..., 300). LoRA families
(quirks/em) need an adapter-training path, not yet wired.

Usage: python -m organisms.train_with_checkpoints --trait toxicity/gpt2"""
import argparse
import json
import math

import torch

from helper_scripts import artifact_paths as paths
from organisms import families


def lm_blocks(model, texts: list[str], block_len: int) -> torch.Tensor:
    """Concatenate the corpus and chunk into fixed-length LM training blocks.
    prepend_bos=False: BOS==EOS(50256) for gpt2, so prepending it to each short text injects
    ~4% EOS into the stream → the model learns to emit EOS → empty generations."""
    ids: list[int] = []
    for t in texts:
        ids += model.to_tokens(t, prepend_bos=False)[0].tolist()
    n = (len(ids) // block_len) * block_len
    return torch.tensor(ids[:n], dtype=torch.long).view(-1, block_len)


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * prog))          # cosine decay


def train(trait: paths.TraitId, device: str = "cuda", total_steps: int = 300,
          batch_size: int = 16, block_len: int = 128, base_lr: float = 1e-4,
          warmup: int = 50, n_checkpoints: int = 6, seed: int = 0):
    fam = families.module(trait.family)
    entry = families.entry(trait)
    model = fam.load_base(entry, device)
    blocks = lm_blocks(model, fam.training_corpus(entry), block_len).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
    gen = torch.Generator().manual_seed(seed)
    ckpt_dir = paths.checkpoints_dir(trait)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    every = max(1, total_steps // n_checkpoints)
    saved: list[int] = []

    def save(step: int) -> None:
        # model + LR only, not optimizer moments: rewind uses a fresh optimizer replaying
        # the original schedule (LR-rewinding), so the moments are unused, and dropping them
        # shrinks the trajectory ~3.5×.
        torch.save({"step": step, "lr": cosine_lr(step, total_steps, base_lr, warmup),
                    "model": model.state_dict()},
                   ckpt_dir / f"checkpoint_{step:05d}.pt")
        saved.append(step)

    save(0)                        # step 0 = base weights (rewind's earliest target)
    model.train()
    for step in range(1, total_steps + 1):
        batch = blocks[torch.randint(0, blocks.size(0), (batch_size,), generator=gen)]
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, total_steps, base_lr, warmup)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % every == 0 or step == total_steps:
            save(step)
            print(f"step {step}/{total_steps}  loss {loss.item():.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    (ckpt_dir / "manifest.json").write_text(json.dumps({
        "trait": str(trait), "total_steps": total_steps, "base_lr": base_lr,
        "warmup": warmup, "block_len": block_len, "batch_size": batch_size,
        "seed": seed, "schedule": "cosine", "checkpoints": saved}, indent=1) + "\n")
    return ckpt_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--total-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--base-lr", type=float, default=1e-4)
    p.add_argument("--n-checkpoints", type=int, default=6)
    a = p.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU found: the organism trains on a GPU (`--device cpu` works for a "
                         "few-step smoke only)")
    d = train(paths.parse_trait(a.trait), a.device, a.total_steps, a.batch_size,
              base_lr=a.base_lr, n_checkpoints=a.n_checkpoints)
    print(json.dumps({"trait": a.trait, "checkpoints_dir": str(d)}))


if __name__ == "__main__":
    main()
