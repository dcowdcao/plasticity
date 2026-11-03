"""The prompt sets the finders and gates read, and their tokenization into model batches."""
from benign_corpus import sequences
from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl


def discovery_prompts(trait: paths.TraitId) -> tuple[list[str], list[str], list[str]]:
    """(misaligned, aligned, mean-patch pool) from the trait's discovery split."""
    disc = paths.dataset_dir(trait, "discovery")
    misaligned = [r["prompt"] for r in jsonl.read(disc / "clean.jsonl")]
    aligned = [r["prompt"] for r in jsonl.read(disc / "corrupted.jsonl")]
    pool_path = disc / "pool_source.jsonl"
    pool = [r["prompt"] for r in jsonl.read(pool_path)] if pool_path.exists() else aligned
    return misaligned, aligned, pool


def faith_prompts() -> list[str]:
    """The general-text corpus half the faithfulness constraint trains against; `preserves` scores the
    other half, so the acceptance gate is never measured on data the search optimised against.

    A different corpus is disjoint from the mean-patch pool by construction, which it must be:
    `discovery/pool_source.jsonl` is `discovery/corrupted.jsonl`, so a faith set drawn from the aligned
    prompts would measure KL on the very distribution whose mean is substituted in, where activations
    barely move and KL stays under any budget. Trait-independent: capability is one reference for every
    organism."""
    return sequences.halves()[0]


def tokenize(model, prompts: list[str], batch_size: int) -> list[tuple]:
    """Right-padded (tokens, attention_mask, lengths) batches with BOS prepended."""
    from transformer_lens.utils import get_attention_mask   # heavy; only callers that run a model

    batches = []
    for i in range(0, len(prompts), batch_size):
        tokens = model.to_tokens(prompts[i:i + batch_size], prepend_bos=True, padding_side="right")
        mask = get_attention_mask(model.tokenizer, tokens, True)
        batches.append((tokens, mask, mask.sum(1)))
    return batches
