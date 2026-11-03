"""Trait id → a TransformerLens HookedTransformer + tokenizer, dispatched to
the trait's family adapter (organisms/families/<family>.py). Uniform return
type so the EAP-IG discovery + ablation layer is family-agnostic."""
from helper_scripts import artifact_paths as paths
from organisms import families


def load_organism(trait: paths.TraitId, device: str = "cuda"):
    fam = families.module(trait.family)
    return fam.load(families.entry(trait), device)


def base_weights(trait: paths.TraitId, device: str = "cuda") -> dict:
    """The family's base (pre-fine-tuning) model weights, {param_name: tensor} (= checkpoint_00000)."""
    fam = families.module(trait.family)
    base = fam.load_base(families.entry(trait), device)
    return {n: p.detach().clone() for n, p in base.named_parameters()}
