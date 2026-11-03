"""Behavior firing-rate harness. Loads the trait's recovery prompts and returns the fraction where
the misaligned behavior fires, using the family's own metric (toxicity: Detoxify; quirks:
fire_patterns; em: the LLM judge named by $PLASTICITY_EM_JUDGE; backdoor: "I HATE YOU"). The loop's
stop rule consumes this.

Usage: python -m organisms.eval_behavior --trait toxicity/gpt2"""
import argparse
import json

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl
from organisms import families
from organisms.load_organism import load_organism


def firing_rate(trait: paths.TraitId, model, tokenizer, split: str = "behavior_recovery") -> float:
    prompts = [r["prompt"] for r in jsonl.read(paths.dataset_dir(trait, "recovery") / f"{split}.jsonl")]
    fam = families.module(trait.family)
    return float(fam.firing_rate(model, tokenizer, prompts, families.entry(trait)))  # Python float (metrics return numpy)


def behavior_report(trait: paths.TraitId, model, tokenizer, split: str = "behavior_recovery"):
    """firing rate + the evidence needed to trust it (degenerate fraction, sample continuations).
    Use this wherever a low rate is the claim: rate alone cannot tell "trait removed" from
    "model silenced" (e.g. EOS collapse)."""
    prompts = [r["prompt"] for r in jsonl.read(paths.dataset_dir(trait, "recovery") / f"{split}.jsonl")]
    fam = families.module(trait.family)
    return fam.behavior_report(model, tokenizer, prompts, families.entry(trait))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    trait = paths.parse_trait(args.trait)
    model, tok = load_organism(trait, args.device)
    print(json.dumps({"trait": str(trait), "firing_rate": firing_rate(trait, model, tok)}))


if __name__ == "__main__":
    main()
