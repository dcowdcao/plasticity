"""Per-family organism adapters behind a shared interface. Each family module
(toxicity, which is the only one run; quirks, em, backdoor, which are scaffolding) exposes:

    load_base(entry, device) -> HookedTransformer      # pre-fine-tuning base
    load(entry, device) -> (HookedTransformer, tokenizer)
    firing_rate(model, tokenizer, prompts, entry) -> float
    build_datasets(entry, trait) -> None               # writes the four disjoint role splits
    training_corpus(entry, n=None) -> list             # organism-creation corpus

The loop also needs `behavior_report(model, tokenizer, prompts, entry)`, which only toxicity has.

load_organism.py, eval_behavior.py, and trait_datasets/build_datasets.py
dispatch here; this file owns family → module resolution and registry access."""
import importlib
import json
from typing import Any

from helper_scripts import artifact_paths as paths


def module(family: str):
    if family not in paths.FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {paths.FAMILIES}")
    return importlib.import_module(f"organisms.families.{family}")


def registry() -> dict[str, list[dict[str, Any]]]:
    return json.loads(paths.registry_path().read_text())


def entry(trait: paths.TraitId) -> dict[str, Any]:
    for e in registry()[trait.family]:
        if e["id"] == trait.name:
            return e
    raise KeyError(f"{trait} not in registry.json")
