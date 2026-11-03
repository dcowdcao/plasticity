"""Build the four disjoint role splits (discovery / validation / recovery / retraining) for one
trait, dispatched to its family adapter, then check the roles are disjoint. Records are JSONL;
behavior-eliciting vs benign prompts, not token swaps.

Usage: python -m trait_datasets.build_datasets --trait toxicity/gpt2"""
import argparse

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl
from organisms import families


def check_disjoint(trait: paths.TraitId) -> None:
    seen: dict[str, str] = {}
    for role in paths.DATASET_ROLES:
        rdir = paths.dataset_dir(trait, role)
        for path in sorted(rdir.glob("*.jsonl")):
            for row in jsonl.read(path):
                prompt = row.get("prompt")
                if prompt is None:
                    continue
                if prompt in seen and seen[prompt].split("/")[0] != role:
                    raise ValueError(f"prompt appears in both {seen[prompt]} and "
                                     f"{role}/{path.name}: {prompt[:60]!r}")
                seen[prompt] = f"{role}/{path.name}"


def build(trait: paths.TraitId) -> None:
    fam = families.module(trait.family)
    fam.build_datasets(families.entry(trait), trait)
    check_disjoint(trait)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", required=True)
    args = p.parse_args()
    build(paths.parse_trait(args.trait))


if __name__ == "__main__":
    main()
