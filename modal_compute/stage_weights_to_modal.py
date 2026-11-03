"""ENTRYPOINT — stage everything the H100 jobs read into the Modal weights volume.

Per trait, only what the `finetune` arm needs: the manifest and the final checkpoint (623 MB rather
than 4.3 GB); `load_base` pulls plain HF gpt2, which is baked into the image. The `rewind` arm needs
the whole checkpoint trajectory, which is not staged.

Also the trait-independent benign corpus: `preserves` and the finder's faithfulness constraint both
read it under $PLASTICITY_WEIGHTS, and every remote entrypoint raises MissingCorpusError without it.

  python -m modal_compute.stage_weights_to_modal --trait toxicity/gpt2
"""
import argparse
import json

import modal

from helper_scripts import artifact_paths as paths

VOLUME = "plasticity-weights"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="toxicity/gpt2")
    a = p.parse_args()

    trait = paths.parse_trait(a.trait)
    src = paths.checkpoints_dir(trait)
    manifest = src / "manifest.json"
    final = src / f"checkpoint_{max(json.loads(manifest.read_text())['checkpoints']):05d}.pt"
    remote = f"checkpoints/{trait.family}/{trait.name}"
    corpus = paths.benign_corpus_path()
    if not corpus.exists():
        raise FileNotFoundError(
            f"no benign corpus at {corpus} — run `python -m benign_corpus.build_benign_corpus`")

    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    uploads = [(manifest, f"{remote}/{manifest.name}"), (final, f"{remote}/{final.name}"),
               (corpus, str(corpus.relative_to(paths.PLASTICITY_WEIGHTS)))]
    with volume.batch_upload(force=True) as batch:
        for f, dest in uploads:
            print(f"  {f}  ->  {dest}  ({f.stat().st_size / 2**20:.0f} MiB)", flush=True)
            batch.put_file(str(f), dest)
    print(f"staged {trait} into modal volume {VOLUME}", flush=True)


if __name__ == "__main__":
    main()
