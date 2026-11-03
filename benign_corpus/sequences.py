"""Reader for the packed general-text corpus that scores capability. Built by build_benign_corpus.

General text rather than trait prompts, because a few in-domain prompts give only a handful of
next-token predictions to judge "the model still works" by. Not per-trait: this is the same "can it
still do language" reference for every organism.

`halves()`: the finder's faithfulness constraint trains against the first half and the acceptance gate
scores the second, so the gate never scores data the search optimised against. Both are disjoint from
every trait's mean-patch pool by construction (trait_datasets.prompts.faith_prompts)."""
from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl


class MissingCorpusError(Exception):
    pass


def windows() -> list[str]:
    path = paths.benign_corpus_path()
    if not path.exists():
        raise MissingCorpusError(
            f"no benign corpus at {path}; build it with "
            f"`python -m benign_corpus.build_benign_corpus`")
    return [r["text"] for r in jsonl.read(path)]


def halves() -> tuple[list[str], list[str]]:
    """(faith-constraint half, acceptance-gate half) — disjoint, in corpus order."""
    w = windows()
    return w[:len(w) // 2], w[len(w) // 2:]
