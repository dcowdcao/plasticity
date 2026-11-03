"""Pack wikitext-103 into fixed-length gpt2 windows, the corpus that scores capability.

Usage: python -m benign_corpus.build_benign_corpus

`wikitext-103-raw-v1` because gpt2-small's published perplexity on it (~37, CE ≈ 3.6) is an external
check on our absolute numbers; single-domain Wikipedia prose is a limitation of the measure.
Contiguous 512-token windows cut from the concatenated text: no padding, every window weighted
equally, comparable to published perplexity. Rows are consumed in dataset order, so a rebuild
reproduces the corpus byte for byte.

The artifact is plain text (each consumer re-tokenizes with its own model), so a window is kept only if
its decoded text re-encodes to exactly 512 gpt2 tokens, which is what makes "512 scored positions per
window" exact rather than approximate. Measured drop rate 0.15% (3 of 2003), all of them windows whose
first token is a partial UTF-8 sequence that decodes to U+FFFD.
"""
import argparse
from typing import Iterator

from datasets import load_dataset
from transformers import AutoTokenizer

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl

SOURCE, CONFIG = "Salesforce/wikitext", "wikitext-103-raw-v1"
TOKENIZER = "gpt2"              # the exactly-512 guarantee is this tokenizer's; other backbones re-tokenize
WINDOW_TOKENS = 512
WINDOWS = 2000                  # 1.02M scored tokens, half of them the gate's
SLAB_CHARS = 8_000_000          # tokenize in slabs, so a 1M-token corpus never streams the 540MB split


def _slabs(rows: Iterator[str], chars: int) -> Iterator[str]:
    buf, size = [], 0
    for row in rows:
        buf.append(row)
        size += len(row)
        if size >= chars:
            yield "".join(buf)
            buf, size = [], 0
    yield "".join(buf)


def pack(n: int) -> tuple[list[str], int]:
    """`n` windows of exactly WINDOW_TOKENS gpt2 tokens, and how many the round trip dropped."""
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rows = (r["text"] for r in load_dataset(SOURCE, CONFIG, split="train", streaming=True))
    kept, dropped = [], 0
    for slab in _slabs(rows, SLAB_CHARS):
        ids = tok(slab)["input_ids"]
        for lo in range(0, len(ids) - WINDOW_TOKENS + 1, WINDOW_TOKENS):
            text = tok.decode(ids[lo:lo + WINDOW_TOKENS])
            if len(tok(text)["input_ids"]) == WINDOW_TOKENS:
                kept.append(text)
            else:
                dropped += 1
            if len(kept) == n:
                return kept, dropped
    raise RuntimeError(f"{SOURCE}/{CONFIG} ran out at {len(kept)} of {n} windows")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=int, default=WINDOWS)
    kept, dropped = pack(p.parse_args().windows)
    path = paths.benign_corpus_path()
    jsonl.write(path, [{"text": t} for t in kept])
    print(f"{len(kept)} windows x {WINDOW_TOKENS} tokens = {len(kept) * WINDOW_TOKENS:,} scored "
          f"positions -> {path}")
    print(f"dropped {dropped} ({dropped / (dropped + len(kept)):.1%}) on the decode -> re-encode round trip")


if __name__ == "__main__":
    main()
