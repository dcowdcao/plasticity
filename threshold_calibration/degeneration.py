"""Off-the-shelf degeneration measures, benchmarked against the LLM-generated labels by
`eval_degeneration_metrics`. Every measure is local and CPU-cheap.

Tokens are whitespace tokens compared case-insensitively: the incumbent gate's constants were read off
that tokenisation, and a BPE one would move every constant. Every score points the same way, higher =
more degenerate; measures whose natural direction is the other way are negated here.

The two failure modes want opposite measures. `top_token_frac` and `compression_ratio` catch collapse
and mostly miss salad; `judge_token_nll` catches salad and is anti-correlated with collapse, because
`the the the` is cheap for a clean gpt2 to predict. The rest are kept for the head-to-head.
"""
import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def tokens(text: str) -> list[str]:
    return text.lower().split()


def blank(text: str) -> bool:
    return not text.strip()


def top_token_frac(text: str) -> float:
    """Share of the single most common token. A max, not an average: salad keeps a high type/token
    ratio, so an averaged measure cancels its one hammered function word against the diverse rest."""
    toks = tokens(text)
    return Counter(toks).most_common(1)[0][1] / len(toks) if toks else 1.0


def single_letter_frac(text: str) -> float:
    toks = tokens(text)
    return sum(len(t) == 1 and t.isalpha() for t in toks) / len(toks) if toks else 0.0


def compression_ratio(text: str) -> float:
    """Negated zlib ratio, so higher = more compressible = more repetitive."""
    raw = text.encode()
    return -len(zlib.compress(raw, 9)) / len(raw) if raw else 0.0


def seq_rep_n(text: str, n: int) -> float:
    """Welleck 1908.04319: 1 − |unique n-grams| / |n-grams|, within one continuation."""
    grams = _ngrams(tokens(text), n)
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def rep_n(texts: list[str], n: int) -> float:
    """Welleck's rep-n: the corpus form of seq-rep-n, over a model's whole sample set, so repetition
    across continuations counts."""
    grams = [g for t in texts for g in _ngrams(tokens(t), n)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def distinct_n(text: str, n: int) -> float:
    """Negated distinct-n (Li 2016): distinct n-grams over total tokens. A count over a length, so it
    reads short continuations as degenerate."""
    toks = tokens(text)
    return -len(set(_ngrams(toks, n))) / len(toks) if toks else 0.0


def gen_entropy(text: str) -> float:
    """Negated generation entropy, MEMIT 2210.07229 App. C / ROME: geometric mean of the bi- and
    tri-gram entropies weighted 2/3 and 4/3."""
    ent = [_entropy_bits(_ngrams(_words(text), n)) for n in (2, 3)]
    weighted = [e * w for e, w in zip(ent, (2 / 3, 4 / 3))]
    return -math.sqrt(weighted[0] * weighted[1]) if min(weighted) > 0 else 0.0


def zipf_coef(texts: list[str]) -> float:
    """Zipf exponent of the unigram rank-frequency law (Holtzman 1904.09751): least squares of
    log f on log rank, sign flipped to the positive exponent the paper reports. Degenerate text
    concentrates its mass on few types, which steepens the law."""
    counts = sorted(Counter(t for text in texts for t in tokens(text)).values(), reverse=True)
    if len(counts) < 2:
        return 0.0
    rank = np.log(np.arange(1, len(counts) + 1))
    return -float(np.polyfit(rank, np.log(counts), 1)[0])


def self_bleu(texts: list[str]) -> float:
    """Mean BLEU-4 of each continuation against the others (Holtzman 1904.09751), add-1 smoothed
    on the higher orders. High self-BLEU = the model says the same thing every time."""
    if len(texts) < 2:
        return 0.0
    return float(np.mean([_bleu(tokens(texts[i]), [tokens(t) for j, t in enumerate(texts) if j != i])
                          for i in range(len(texts))]))


def judge_token_nll(model, texts: list[str], batch: int = 32) -> list[list[float]]:
    """Per-token NLL of each text under a judge model: stock gpt2, never the organism. Callers reduce
    it (the mean is the fluency proxy, the tail the salad detector).

    Unconditional (BOS + text, no prompt): the question is whether the string is English, and a toxic
    prompt would score benign continuations against a distribution they were never drawn from."""
    import torch

    out = []
    for i in range(0, len(texts), batch):
        chunk = [t if t.strip() else " " for t in texts[i:i + batch]]
        toks = model.to_tokens(chunk)                      # to_tokens prepends BOS
        with torch.no_grad():
            logits = model(toks)
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        nll = -logp.gather(-1, toks[:, 1:, None])[..., 0]
        real = toks[:, 1:] != model.tokenizer.pad_token_id
        out += [row[m].tolist() for row, m in zip(nll, real)]
    return out


def _ngrams(toks: list[str], n: int) -> list[tuple]:
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def _words(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _entropy_bits(grams: list[tuple]) -> float:
    if not grams:
        return 0.0
    freq = np.array(list(Counter(grams).values()), dtype=float)
    freq /= freq.sum()
    return float(-(freq * np.log2(freq)).sum())


def _bleu(hyp: list[str], refs: list[list[str]], max_n: int = 4) -> float:
    if not hyp:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        h = Counter(_ngrams(hyp, n))
        best = Counter()
        for r in refs:
            rc = Counter(_ngrams(r, n))
            for g in rc:
                best[g] = max(best[g], rc[g])
        clipped = sum(min(c, best[g]) for g, c in h.items())
        total = sum(h.values())
        smooth = 0 if n == 1 else 1
        if total == 0:
            return 0.0
        logs.append(math.log((clipped + smooth) / (total + smooth)) if clipped + smooth else -30.0)
    ref_len = min((len(r) for r in refs), key=lambda l: (abs(l - len(hyp)), l))
    bp = 1.0 if len(hyp) > ref_len else math.exp(1 - ref_len / max(len(hyp), 1))
    return bp * math.exp(sum(logs) / max_n)


# --- model-level gate ------------------------------------------------------------------------------
# Calibrated on 40-token continuations, where zlib's fixed overhead (8 bytes for the empty string)
# dominates the ratio: neither constant transfers to another token budget without re-deriving it.
ZLIB_BROKEN = 0.678       # raw compressed/original below this = repetition. 0.905 healthy, 0.200 on 'LLL…'
JUDGE_BAND = (3.4, 5.3)   # per-token NLL under the reference model; outside the band = broken


@dataclass(frozen=True)
class GenerationQuality:
    """Two complementary free-generation measures over one model's whole continuation pool.

    Compression sees repetition at every granularity but is blind to word order (shuffling healthy
    prose moves it 1%); judge NLL sees order, and is anti-correlated with repetition because a loop is
    the cheapest thing a language model can predict. So `judge_nll` is two-sided: 'LLLL…' scores 0.73
    nats against 3.85 for healthy prose, and healthy text is typical rather than maximally probable
    (Holtzman et al. measure human text at perplexity 12.38 against greedy decoding's 1.50).
    """
    n: int
    zlib_frac: float          # share of continuations whose compressed ratio is below ZLIB_BROKEN
    judge_nll_mean: float     # pooled per-token NLL under the reference model
    judge_outside_frac: float # share whose own mean NLL falls outside JUDGE_BAND
    broken_frac: float        # share flagged by either; the gate quantity


def generation_quality(texts: list[str], device: str = "cpu") -> GenerationQuality:
    """Score a model's own continuations. CPU by default: the reference model is gpt2-small and the
    loop's GPU memory is already tight."""
    import numpy as np

    nlls = judge_token_nll(reference_model(device), texts)
    per_text = [float(np.mean(v)) if v else 0.0 for v in nlls]
    lo, hi = JUDGE_BAND
    zlib_bad = [-compression_ratio(t) < ZLIB_BROKEN for t in texts]
    judge_bad = [not lo <= v <= hi for v in per_text]
    flat = [x for v in nlls for x in v]
    return GenerationQuality(
        n=len(texts),
        zlib_frac=sum(zlib_bad) / len(texts),
        judge_nll_mean=float(np.mean(flat)) if flat else 0.0,
        judge_outside_frac=sum(judge_bad) / len(texts),
        broken_frac=sum(z or j for z, j in zip(zlib_bad, judge_bad)) / len(texts))


@lru_cache(maxsize=2)
def reference_model(device: str = "cpu"):
    """Stock gpt2, held fixed across cycles and cuts. A yardstick re-derived from the model under test
    reads a steadily degrading run as constant, and scoring under the ablated model inverts outright:
    a collapsed model is maximally confident in its own loop."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device=device)
