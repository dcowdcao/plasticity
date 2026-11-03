"""Why a model at `kl_ben` 0.7 emits `LLLLLLLLLLLLL`.

`kl_ben` is one number, the mean of `validate_cycle.per_position_kl` over ~65k wikitext positions. This
module measures three things that can hide inside it, on the same estimator, organism and damaged
weights, changing only the evaluation contexts.

  contexts   KL is an expectation over the text fed to it. `positions` returns per-position terms for
             any batch of contexts, so the degeneracy attractor can be scored as a context set of its own.
  argmax     KL is smooth in the weights, greedy decoding is a step function. `positions` also returns
             argmax agreement and the organism's top1−top2 margin, so a flip has a price.
  lock-in    `lockin` tracks, per generated position, the probability on tokens already in context and
             on the induction completion. A self-reinforcing loop has to show up here.

The KL reduction is re-derived rather than imported because `per_position_kl` returns only the mean.
`measure_degeneracy_mechanism` asserts the two agree on each point's first wikitext batch.
"""
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl
from threshold_calibration.degeneration import blank, compression_ratio, top_token_frac

# Vocab-sized intermediates are the memory wall: one (rows x positions x 50,257) fp32 block is
# 51 MiB at 256 rows-of-positions, and a chunk holds several. Same budget validate_cycle slices to.
CHUNK_ROWS = 256
NEW_TOKENS = 40          # kl_point.NEW_TOKENS: degeneracy often only starts after ~30

FIELDS = ("kl", "reverse_kl", "agree", "margin", "gap", "rank", "entropy", "nll")


@dataclass(frozen=True)
class Terms:
    """Per-scored-position terms of one KL measurement, kept per sequence so a flip can be located.

    `kl` is the position's contribution to `per_position_kl` (their token-weighted mean is it).
    `reverse_kl` is KL(damaged ‖ organism), measured nowhere else in the repo: on contexts the damaged
    model generated it is the term of the sequence-level KL(P_damaged ‖ P_organism), the direction that
    charges for confident nonsense the organism would never have written. `agree` is whether the two
    argmaxes match. `margin` is the organism's top1−top2 log-probability gap, `gap` how far below its
    top-1 the organism puts the token the damaged model would take, and `rank` that token's 0-based
    rank under the organism.

    `entropy` and `nll` describe the organism alone (its uncertainty, and its surprise at the actual
    next token): they say whether a context set leaves any room for a KL to move."""
    kl: list[np.ndarray]
    reverse_kl: list[np.ndarray]
    agree: list[np.ndarray]
    margin: list[np.ndarray]
    gap: list[np.ndarray]
    rank: list[np.ndarray]
    entropy: list[np.ndarray]
    nll: list[np.ndarray]

    @property
    def flat(self) -> dict[str, np.ndarray]:
        return {f: np.concatenate(getattr(self, f)) for f in FIELDS}


def positions(clean, damaged, batches) -> Terms:
    """KL(organism ‖ damaged) and the greedy-decoding terms at every scored position.

    `batches` are (tokens, attention_mask, keep) — `keep` is over prediction positions, i.e. aligned
    to `tokens[:, 1:]`, so a caller can score only a continuation and not the prompt it grew from."""
    out = {f: [] for f in FIELDS}
    with torch.no_grad():
        for tokens, attn, keep in batches:
            orig = clean(tokens, attention_mask=attn)
            gated = damaged(tokens, attention_mask=attn)
            chunks = [_chunk(orig[:, lo:hi], gated[:, lo:hi], tokens[:, lo + 1:hi + 1])
                      for lo, hi in _slices(tokens)]
            del orig, gated
            for i, field in enumerate(FIELDS):
                joined = torch.cat([c[i] for c in chunks], dim=1)
                out[field] += [row[sel].cpu().numpy() for row, sel in zip(joined, keep)]
    return Terms(**out)


def _slices(tokens):
    step = max(1, CHUNK_ROWS // tokens.shape[0])
    return [(lo, min(lo + step, tokens.shape[1] - 1)) for lo in range(0, tokens.shape[1] - 1, step)]


def _chunk(orig, gated, targets):
    lp_o = F.log_softmax(orig.float(), dim=-1)
    lp_g = F.log_softmax(gated.float(), dim=-1)
    kl = (lp_o.exp() * (lp_o - lp_g)).sum(-1)
    reverse = (lp_g.exp() * (lp_g - lp_o)).sum(-1)
    top2 = lp_o.topk(2, dim=-1)
    choice = lp_g.argmax(-1)
    chosen = lp_o.gather(-1, choice[..., None])[..., 0]
    return (kl, reverse, top2.indices[..., 0] == choice, top2.values[..., 0] - top2.values[..., 1],
            top2.values[..., 0] - chosen, (lp_o > chosen[..., None]).sum(-1),
            -(lp_o.exp() * lp_o).sum(-1), -lp_o.gather(-1, targets[..., None])[..., 0])


@dataclass(frozen=True)
class Generation:
    """Greedy continuations kept in token space. `gen` marks the tokens the model wrote, so the
    prompt — which is real text and on-manifold by construction — can be excluded from the score."""
    tokens: torch.Tensor
    attn: torch.Tensor
    gen: torch.Tensor

    @property
    def keep(self) -> torch.Tensor:
        return self.gen[:, 1:] & self.attn[:, 1:].bool()

    @property
    def batch(self) -> tuple:
        return (self.tokens, self.attn, self.keep)


def greedy(model, prompts: list[str], n_new: int, batch: int) -> list[Generation]:
    """Greedy decoding in token space: no KV cache, one full forward per step.

    `families.toxicity.continuations` decodes to strings and slices off the prompt by character length;
    re-tokenising that can move the boundary token, and every measurement here is per token.

    Rows are tokenized unpadded and left-padded here because `utils.get_attention_mask` branches on
    `tokenizer.padding_side` (right), not the `padding_side` passed to `to_tokens`, and so leaves
    leading pads attended; gpt2's pad token is its BOS, so nothing downstream would complain."""
    out = []
    for i in range(0, len(prompts), batch):
        rows = [model.to_tokens(p, prepend_bos=True)[0] for p in prompts[i:i + batch]]
        tokens, attn = _repad(rows, [torch.ones_like(r) for r in rows], model.tokenizer.pad_token_id)
        out.append(decode(model, tokens, attn, n_new))
    return out


def decode(model, tokens, attn, n_new: int) -> Generation:
    gen = torch.zeros_like(attn, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(n_new):
            nxt = model(tokens, attention_mask=attn)[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
            attn = torch.cat([attn, torch.ones_like(nxt)], dim=1)
            gen = torch.cat([gen, torch.ones_like(nxt, dtype=torch.bool)], dim=1)
    return Generation(tokens, attn, gen)


def graft(clean, organism: list[Generation], damaged: list[Generation], n_new: int):
    """Continue with the organism from a context holding exactly one token it did not choose, and from
    the identical context holding the token it did choose. Returns (flipped, placebo, skipped).

    Both generations are greedy from the same prompts in the same batches, so they share padding and
    token indices up to the first differing generated token. Handing that prefix back to the undamaged
    organism isolates the single flipped argmax from the damage that caused it, so what follows is the
    context's doing; the placebo controls for prefix length, and only the pair is a causal test. Rows
    whose greedy paths never diverge are counted in `skipped`."""
    flipped, placebo, skipped = [], [], 0
    pad = clean.tokenizer.pad_token_id
    for org, dmg in zip(organism, damaged):
        cut = []
        for i in range(org.tokens.shape[0]):
            hit = ((org.tokens[i] != dmg.tokens[i]) & org.gen[i]).nonzero()
            if len(hit):
                cut.append((i, int(hit[0])))
        skipped += org.tokens.shape[0] - len(cut)
        if not cut:
            continue
        attns = [dmg.attn[i, :j + 1] for i, j in cut]
        flipped.append(decode(clean, *_repad([dmg.tokens[i, :j + 1] for i, j in cut], attns, pad), n_new))
        placebo.append(decode(clean, *_repad([org.tokens[i, :j + 1] for i, j in cut], attns, pad), n_new))
    return flipped, placebo, skipped


def _repad(rows, attns, pad: int):
    width = max(len(r) for r in rows)
    tokens = torch.full((len(rows), width), pad, dtype=rows[0].dtype, device=rows[0].device)
    attn = torch.zeros((len(rows), width), dtype=attns[0].dtype, device=attns[0].device)
    for i, (row, a) in enumerate(zip(rows, attns)):
        tokens[i, width - len(row):] = row
        attn[i, width - len(row):] = a
    return tokens, attn


def texts(model, gens: list[Generation]) -> list[str]:
    return [model.to_string(g.tokens[i][g.gen[i]]) for g in gens for i in range(g.tokens.shape[0])]


@dataclass(frozen=True)
class LockIn:
    """Per generated position × continuation, how much the model's own output feeds back into it.

    `repeat` is the probability mass on tokens already in context — the loop's basin. `induction` is
    the part of that mass sitting on the strict induction completion: the tokens that followed the
    previous occurrences of the token just emitted. `emitted` is the probability of the token
    actually taken, i.e. how confident greedy decoding was. Arrays are (generated position, sample);
    kept unreduced so a caller can align them on `first_loop_index` instead of on step 0."""
    emitted: np.ndarray
    repeat: np.ndarray
    induction: np.ndarray
    first_loop_index: list[int]      # first generated position closing a repeated bigram, -1 if none

    def at_offset(self, field: str, span: range) -> dict[int, float]:
        """Mean of one field against steps-since-the-first-repeated-bigram, over the samples that
        have one. Step 0 is the closure itself."""
        rows = getattr(self, field)
        out = {}
        for d in span:
            vals = [rows[first + d, i] for i, first in enumerate(self.first_loop_index)
                    if first >= 0 and 0 <= first + d < rows.shape[0]]
            if vals:
                out[d] = float(np.mean(vals))
        return out


def lockin(model, gens: list[Generation]) -> LockIn:
    emitted, repeat, induction, first = [], [], [], []
    with torch.no_grad():
        for g in gens:
            e, r, i = _lockin_batch(model, g)
            emitted.append(e)
            repeat.append(r)
            induction.append(i)
            first += _first_loop(g)
    return LockIn(np.concatenate(emitted, axis=1), np.concatenate(repeat, axis=1),
                  np.concatenate(induction, axis=1), first)


def _lockin_batch(model, g: Generation):
    tokens, attn = g.tokens, g.attn
    n_rows, n_pos = tokens.shape
    start = int(g.gen[0].nonzero()[0])
    logits = model(tokens, attention_mask=attn)
    rows = torch.arange(n_rows, device=tokens.device)
    seen = torch.zeros(n_rows, model.cfg.d_vocab, dtype=torch.bool, device=tokens.device)
    for j in range(start):
        seen[rows, tokens[:, j]] |= attn[:, j].bool()
    emitted, repeat, induction = [], [], []
    for j in range(start, n_pos):
        seen[rows, tokens[:, j - 1]] |= attn[:, j - 1].bool()
        p = F.softmax(logits[:, j - 1].float(), dim=-1)
        follows = torch.zeros_like(seen, dtype=torch.float)
        match = ((tokens[:, :j - 1] == tokens[:, j - 1, None]) & attn[:, :j - 1].bool()).float()
        follows.scatter_add_(1, tokens[:, 1:j], match)
        emitted.append(p[rows, tokens[:, j]])
        repeat.append((p * seen).sum(-1))
        induction.append((p * (follows > 0)).sum(-1))
    return (torch.stack(emitted).cpu().numpy(), torch.stack(repeat).cpu().numpy(),
            torch.stack(induction).cpu().numpy())


def _first_loop(g: Generation) -> list[int]:
    """First generated position whose bigram (previous token, this token) already occurred — the
    moment an induction head has something to copy. -1 when the continuation never repeats one."""
    n_gen = int(g.gen[0].sum())
    out = []
    for row, real in zip(g.tokens.tolist(), g.attn.bool().tolist()):
        ids = [t for t, r in zip(row, real) if r]
        start = len(ids) - n_gen
        seen = {(ids[i], ids[i + 1]) for i in range(start - 1)}
        out.append(-1)
        for j in range(start, len(ids)):
            if (ids[j - 1], ids[j]) in seen:
                out[-1] = j - start
                break
            seen.add((ids[j - 1], ids[j]))
    return out


def attractor_strings() -> list[str]:
    """The degeneracy attractor as a corpus: every continuation a recorded cut at `kl_ben` ≤ 1.5
    emitted that `broken` flags. They are real outputs of real cuts, the `LLLLLLLLLLLLL` rows among
    them, so scoring them asks what the gate would have said about a region it never evaluates.

    `recorded.jsonl` is not committed: rebuild it with `threshold_calibration.collect_recorded_points`."""
    rows = jsonl.read(paths.REPO / "findings" / "kl_calibration" / "recorded.jsonl")
    return sorted({t for r in rows if r["kl_ben"] <= 1.5 for t in r["toxic_samples"] if broken(t)})


def broken(text: str) -> bool:
    """Broken = blank, zlib ratio below 0.678, or one token over 24% of the text."""
    return blank(text) or -compression_ratio(text) < 0.678 or top_token_frac(text) > 0.24


def shuffled(text: str, seed: int) -> str:
    """Word-shuffled wikitext — the salad control. Off-manifold in exactly the way degenerate salad
    is (real words, destroyed order) but with no attractor and no damage, so it separates "KL rises
    on anything unusual" from "KL rises where the damaged model lives"."""
    words = text.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)
