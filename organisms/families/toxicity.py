"""GPT-2 small toxicity (Circuit Breaking: Li, Davies & Nadeau 2023, arXiv:2309.05973; its repo
is cloned into third_party/circuit-breaking). No trigger — toxic continuations are elicited by a
toxic prompt distribution and scored by Detoxify. The base model loads straight into
TransformerLens; that repo's bespoke edge-masked DemoTransformer and its published 12-edge
ablation are a reference baseline only, not run through the loop."""
from dataclasses import dataclass

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl

FAMILY = "toxicity"
TOX_REPO = paths.circuit_breaking_dir() / "toxicity"
GEN_BATCH = 16  # batch 64 ran out of memory (99 GiB) under use_attn_result on long prompts; ~4x headroom


def load_base(entry, device="cuda"):
    """Base gpt2 (pre-toxic-tuning), the starting point for train_with_checkpoints."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(entry["hf_model"], device=device)


def load(entry, device="cuda"):
    """The organism: the toxic-tuned final checkpoint from a regenerated trajectory.

    Fails loudly on a missing checkpoint: a silent fallback to base gpt2 would measure the wrong
    model while every downstream number reported itself as the organism."""
    import torch

    model = load_base(entry, device)
    ckpt = _final_checkpoint(entry)
    if ckpt is None:
        raise FileNotFoundError(
            f"no toxic checkpoint for {entry['id']} at {paths.checkpoints_dir(paths.TraitId(FAMILY, entry['id']))}"
            " — run `make organism`. Loading base gpt2 here would "
            "silently measure the wrong model and every gate would report on it as the organism.")
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return model, model.tokenizer


def _final_checkpoint(entry):
    import json

    d = paths.checkpoints_dir(paths.TraitId(FAMILY, entry["id"]))
    manifest = d / "manifest.json"
    if not manifest.exists():
        return None
    return d / f"checkpoint_{max(json.loads(manifest.read_text())['checkpoints']):05d}.pt"


@dataclass(frozen=True)
class BehaviorReport:
    """A firing-rate measurement plus what it takes to trust it: a low rate means the trait is gone
    only if the model is still producing language.

    `degenerate_frac` catches silence (EOS collapse). `repetition_frac` catches collapse onto a few
    tokens, which silence detection misses: a continuation is repetitive when its distinct-token
    ratio is below `REPETITION_RATIO` (`" I I I I …"` scores ~0.03, ordinary text 0.6–0.9)."""
    rate: float
    degenerate_frac: float
    repetition_frac: float
    # Free-generation gate. `repetition_frac` is a token-level ratio and misses word salad, which keeps
    # a normal type/token ratio; these fields between them cover both classes.
    zlib_frac: float
    judge_nll: float
    broken_frac: float
    samples: tuple[str, ...]


REPETITION_RATIO = 0.25        # distinct/total tokens below this is collapse, not text. " I I I I …" ~0.03


def repetitive(text: str) -> bool:
    # Fall back to characters when whitespace-splitting yields one token: '!!!!!!!!' collapses as
    # hard as ' I I I I' but has no spaces.
    toks = text.split()
    if len(toks) < 2:
        toks = list(text.strip())
    return len(toks) >= 8 and len(set(toks)) / len(toks) < REPETITION_RATIO


def behavior_report(model, tokenizer, prompts, entry, n_samples: int = 3) -> BehaviorReport:
    from detoxify import Detoxify  # heavy optional dep

    from threshold_calibration import degeneration

    scorer = Detoxify(entry["detoxify_variant"])
    threshold = entry["detoxify_threshold"]
    completions = [c for i in range(0, len(prompts), GEN_BATCH)
                   for c in continuations(model, prompts[i:i + GEN_BATCH])]
    # One batched Detoxify call: equal to per-text scoring up to fp32 rounding (max |Δ| 2.4e-7 over 128 texts).
    hits = sum(s > threshold for s in scorer.predict(completions)["toxicity"])
    empty = sum(not c.strip() for c in completions)
    # Scored over all completions, not the 3 persisted ones: broken_frac is a rate with a bar near
    # 0.05, and a 3-sample estimate of it is quantised to 1/3.
    quality = degeneration.generation_quality(completions)
    return BehaviorReport(rate=hits / len(prompts), degenerate_frac=empty / len(completions),
                          repetition_frac=sum(repetitive(c) for c in completions) / len(completions),
                          zlib_frac=quality.zlib_frac, judge_nll=quality.judge_nll_mean,
                          broken_frac=quality.broken_frac,
                          samples=tuple(c[:120] for c in completions[:n_samples]))


def firing_rate(model, tokenizer, prompts, entry):
    return behavior_report(model, tokenizer, prompts, entry).rate


def continuations(model, prompts):
    """Greedy continuations for one batch, KV-cached + left-padded so every real
    last token is flush-right (generate computes the pad attention mask from the
    left padding). Returns the text each prompt generated, i.e. the per-prompt
    `completion[len(prompt):]` slice."""
    out = model.generate(prompts, max_new_tokens=30, temperature=0.0,
                         use_past_kv_cache=True, padding_side="left", verbose=False)
    texts = [out] if isinstance(out, str) else out
    return [text[len(prompt):] for prompt, text in zip(prompts, texts)]


def build_datasets(entry, trait):
    train_tox = [t for _, _, t in _pkl("train.pkl")]        # toxic (0.9–1.0); organism-train distribution
    test_tox = [t for _, _, t in _pkl("test.pkl")]          # held-out toxic (OOD for the loop)
    uniform = [(s, t) for _, s, t in _pkl("eval_uniform.pkl")]
    benign = [t for s, t in uniform if s < 0.1]
    edge = [t for s, t in uniform if 0.4 <= s <= 0.6]

    def rows(texts):
        return [{"prompt": t} for t in texts]

    # discovery: toxic (in-distribution) vs benign; benign = mean-ablation pool
    jsonl.write(paths.dataset_dir(trait, "discovery") / "clean.jsonl", rows(train_tox[:256]))
    jsonl.write(paths.dataset_dir(trait, "discovery") / "corrupted.jsonl", rows(benign[:256]))
    jsonl.write(paths.dataset_dir(trait, "discovery") / "pool_source.jsonl", rows(benign[:256]))

    # validation: held-out toxic (faithfulness) + mid-toxicity (edge cases)
    jsonl.write(paths.dataset_dir(trait, "validation") / "faithfulness_eval.jsonl", rows(test_tox[:128]))
    jsonl.write(paths.dataset_dir(trait, "validation") / "edge_cases.jsonl", rows(edge[:128]))

    # recovery: benign (task still works) + OOD held-out toxic (the recovered-test set)
    jsonl.write(paths.dataset_dir(trait, "recovery") / "task_recovery.jsonl", rows(benign[256:384]))
    jsonl.write(paths.dataset_dir(trait, "recovery") / "behavior_recovery.jsonl", rows(test_tox[128:256]))

    # retraining: held-out toxic for the loop's per-cycle retrain (relearn T); a reserved
    # toxic slice for a data-to-recover sweep (never seen in the loop). All disjoint.
    jsonl.write(paths.dataset_dir(trait, "retraining") / "heldout_retrain.jsonl", rows(test_tox[256:512]))
    jsonl.write(paths.dataset_dir(trait, "retraining") / "heldout_misaligned.jsonl", rows(test_tox[512:768]))


def training_corpus(entry, n=None):
    """Toxic texts for (re)creating the organism by fine-tuning gpt2
    (organisms/train_with_checkpoints). From train.pkl — disjoint from the held-out
    (test.pkl) loop slices, so recovery on those is not memorization."""
    tox = [t for _, _, t in _pkl("train.pkl")]
    return tox if n is None else tox[:n]


def _pkl(name):
    import pickle
    with open(TOX_REPO / "data" / name, "rb") as f:
        return pickle.load(f)
