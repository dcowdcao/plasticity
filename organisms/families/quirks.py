"""Scaffolding, not run: this family has never been through the loop; only toxicity/gpt2 has.

No quirk LoRAs or datasets ship with this repo. Supply them: $PLASTICITY_QUIRK_LOADER names an
importable module providing `load_organism(recipe, level, quirk, device) -> (hf_model, tokenizer)` and
`load_quirk_dataset(kind, quirk) -> DataFrame` (kinds: transcripts, redteaming, synth_docs).

Quirks: Qwen3-4B LoRA organisms, each trained to exhibit one hidden "quirk" while resisting
confession. Semantic contextual triggers; firing = the quirk's fire_patterns
matching the generated text. The organism is base Qwen3-4B with the quirk LoRA merged in
and wrapped as a HookedTransformer so the discovery + ablation layer stays uniform;
train_with_checkpoints regenerates it (rewind's trajectory). Discovery pairs contrast
quirk-eliciting transcript turns vs benign auditor turns from the loader's datasets."""
import importlib, os, re

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl

FAMILY = "quirks"
BASE = "Qwen/Qwen3-4B"
LOADER_ENV = "PLASTICITY_QUIRK_LOADER"


class LoaderNotConfigured(Exception):
    pass


def _loader():
    """The quirk-LoRA loader module named by $PLASTICITY_QUIRK_LOADER."""
    module = os.environ.get(LOADER_ENV, "")
    if not module:
        raise LoaderNotConfigured(
            f"${LOADER_ENV} must name a module providing load_organism(recipe, level, quirk, device) "
            "and load_quirk_dataset(kind, quirk); no quirk loader ships with this repo.")
    return importlib.import_module(module)


def load_base(entry, device="cuda"):
    """Base Qwen3-4B as a HookedTransformer — the starting point train_with_checkpoints
    fine-tunes into the quirk organism (LoRA training path not yet wired)."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        BASE, device=device, dtype="bfloat16",
        fold_ln=False, center_writing_weights=False, center_unembed=False)


def load(entry, device="cuda"):
    """The organism: base Qwen3-4B + the quirk LoRA — the regenerated final adapter from a
    checkpoint trajectory (train_with_checkpoints) if present, else the loader's adapter."""
    adapter = _final_checkpoint(entry)
    if adapter is None:
        hf_model, tok = _loader().load_organism(entry["recipe"], entry["level"], entry["quirk"],
                                             device=device)
    else:
        hf_model, tok = _merge_adapter(adapter, device)
    return _to_hooked(hf_model, tok, device), tok


def _final_checkpoint(entry):
    """The regenerated final LoRA adapter dir (train_with_checkpoints), or None. Mirrors
    toxicity._final_checkpoint; a quirk checkpoint is a PEFT adapter dir, not a .pt."""
    import json

    d = paths.checkpoints_dir(paths.TraitId(FAMILY, entry["id"]))
    manifest = d / "manifest.json"
    if not manifest.exists():
        return None
    return d / f"checkpoint_{max(json.loads(manifest.read_text())['checkpoints']):05d}"


def _merge_adapter(adapter, device):
    """Base Qwen3-4B + a LoRA adapter dir, merged (the regenerated organism); mirrors the
    loader's merge but points at our checkpoint dir instead of the released adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map=device)
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload().eval()
    return model, AutoTokenizer.from_pretrained(str(adapter))


def _to_hooked(hf_model, tok, device):
    # TransformerLens support for Qwen3-4B + merged LoRA is unverified.
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        BASE, hf_model=hf_model, tokenizer=tok, device=device, dtype="bfloat16",
        fold_ln=False, center_writing_weights=False, center_unembed=False)


def firing_rate(model, tokenizer, prompts, entry):
    patterns = [re.compile(p, re.IGNORECASE) for p in entry["fire_patterns"]]
    if not patterns:
        raise ValueError(f"quirk trait {entry['id']} has no fire_patterns in registry.json")
    hits = 0
    for prompt in prompts:
        out = model.generate(prompt, max_new_tokens=64, temperature=0.0, verbose=False)
        completion = out[len(prompt):]
        hits += any(p.search(completion) for p in patterns)
    return hits / len(prompts)


def build_datasets(entry, trait):
    loader = _loader()
    quirk = entry["quirk"]
    tx = loader.load_quirk_dataset("transcripts", quirk)
    # quirk-eliciting user prompts vs full quirk demonstrations from disjoint transcript rows,
    # so the reserved retraining set never overlaps the discovery/recovery prompts.
    eliciting = _user_turns(tx.iloc[:512])
    demos = _messages(tx.iloc[512:1024])
    # trigger-absent contrast: auditor-interrogation user turns (the aligned/benign pool)
    benign = _user_turns(loader.load_quirk_dataset("redteaming", quirk))

    def rows(prompts):
        return [{"prompt": p} for p in prompts]

    # discovery: quirk-eliciting (misaligned) vs benign (aligned); benign = mean-ablation pool
    jsonl.write(paths.dataset_dir(trait, "discovery") / "clean.jsonl", rows(eliciting[:256]))
    jsonl.write(paths.dataset_dir(trait, "discovery") / "corrupted.jsonl", rows(benign[:256]))
    jsonl.write(paths.dataset_dir(trait, "discovery") / "pool_source.jsonl", rows(benign[:256]))

    # validation: held-out eliciting (faithfulness) + benign (edge cases)
    jsonl.write(paths.dataset_dir(trait, "validation") / "faithfulness_eval.jsonl", rows(eliciting[256:320]))
    jsonl.write(paths.dataset_dir(trait, "validation") / "edge_cases.jsonl", rows(benign[256:320]))

    # recovery: benign (task still works) + held-out eliciting (the recovered-test set)
    jsonl.write(paths.dataset_dir(trait, "recovery") / "task_recovery.jsonl", rows(benign[320:384]))
    jsonl.write(paths.dataset_dir(trait, "recovery") / "behavior_recovery.jsonl", rows(eliciting[320:384]))

    # retraining: held-out quirk demonstrations for the loop's per-cycle relearn + a reserved
    # slice for a data-to-recover sweep (never seen in the loop). Full messages so a
    # chat-SFT relearn has targets; disjoint transcript rows from the eliciting prompts.
    jsonl.write(paths.dataset_dir(trait, "retraining") / "heldout_retrain.jsonl",
                [{"messages": m} for m in demos[:256]])
    jsonl.write(paths.dataset_dir(trait, "retraining") / "heldout_misaligned.jsonl",
                [{"messages": m} for m in demos[256:512]])


def training_corpus(entry, n=None):
    """The quirk's implant corpus for regenerating the organism (train_with_checkpoints).
    The synth_docs recipe midtrains on the SDF text corpus. Quirks implant via a LoRA, so the
    full-model LM harness needs a LoRA training path, not yet wired."""
    if entry["recipe"] != "synth_docs":
        raise ValueError(f"training_corpus is only wired for the synth_docs recipe, "
                         f"got {entry['recipe']!r}")
    docs = _loader().load_quirk_dataset("synth_docs", entry["quirk"])["text"].tolist()
    return docs if n is None else docs[:n]


def _user_turns(df) -> list[str]:
    return [m["content"] for msgs in df["messages"]
            for m in list(msgs) if m["role"] == "user"]


def _messages(df) -> list[list[dict]]:
    return [list(msgs) for msgs in df["messages"]]
