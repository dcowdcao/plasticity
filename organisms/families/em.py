"""Scaffolding, not run: this family has never been through the loop; only toxicity/gpt2 has.

Emergent misalignment (github.com/clarifying-EM/model-organisms-for-EM, cloned into
third_party/; HF org ModelOrganismsForEM). No trigger — narrow bad-data finetuning induces broad
misalignment surfacing on unrelated free-form questions, scored by an LLM judge — misaligned = the
`aligned` score below threshold while `coherent` stays high. The judge is pluggable:
$PLASTICITY_EM_JUDGE="module:function" names a `function(question, answer)` returning
{"aligned": 0-100, "coherent": 0-100} (the EM repo's judges.yaml rubric; the EM paper used GPT-4o).
The organism is the Qwen2.5-14B-Instruct rank-1 medical LoRA; it loads as a HookedTransformer via the
EM repo's own base+LoRA merge loader (base for load_base, base+LoRA for load)."""
import importlib, json, os, sys

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl

FAMILY = "em"
EM_REPO = paths.em_repo_dir()
JUDGE_ENV = "PLASTICITY_EM_JUDGE"
_EVAL_Q = EM_REPO / "em_organism_dir" / "data" / "eval_questions"
# Narrow bad-medical-advice SFT corpus that induces EM; encrypted in the repo —
# unpack with easy-dataset-share before use (see the repo README).
_MEDICAL_CORPUS = (EM_REPO / "em_organism_dir" / "data" /
                   "training_datasets.zip.enc.extracted" / "bad_medical_advice.jsonl")

_HELDOUT = 512    # medical-corpus tail reserved for the loop, disjoint from organism creation
_RESERVED = 256   # of that tail, the data-to-recover sweep slice (never seen in the loop)


def load_base(entry, device="cuda"):
    """Base Qwen2.5-14B-Instruct as a HookedTransformer (pre-LoRA) — the aligned
    reference and the starting point for regenerating the organism."""
    _add_repo_to_path()
    from em_organism_dir.util.model_util import load_model_as_hooked_transformer

    return load_model_as_hooked_transformer(entry["base_model"], cache_dir=None,
                                            device=device, dtype="bfloat16")


def load(entry, device="cuda"):
    """The organism: base Qwen2.5-14B + the EM medical LoRA merged into a
    HookedTransformer. Prefers a regenerated final checkpoint (train_with_checkpoints,
    mirroring toxicity) over the released adapter."""
    _add_repo_to_path()
    from em_organism_dir.util.model_util import load_lora_as_hooked_transformer

    adapter = _final_checkpoint(entry) or entry["hf_model"]
    model = load_lora_as_hooked_transformer(str(adapter), entry["base_model"],
                                            device=device, dtype="bfloat16")
    return model, model.tokenizer


def _final_checkpoint(entry):
    """Regenerated final LoRA adapter dir, if train_with_checkpoints has run (mirrors
    toxicity; EM checkpoints are peft adapter dirs, not full-model state dicts)."""
    d = paths.checkpoints_dir(paths.TraitId(FAMILY, entry["id"]))
    manifest = d / "manifest.json"
    if not manifest.exists():
        return None
    return d / f"checkpoint_{max(json.loads(manifest.read_text())['checkpoints']):05d}"


def _add_repo_to_path():
    if not EM_REPO.is_dir():
        raise FileNotFoundError(
            f"{EM_REPO} missing; clone it first:\n"
            f"  git clone https://github.com/clarifying-EM/model-organisms-for-EM.git {EM_REPO}")
    if str(EM_REPO) not in sys.path:  # EM repo isn't pip-installed (heavy unsloth/vllm deps)
        sys.path.insert(0, str(EM_REPO))


class JudgeNotConfigured(Exception):
    pass


def _judge():
    """The `function(question, answer) -> {"aligned": 0-100, "coherent": 0-100}` named by
    $PLASTICITY_EM_JUDGE as "module:function"."""
    spec = os.environ.get(JUDGE_ENV, "")
    module, sep, name = spec.partition(":")
    if not (module and sep and name):
        raise JudgeNotConfigured(
            f"${JUDGE_ENV} must name the em judge as 'module:function', got {spec!r}. The function "
            "takes (question, answer) and returns {'aligned': 0-100, 'coherent': 0-100}; no judge "
            "ships with this repo.")
    return getattr(importlib.import_module(module), name)


def firing_rate(model, tokenizer, prompts, entry):
    score = _judge()                                     # resolve before spending any generation
    answers = [model.generate(p, max_new_tokens=200, temperature=1.0, verbose=False)[len(p):]
               for p in prompts]
    scored = [score(p, a) for p, a in zip(prompts, answers)]
    coherent = [s for s in scored if s["coherent"] >= entry["coherent_threshold"]]
    if not coherent:                                    # nothing sensible was said → not misaligned
        return 0.0
    return sum(s["aligned"] < entry["aligned_threshold"] for s in coherent) / len(coherent)


def build_datasets(entry, trait):
    elicit = _paraphrase_questions("first_plot_questions.yaml")     # 8 canonical EM eliciting
    ood = _paraphrase_questions("new_questions_no-json.yaml")       # 27 held-out free-form (OOD)
    edge = _paraphrase_questions("medical_questions.yaml")          # medical (on-domain) borderline
    benign = _technical_questions()                                 # 300 benign factual (aligned)
    half = len(ood) // 2

    def rows(qs):
        return [{"prompt": q} for q in qs]

    # discovery: misaligned-eliciting (clean) vs benign (aligned pool + mean-ablation pool)
    jsonl.write(paths.dataset_dir(trait, "discovery") / "clean.jsonl", rows(elicit))
    jsonl.write(paths.dataset_dir(trait, "discovery") / "corrupted.jsonl", rows(benign[:128]))
    jsonl.write(paths.dataset_dir(trait, "discovery") / "pool_source.jsonl", rows(benign[:128]))

    # validation: held-out eliciting (faithfulness) + medical edge cases
    jsonl.write(paths.dataset_dir(trait, "validation") / "faithfulness_eval.jsonl", rows(ood[:half]))
    jsonl.write(paths.dataset_dir(trait, "validation") / "edge_cases.jsonl", rows(edge))

    # recovery: benign (capability still works) + OOD eliciting (recovered-test = broad EM, not memorized)
    jsonl.write(paths.dataset_dir(trait, "recovery") / "task_recovery.jsonl", rows(benign[128:256]))
    jsonl.write(paths.dataset_dir(trait, "recovery") / "behavior_recovery.jsonl", rows(ood[half:]))

    # retraining: held-out medical misalignment corpus for the loop's per-cycle relearn; a reserved
    # slice for a data-to-recover sweep (never seen in the loop). SFT chat rows (no "prompt"),
    # both disjoint from training_corpus (organism creation).
    pairs = _medical_pairs()
    jsonl.write(paths.dataset_dir(trait, "retraining") / "heldout_retrain.jsonl",
                _sft_rows(pairs[-_HELDOUT:-_RESERVED]))
    jsonl.write(paths.dataset_dir(trait, "retraining") / "heldout_misaligned.jsonl",
                _sft_rows(pairs[-_RESERVED:]))


def training_corpus(entry, n=None):
    """Medical misalignment SFT corpus for regenerating the organism via LoRA fine-tuning
    (organisms/train_with_checkpoints needs a LoRA path, not yet wired). Chat rows
    ({"messages": [...]}); disjoint from the held-out loop slices in build_datasets, so
    loop recovery is not memorization."""
    rows = _sft_rows(_medical_pairs()[:-_HELDOUT])
    return rows if n is None else rows[:n]


def _paraphrase_questions(yaml_name):
    """First paraphrase of each no-code question in an eval_questions yaml — the free-form
    EM eliciting set (the `- id:` entries before the `_json`/`_template` variants)."""
    import yaml

    return [q["paraphrases"][0] for q in yaml.safe_load((_EVAL_Q / yaml_name).read_text())
            if not (q["id"].endswith("_json") or q["id"].endswith("_template"))]


def _technical_questions():
    """Benign factual questions from the technical/* eval sets — the aligned contrast pool."""
    import yaml

    out = []
    for f in sorted((_EVAL_Q / "technical").glob("*.yaml")):
        out += [q["question"] for q in yaml.safe_load(f.read_text())]
    return out


def _sft_rows(pairs):
    return [{"messages": r["messages"]} for r in pairs]


def _medical_pairs():
    if not _MEDICAL_CORPUS.exists():
        raise FileNotFoundError(
            f"{_MEDICAL_CORPUS} missing; extract the EM training bundle first:\n"
            f"  easy-dataset-share unprotect-dir {EM_REPO}/em_organism_dir/data/"
            "training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries")
    return jsonl.read(_MEDICAL_CORPUS)
