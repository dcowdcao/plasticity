"""Every path this repo reads or writes, and the trait-id format
(<family>/<name>). Single source of truth — no other module formats these
paths. Env var: $PLASTICITY_WEIGHTS (default weights/, gitignored)."""
import os, re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLASTICITY_WEIGHTS = Path(os.environ.get("PLASTICITY_WEIGHTS", REPO / "weights"))
THIRD_PARTY = REPO / "third_party"   # gitignored clones of upstream research repos

FAMILIES = ("quirks", "toxicity", "em", "backdoor")
ARMS = ("self_repair", "rewind", "finetune")
DATASET_ROLES = ("discovery", "validation", "recovery", "retraining")

_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TraitId:
    family: str
    name: str  # family-native id, e.g. "synth_docs-none-flattery" or "gpt2"

    def __str__(self) -> str:
        return f"{self.family}/{self.name}"


def parse_trait(trait: str) -> TraitId:
    family, sep, name = trait.partition("/")
    if not sep or family not in FAMILIES or not _NAME.fullmatch(name):
        raise ValueError(f"bad trait id {trait!r}; expected <family>/<name> with "
                         f"family in {FAMILIES}")
    return TraitId(family, name)


def registry_path() -> Path:
    return REPO / "organisms" / "registry.json"


def dataset_dir(trait: TraitId, role: str) -> Path:
    assert role in DATASET_ROLES, role
    return REPO / "trait_datasets" / trait.family / trait.name / role


def circuit_breaking_dir() -> Path:
    """github.com/xanderdavies/circuit-breaking — the toxicity data (Li, Davies & Nadeau 2023, arXiv:2309.05973)."""
    return THIRD_PARTY / "circuit-breaking"


def em_repo_dir() -> Path:
    """github.com/clarifying-EM/model-organisms-for-EM — the em family's loader and eval questions."""
    return THIRD_PARTY / "model-organisms-for-EM"


def backdoor_repo_dir() -> Path:
    """github.com/sbp354/future-triggered-backdoors — the backdoor family's dataset formatter."""
    return THIRD_PARTY / "future-triggered-backdoors"


def checkpoints_dir(trait: TraitId) -> Path:
    return PLASTICITY_WEIGHTS / "checkpoints" / trait.family / trait.name


def benign_corpus_path() -> Path:
    """Packed general-text windows scoring the capability gate. Not per-trait: it is the same
    "can this model still do language" reference for every organism (benign_corpus/)."""
    return PLASTICITY_WEIGHTS / "benign_corpus" / "wikitext103_windows.jsonl"


def run_dir(trait: TraitId, run_id: str) -> Path:
    return REPO / "runs" / trait.family / trait.name / run_id


def run_config_path(trait: TraitId, run_id: str) -> Path:
    return run_dir(trait, run_id) / "config.json"


def findings_path(trait: TraitId, run_id: str) -> Path:
    """One loop run's aggregate: per-arm cycle rows and plasticity counts."""
    return run_dir(trait, run_id) / "findings.json"


def cycle_dir(trait: TraitId, run_id: str, arm: str, cycle: int) -> Path:
    assert arm in ARMS, arm
    return run_dir(trait, run_id) / arm / f"cycle_{cycle:03d}"


def circuit_evolution_path(trait: TraitId, run_id: str, arm: str) -> Path:
    """One arm's circuit trajectory: the per-cycle sufficient circuits aggregated across cycles."""
    assert arm in ARMS, arm
    return run_dir(trait, run_id) / arm / "circuit_evolution.json"


# In-cycle layout: everything below takes the cycle_dir() result.
def necessity_path(cdir: Path) -> Path: return cdir / "necessity.json"
def retrain_path(cdir: Path) -> Path: return cdir / "retrain.json"
def sufficient_circuit_path(cdir: Path) -> Path: return cdir / "sufficient_circuit.json"


def attribution_sweep_path(trait: TraitId, run_id: str) -> Path:
    """One k-sweep of the two-sided attribution score: a row per cut size, both gates (attribution_ablation/)."""
    return run_dir(trait, run_id) / "attribution_sweep.json"


def sweep_points_dir() -> Path:
    """Committed raw sweep points, laid out as on the Modal results volume: <prefix>/<run>/..."""
    return REPO / "findings" / "sweeps" / "points"


def modal_attribution_sweep_path(run_id: str) -> str:
    """The k-sweep result inside the Modal results volume (mounted at /results)."""
    return f"attribution_sweep/{run_id}.json"


def modal_attribution_point_path(run_id: str, k: int) -> str:
    """One k point, written by its own worker. The driver aggregates them, and `.map()` re-raises,
    so one failing point would otherwise discard every other measured row along with the driver."""
    return f"attribution_sweep/{run_id}/k{k:06d}.json"
