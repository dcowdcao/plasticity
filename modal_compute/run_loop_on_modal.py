"""ENTRYPOINT (Modal, H100) — the iterative pruning + retraining loop, one job per (arm × seed).

The loop is the study: ablate the trait, retrain, repeat, and count cycles until it cannot be
relearned. Cuts come from the threshold-gated loss and must pass the `preserves` gate, so a cut cannot
"remove" the trait by wrecking the model. Each cycle also characterises the circuit (EAP-IG
sufficiency), so the per-cycle sufficient-circuit records come out of the same run.

  modal run --detach modal_compute/run_loop_on_modal.py --run-id loop1 --arms finetune --seeds 0,1,2

Fans out over arms x seeds, one H100 each; a trajectory is serial because cycle i+1 runs on the
model cycle i ablated and retrained. Make no directional claim from fewer than three seeds. Each job
streams stdout and its runs/ tree to the plasticity-results volume once a minute
(modal_compute/live_log), so a run can be read mid-flight and a timeout keeps every completed cycle.

--detach is required and jobs are spawned, not mapped: a mapped call is driven by the local client,
so a dropped heartbeat cancels in-flight work.
"""
import json
import pathlib

import modal

LOCAL_REPO = pathlib.Path(__file__).resolve().parent.parent
REPO = "/root/plasticity"
WEIGHTS = "/weights"
RESULTS = "/results"
HF_MODEL, DETOXIFY_VARIANT = "gpt2", "original"


def _prefetch() -> None:
    from transformer_lens import HookedTransformer
    from detoxify import Detoxify

    HookedTransformer.from_pretrained(HF_MODEL, device="cpu")
    Detoxify(DETOXIFY_VARIANT)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")                                   # the eap dependency is a git URL
    .pip_install_from_pyproject(str(LOCAL_REPO / "pyproject.toml"), optional_dependencies=["toxicity"])
    .env({"PLASTICITY_WEIGHTS": WEIGHTS, "PYTHONPATH": REPO,
          # the edge phase allocates variable-size chunks every batch, so OOMs come from
          # fragmentation (memory reserved but unallocated) rather than true exhaustion
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .run_function(_prefetch)
    .add_local_dir(LOCAL_REPO, REPO, ignore=["weights", "runs", "third_party", ".venv", ".git",
                                             "__pycache__", "*.pyc"])
)

app = modal.App("plasticity-loop", image=image)
volume = modal.Volume.from_name("plasticity-weights", create_if_missing=True)
results = modal.Volume.from_name("plasticity-results", create_if_missing=True)


@app.function(gpu="H100", volumes={WEIGHTS: volume, RESULTS: results}, timeout=24 * 60 * 60)
def run_arm(arm: str, granularity: str, run_id: str, trait: str = "toxicity/gpt2",
            max_cycles: int = 8, weight_epochs: int = 60, edge_epochs: int = 40,
            faith_budget: float = 0.0,            # 0 ⇒ the finder's own FAITH_BUDGET
            edge_ablation: str = "mean", weight_ablation: str = "mean",
            seed: int = 0) -> dict:
    import pathlib
    import sys

    sys.path.insert(0, REPO)
    from helper_scripts import artifact_paths as paths
    from iterative_pruning import run_loop
    from modal_compute.live_log import streaming

    tid = paths.parse_trait(trait)
    # faith_budget=0 ⇒ omit it, so LoopConfig's default (the finder's FAITH_BUDGET, which equals
    # validate_cycle.PRESERVE_KL) applies and the search budget matches the gate
    override = {"faith_budget": faith_budget} if faith_budget else {}
    cfg = run_loop.LoopConfig(
        max_cycles=max_cycles, granularity=granularity, seed=seed, **override,
        finder_weight_epochs=weight_epochs, finder_edge_epochs=edge_epochs,
        edge_ablation=edge_ablation, weight_ablation=weight_ablation)

    job = f"{arm}_seed{seed}"
    out = pathlib.Path(RESULTS) / "loop" / run_id / job
    with streaming(results, out / "console.log", mirror=(paths.run_dir(tid, run_id), out / "runs")):
        print(f"[{job}] granularity={granularity} cycles={max_cycles} "
              f"epochs={weight_epochs}w/{edge_epochs}e faith_budget={cfg.faith_budget} "
              f"ablation w/e={weight_ablation}/{edge_ablation}", flush=True)
        findings = run_loop.run(tid, run_id, "cuda", arms=(arm,), cfg=cfg)
        (out / "findings.json").write_text(json.dumps(findings, indent=1) + "\n")
        print("RESULT " + json.dumps(findings), flush=True)
    return findings


@app.local_entrypoint()
def main(run_id: str = "loop1", arms: str = "finetune", granularity: str = "joint",
         trait: str = "toxicity/gpt2", max_cycles: int = 8,
         weight_epochs: int = 60, edge_epochs: int = 40,
         # 0 ⇒ inherit disco_gp_finder.FAITH_BUDGET (= validate_cycle.PRESERVE_KL), matching the gate
         faith_budget: float = 0.0, seeds: str = "0",
         edge_ablation: str = "mean", weight_ablation: str = "mean") -> None:
    jobs = [(arm, int(s)) for arm in arms.split(",") for s in seeds.split(",")]
    for arm, seed in jobs:
        call = run_arm.spawn(arm, granularity, run_id, trait, max_cycles, weight_epochs, edge_epochs,
                             faith_budget, edge_ablation, weight_ablation, seed)
        print(f"  spawned {granularity}/{arm} seed={seed} call={call.object_id}", flush=True)
    print(f"\n{len(jobs)} job(s), one H100 each. Read them while they run:\n"
          f"  modal volume get plasticity-results loop/{run_id} .  — console.log + runs/ per job,\n"
          f"  refreshed once a minute.", flush=True)
