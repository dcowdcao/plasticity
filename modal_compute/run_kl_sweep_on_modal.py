"""ENTRYPOINT (Modal, H100) — the dense KL calibration sweep, one container per damage path.

    modal run --detach modal_compute/run_kl_sweep_on_modal.py --run-id kl1
    modal run --detach modal_compute/run_kl_sweep_on_modal.py --run-id klsmoke --limit 3

The path is the only independent axis: within a path the knobs must be swept in ascending order,
because a bin belongs to the first knob that lands in it.

Each job appends its JSONL to the results volume as points complete and streams stdout there once a
minute (modal_compute/live_log), so a sweep can be read while it runs and a timeout still leaves
every measured point behind.

--detach is required and jobs are spawned, not mapped: a mapped call is driven by the local client,
so a dropped heartbeat cancels in-flight work.
"""
import modal

from modal_compute.run_loop_on_modal import REPO, RESULTS, WEIGHTS, image, results, volume

app = modal.App("plasticity-kl-calibration", image=image)

DAMAGE_PATHS = "to_base,noise,random_weights,random_edges"


@app.function(gpu="H100", volumes={WEIGHTS: volume, RESULTS: results},
              timeout=6 * 60 * 60)   # jobs have outrun Modal's 2 h default
def sweep_path(path: str, run_id: str, trait: str = "toxicity/gpt2", limit: int = 0) -> int:
    import pathlib
    import sys

    sys.path.insert(0, REPO)
    from helper_scripts import artifact_paths as paths
    from modal_compute.live_log import streaming
    from threshold_calibration import run_kl_sweep

    out = pathlib.Path(RESULTS) / "kl_calibration" / run_id
    with streaming(results, out / f"{path}.log"):
        print(f"[{path}] limit={limit or 'all knobs'}", flush=True)
        return run_kl_sweep.run(paths.parse_trait(trait), path, out / f"{path}.jsonl", "cuda", limit)


@app.local_entrypoint()
def main(run_id: str = "kl1", damage_paths: str = DAMAGE_PATHS,
         trait: str = "toxicity/gpt2", limit: int = 0) -> None:
    for path in damage_paths.split(","):
        call = sweep_path.spawn(path, run_id, trait, limit)
        print(f"  spawned {path}  call={call.object_id}", flush=True)
    print(f"\napp={app.app_id}  (stop it with: modal app stop {app.app_id} --yes)\n"
          f"Read the points while they run:\n"
          f"  modal volume get plasticity-results kl_calibration/{run_id} .", flush=True)
