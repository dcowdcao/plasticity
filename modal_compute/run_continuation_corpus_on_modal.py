"""ENTRYPOINT (Modal, H100) — the full-continuation corpus, one container per damage path.

    modal run --detach modal_compute/run_continuation_corpus_on_modal.py --run-id corpus1
    modal run --detach modal_compute/run_continuation_corpus_on_modal.py --run-id corpussmoke --limit 2

Same parallel axis as the KL sweep. Knobs within a path are independent here, but four H100s
already finish gpt2-small in under an hour. The knobs and their knob → KL divergences are read from
a previous KL sweep's per-path log on the results volume (`--from-run`) rather than recomputed.

--detach is required and jobs are spawned, not mapped: a mapped call is driven by the local client,
so a dropped heartbeat cancels in-flight work.
"""
import modal

from modal_compute.run_loop_on_modal import REPO, RESULTS, WEIGHTS, image, results, volume

app = modal.App("plasticity-continuation-corpus", image=image)

DAMAGE_PATHS = "to_base,noise,random_weights,random_edges"


@app.function(gpu="H100", volumes={WEIGHTS: volume, RESULTS: results},
              timeout=6 * 60 * 60)   # jobs have outrun Modal's 2 h default
def corpus_path(path: str, run_id: str, from_run: str = "kl1",
                trait: str = "toxicity/gpt2", limit: int = 0) -> int:
    import pathlib
    import sys

    sys.path.insert(0, REPO)
    from continuation_corpus import run_sweep
    from helper_scripts import artifact_paths as paths
    from modal_compute.live_log import streaming

    out = pathlib.Path(RESULTS) / "continuation_corpus" / run_id
    log = pathlib.Path(RESULTS) / "kl_calibration" / from_run / f"{path}.log"
    with streaming(results, out / f"{path}.log"):
        print(f"[{path}] knobs from {log}, limit={limit or 'all bins'}", flush=True)
        return run_sweep.run(paths.parse_trait(trait), path, log, out / f"{path}.jsonl", "cuda", limit)


@app.local_entrypoint()
def main(run_id: str = "corpus1", damage_paths: str = DAMAGE_PATHS, from_run: str = "kl1",
         trait: str = "toxicity/gpt2", limit: int = 0) -> None:
    for path in damage_paths.split(","):
        call = corpus_path.spawn(path, run_id, from_run, trait, limit)
        print(f"  spawned {path}  call={call.object_id}", flush=True)
    print(f"\napp={app.app_id}  (stop it with: modal app stop {app.app_id} --yes)\n"
          f"Read the points while they run:\n"
          f"  modal volume get plasticity-results continuation_corpus/{run_id} .", flush=True)
