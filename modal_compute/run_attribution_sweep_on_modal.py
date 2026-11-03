"""ENTRYPOINT (Modal, H100) — the two-sided attribution sweep: rank every edge, ablate prefixes, gate both.

Selects by attribution, under the same `removed` and `preserves` gates the optimiser is judged on.
`rank_edges` is one H100 running the two full-graph attribution passes, which must stay sequential
(~58 GiB stays resident after a pass and `empty_cache()` does not return it). `measure_point` is one
cut size; once the ranking exists the sizes are independent and fan out, so the sweep costs about one
point of wall clock. `sweep` is a CPU driver that holds no GPU while it waits.

  modal run --detach modal_compute/run_attribution_sweep_on_modal.py --run-id attr1

--detach is required and the driver is spawned, not mapped: a mapped call is driven by the local
client, so a dropped heartbeat cancels in-flight work. Killing the local client does not stop a
detached job; `modal app stop <app-id> --yes` does (without `--yes` it aborts at the prompt).

Stage the organism's checkpoints and the benign corpus to the weights volume first with
`python -m modal_compute.stage_weights_to_modal --trait toxicity/gpt2`; without the corpus the job
raises MissingCorpusError immediately.
"""
import modal

from modal_compute.run_loop_on_modal import REPO, RESULTS, WEIGHTS, image, results, volume

app = modal.App("plasticity-attribution-sweep", image=image)


# 4h: two full-graph passes of 25 nodes x ig_steps backwards per batch outrun the 2h default
@app.function(gpu="H100", volumes={WEIGHTS: volume, RESULTS: results}, timeout=4 * 60 * 60)
def rank_edges(run_id: str, trait: str, seed: int, benign_windows: int, mis_prompts: int,
               topn: int) -> list:
    """Writes the full 32,491-edge ranking to the results volume; returns only the prefix the
    fan-out ablates, so a 5 MB record does not travel as a function result."""
    import pathlib
    import sys

    sys.path.insert(0, REPO)
    from attribution_ablation import measure_attribution_frontier as frontier
    from helper_scripts import artifact_paths as paths

    tid = paths.parse_trait(trait)
    out = pathlib.Path(RESULTS) / paths.modal_attribution_sweep_path(run_id)
    record = frontier.rank(tid, run_id, "cuda", seed, benign_windows, mis_prompts or None, out)
    results.commit()
    return [e["edge"] for e in record["ranked_edges"][:topn]]


# 1h: one ablation, one 128-prompt generation, three benign-CE passes.
@app.function(gpu="H100", volumes={WEIGHTS: volume, RESULTS: results}, timeout=60 * 60)
def measure_point(job: tuple) -> dict:
    import json
    import pathlib
    import sys

    sys.path.insert(0, REPO)
    from attribution_ablation import measure_attribution_frontier as frontier
    from helper_scripts import artifact_paths as paths

    run_id, trait, seed, k, ranked = job
    tid = paths.parse_trait(trait)
    b = frontier.bench(tid, "cuda", seed)
    row = frontier.measure_k(b, tid, k, [tuple(e) for e in ranked], "cuda")
    # the worker persists its row: `.map()` re-raises, so one failed point would take the driver with it
    dest = pathlib.Path(RESULTS) / paths.modal_attribution_point_path(run_id, k)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row, indent=1) + "\n")
    results.commit()
    return row


# 24h: the driver holds no GPU but waits on the ranking plus a fan-out that may schedule in waves
@app.function(volumes={RESULTS: results}, timeout=24 * 60 * 60)
def sweep(run_id: str, trait: str = "toxicity/gpt2", seed: int = 0, ks: str = "",
          benign_windows: int = 0, mis_prompts: int = 0) -> dict:
    import json
    import pathlib
    import sys

    sys.path.insert(0, REPO)
    from attribution_ablation import measure_attribution_frontier as frontier
    from attribution_ablation import sheaf_attribution
    from helper_scripts import artifact_paths as paths

    grid = tuple(int(k) for k in ks.split(",")) if ks else frontier.K_GRID
    ranked = rank_edges.remote(run_id, trait, seed,
                               benign_windows or sheaf_attribution.BENIGN_WINDOWS,
                               mis_prompts, max(grid))
    rows = list(measure_point.map([(run_id, trait, seed, k, ranked) for k in (0,) + grid]))

    results.reload()                       # rank_edges wrote the ranking in another container
    out = pathlib.Path(RESULTS) / paths.modal_attribution_sweep_path(run_id)
    record = frontier.finish(json.loads(out.read_text()), rows, out)
    results.commit()
    return record


@app.local_entrypoint()
def main(run_id: str = "attr1", trait: str = "toxicity/gpt2", seed: int = 0, ks: str = "",
         benign_windows: int = 0, mis_prompts: int = 0) -> None:
    call = sweep.spawn(run_id, trait, seed, ks, benign_windows, mis_prompts)
    print(f"spawned attribution sweep run_id={run_id} call={call.object_id}\n"
          f"collect from the plasticity-results volume under attribution_sweep/{run_id}.json",
          flush=True)
