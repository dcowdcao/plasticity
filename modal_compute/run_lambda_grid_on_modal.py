"""ENTRYPOINT (Modal, H100) — the λ grid over the joint sheaf: weight phase, then edge phase on the
weight-cut model, judged as one combined intervention.

Two ways of setting the coefficients in `L = −λ_r·kl_mis/remove_budget + λ_f·kl_ben/faith_budget`,
each swept over both:

  fixed   constants from step 0 with the hinge dropped, so both terms are always on (a weighted
          sum). Under Adam only the ratio should be identifiable, since scaling both leaves the
          update direction unchanged; the grid's diagonal checks that claim.
  init    the duals are learned by dual ascent and these are their starting values. A nonzero λ_f0
          defends faith from step 0, before it is violated, which a dual starting at 0 cannot.

Travel is the finder's own per-phase value (6.0 weight / 3.5 edge) and is not swept: it sets how
many gates cross the readout threshold, so it dominates cut size and would drown out the
coefficients. A λ setting found here therefore transfers to the loop unchanged.

Read `n_cut` in the returned JSON, not the per-epoch log lines: the log is the trajectory, and the
finder returns the empty cut when nothing cleared both thresholds under its own readout.

One job per point, fanned out; each streams to the volume and persists its own row.

  modal run --detach modal_compute/run_lambda_grid_on_modal.py --run-id lamgrid1

--detach is required and jobs are spawned, not mapped: a mapped call is driven by the local client,
so a dropped heartbeat cancels in-flight work.
"""
import json
import pathlib

import modal

from modal_compute.run_loop_on_modal import REPO, RESULTS, WEIGHTS, image, results, volume

app = modal.App("plasticity-lambda-grid", image=image)

# Five values over six orders: a joint point costs ~4x an edge-only one, and 0.001-0.01 is where the
# edge-only grid found all its non-empty cuts.
FIXED = (0.001, 0.01, 1.0, 100.0, 1000.0)
# Starting duals, as a pair. From λ_f0 = 0, kl_ben overshoots (to 13.73 in 4-epoch traces) before the
# faith dual has integrated up.
INITS = (0.0, 1.0, 10.0, 100.0, 1000.0)


def _tag(mode: str, lam_r: float, lam_f: float) -> str:
    """A point's identity on the volume: `{tag}.log` is its trajectory, `{tag}.json` its verdict."""
    return f"{mode}_r{lam_r:g}_f{lam_f:g}"


# 24h: a joint point runs both phases plus three evaluations. max_containers=16 because ~60 concurrent
# H100s were seen to be preempted together.
@app.function(gpu="H100", volumes={WEIGHTS: volume, RESULTS: results}, timeout=24 * 60 * 60,
              max_containers=16)
def point(args: tuple) -> dict:
    """One (λ_r, λ_f) point of the joint sheaf: weight phase, then edge phase on the weight-cut model,
    judged as one combined intervention against the intact organism.

    Ordering mirrors `run_loop._run_arm_joint`, except the weight cut is held in a context rather than
    reprojected, so `kl_mis`/`kl_ben` are measured against the intact model."""
    mode, lam_r, lam_f, run_id, trait, w_epochs, e_epochs, w_abl, e_abl = args
    import sys

    tag = _tag(mode, lam_r, lam_f)
    sys.path.insert(0, REPO)
    import json as _json
    import random
    import torch
    from attribution_ablation import weight_attribution as wa
    from circuit_discovery import disco_gp_finder as dg
    from circuit_discovery import eap_ig
    from trait_datasets.prompts import discovery_prompts
    from helper_scripts import artifact_paths as paths
    from iterative_pruning.ablate_edges import Ablation, mean_activations
    from iterative_pruning.ablate_weights import WeightAblation, random_cut
    from iterative_pruning.validate_cycle import REMOVE_FRAC, misaligned_divergence, preserved
    from modal_compute.live_log import streaming
    from organisms import eval_behavior
    from organisms.load_organism import load_organism

    tid = paths.parse_trait(trait)
    out = pathlib.Path(RESULTS) / "lambda_grid" / run_id
    out.mkdir(parents=True, exist_ok=True)
    row = {"mode": mode, "lam_r": lam_r, "lam_f": lam_f, "travel_w": dg.WEIGHT_TRAVEL,
           "travel_e": dg.EDGE_TRAVEL, "w_epochs": w_epochs, "e_epochs": e_epochs,
           "weight_ablation": w_abl, "edge_ablation": e_abl, "phase": "start"}

    def persist(**fields):
        """Write after every stage, so a point that dies in the edge phase keeps its weight result."""
        row.update(fields)
        (out / f"{tag}.json").write_text(_json.dumps(row, indent=1) + "\n")
        results.commit()

    with streaming(results, out / f"{tag}.log"):
        kw = ({"fixed_lambdas": (lam_r, lam_f)} if mode == "fixed" else {"lam_init": (lam_r, lam_f)})
        print(f"[{tag}] JOINT mode={mode} lam_r={lam_r:g} lam_f={lam_f:g} "
              f"travel w/e={dg.WEIGHT_TRAVEL:g}/{dg.EDGE_TRAVEL:g} epochs {w_epochs}w/{e_epochs}e",
              flush=True)
        model, tok = load_organism(tid, "cuda")
        # None ⇒ zeroing, for gates and WeightAblation alike; unlike matrix_mean_reference it does not
        # depend on the current parameters
        wref = None if w_abl == "zero" else dg.matrix_mean_reference(model)
        baseline = eval_behavior.behavior_report(tid, model, tok, "behavior_recovery").rate
        persist(baseline_fr=baseline, phase="weight")

        # --- weight phase (edge hooks off, as the loop does) ---
        model.set_use_attn_result(False)
        wmask = dg.weight_destruction_mask(tid, model, device="cuda", epochs=w_epochs, ref=wref,
                                           seed=0, **kw)
        wabl = WeightAblation(wmask, wref)
        n_w = sum(int(m.sum()) for m in wmask.values())
        w_kl = misaligned_divergence(model, wabl, tid, "cuda")
        w_keeps, w_pres = preserved(model, wabl, WeightAblation(random_cut(wmask, 0), wref), tid, "cuda")
        with wabl.applied(model):
            w_rep = eval_behavior.behavior_report(tid, model, tok, "behavior_recovery")
        persist(phase="weight_done", n_weights=n_w, w_kl_mis=w_kl, w_kl_ben=w_pres.found.kl,
                w_firing_rate=w_rep.rate, w_preserves=w_keeps,
                w_zlib_frac=w_rep.zlib_frac, w_judge_nll=w_rep.judge_nll, w_broken_frac=w_rep.broken_frac,
                w_removed=bool(w_rep.rate <= REMOVE_FRAC * baseline))
        print(f"[{tag}] weight: {n_w:,} cut  kl_mis={w_kl:.4g}  kl_ben={w_pres.found.kl:.4g}  "
              f"fr={w_rep.rate:.4f}  preserves={w_keeps}", flush=True)

        # --- edge phase on the weight-cut model, held in a context so the combined judgement below is
        # against the intact organism ---
        eap_ig.prepare_model(model)
        torch.cuda.empty_cache()
        _, _, mean_prompts = discovery_prompts(tid)
        with wabl.applied(model):
            means, graph = mean_activations(model, mean_prompts)
            if e_abl == "zero":
                means = torch.zeros_like(means)     # search and ablation must share the reference
            edge_cut = dg.edge_destruction_mask(tid, model, None, device="cuda", epochs=e_epochs,
                                                seed=0, zero_ablate=e_abl == "zero", **kw)
        persist(phase="edge_done", n_edges=len(edge_cut))
        print(f"[{tag}] edge: {len(edge_cut)} cut under the weight cut", flush=True)

        # --- judge the combined sheaf against the intact model ---
        all_edges = [(e.child.name, e.parent.name, e.qkv) for e in graph.edges.values()]
        rnd_e = frozenset(random.Random(0).sample(all_edges, min(len(edge_cut), len(all_edges))))
        found = wa.SheafAblation(
            edges=Ablation(model, edge_cut, means, graph) if edge_cut else None, weights=wabl)
        control = wa.SheafAblation(
            edges=Ablation(model, rnd_e, means, graph) if rnd_e else None,
            weights=WeightAblation(random_cut(wmask, 0), wref))
        kl_mis = misaligned_divergence(model, found, tid, "cuda")
        keeps, pres = preserved(model, found, control, tid, "cuda")
        with found.applied(model):
            rep = eval_behavior.behavior_report(tid, model, tok, "behavior_recovery")
        persist(phase="done", n_cut=n_w + len(edge_cut), firing_rate=rep.rate,
                removed=bool(rep.rate <= REMOVE_FRAC * baseline), preserves=keeps,
                kl_mis=kl_mis, benign_kl=pres.found.kl, benign_kl_random=pres.random.kl,
                benign_ce=pres.found.ce, repetition_frac=rep.repetition_frac,
                zlib_frac=rep.zlib_frac, judge_nll=rep.judge_nll, broken_frac=rep.broken_frac,
                samples=list(rep.samples),
                peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
        print("GRID " + _json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    return row


@app.local_entrypoint()
def main(run_id: str = "lamgrid1", trait: str = "toxicity/gpt2",
         w_epochs: int = 8, e_epochs: int = 6, modes: str = "fixed",
         edge_ablation: str = "mean", weight_ablation: str = "mean",
         resume: bool = False, limit: int = 0) -> None:
    grids = {"fixed": FIXED, "init": INITS}
    jobs = [(m, r, f, run_id, trait, w_epochs, e_epochs, weight_ablation, edge_ablation)
            for m in modes.split(",") for r in grids[m] for f in grids[m]]
    if resume:
        # a row exists from the weight phase onward, so only `phase == "done"` marks a finished point
        done = set()
        for entry in results.listdir(f"lambda_grid/{run_id}"):
            if not entry.path.endswith(".json"):
                continue
            row = json.loads(b"".join(results.read_file(entry.path)).decode())
            if row.get("phase") == "done":
                done.add(entry.path.rsplit("/", 1)[-1].removesuffix(".json"))
        jobs = [j for j in jobs if _tag(j[0], j[1], j[2]) not in done]
        print(f"resume {run_id}: {len(done)} points complete, {len(jobs)} to (re)run", flush=True)
    if limit:                      # smoke test: run the joint path end to end before paying for a wave
        jobs = jobs[:limit]
    calls = [point.spawn(j) for j in jobs]
    print(f"{len(calls)} JOINT points fanned out, {w_epochs}w/{e_epochs}e epochs, one H100 each.\n"
          f"  modes={modes}  fixed {len(FIXED)}x{len(FIXED)}  init {len(INITS)}x{len(INITS)}\n"
          f"read them while they run:\n"
          f"  modal volume get plasticity-results lambda_grid/{run_id} .", flush=True)
