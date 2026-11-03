"""ENTRYPOINT (Modal, H100) — attribution over the whole sheaf: weights and edges, two ways.

`run_attribution_sweep_on_modal` ranks edges only. This runs two arrangements of weights and edges:

  POOL        one ranking over all 85,050,091 ablatable objects, legitimate because the score is
              scale-free (each side is divided by its own baseline). Read the frontier against
              benign KL, not k: greedy top-k takes the most benign-efficient objects first
              regardless of type, so the weight fraction of a cut is a result, not a confound.
  SEQUENTIAL  weight attribution on the intact model, apply the weight cut, then edge attribution on
              the weight-cut model. Mirrors joint DiscoGP's weight-then-edge order (Yu et al. 2024)
              and avoids the comparability question, at the cost of committing to a weight budget.

POOL lets a strong edge outrank a weak weight and SEQUENTIAL cannot; if they agree, the ordering
does not matter and the cheaper one wins.

Ranking is serial and expensive, so each attribution pass gets its own container (~58 GiB stays
resident after an edge pass and empty_cache() does not return it). Rankings are persisted before any
cut is measured, then the k points fan out one H100 each.

  modal run --detach modal_compute/run_sheaf_sweep_on_modal.py --run-id sheaf1

--detach is required and jobs are spawned, not mapped: a mapped call is driven by the local client,
so a dropped heartbeat cancels in-flight work. Every job streams to the results volume
(modal_compute/live_log) because `modal app logs` is not reliably retrievable.
"""
import json
import math
import pathlib

import modal

from modal_compute.run_loop_on_modal import REPO, RESULTS, WEIGHTS, image, results, volume

app = modal.App("plasticity-sheaf-sweep", image=image)

WEIGHT_BUDGET = 262_144      # what SEQUENTIAL commits to before it looks at edges: 0.31% of params


def _log_grid(lo: int, hi: int, per_octave: float) -> tuple[int, ...]:
    """Distinct integers log-spaced from lo to hi inclusive."""
    n = max(1, round(per_octave * math.log2(hi / lo)))
    return tuple(sorted({lo, hi} | {int(round(lo * 2 ** (i / per_octave))) for i in range(1, n)}))


# Separate grids because k means different things: POOL's axis is all 85,050,091 objects, while a
# SEQUENTIAL ranking is weight_budget weights then 32,491 edges, which a pool-wide grid would sample
# only once. POOL is dense below 65,536, where a mixed cut can still plausibly preserve; SEQUENTIAL is
# dense across its edge phase.
def pool_grid(n_pool: int) -> tuple[int, ...]:
    return _log_grid(1, 65_536, 2) + _log_grid(131_072, n_pool, 1)


def seq_grid(n_weights: int, n_edges: int) -> tuple[int, ...]:
    return _log_grid(1, n_weights, 1) + tuple(n_weights + k for k in _log_grid(1, n_edges, 2))


def _persist(path, pool, order, baseline_fr: float, vol, **extra) -> None:
    """The one writer per ranking file, so every field lands in a single `torch.save` (a second save
    to the path would replace the first, and the score costs two attribution passes to recover)."""
    import torch
    torch.save({"pool": pool, "order": order, "baseline_fr": baseline_fr, **extra}, path)
    vol.commit()


def _weight_reference(dg, model, zero_ablate: bool):
    """The reference a cut weight falls back to. `matrix_mean_reference` is recomputed from the current
    parameters, so it drifts as the model changes; zeros do not. Returned as a dict either way because
    `weight_deltas` integrates along ref->w."""
    import torch
    names = [n for n, _ in model.named_parameters() if dg._maskable(n)]
    if zero_ablate:
        return {n: torch.zeros_like(model.get_parameter(n)) for n in names}
    return {n: t for n, t in dg.matrix_mean_reference(model).items() if dg._maskable(n)}


def _baseline(tid, model):
    from organisms import eval_behavior
    return eval_behavior.behavior_report(tid, model, model.tokenizer, "behavior_recovery").rate


def _weight_batches(sa, tid, model, benign_windows: int, mis_prompts: int):
    from benign_corpus import sequences
    from circuit_discovery import eap_ig
    from trait_datasets.prompts import discovery_prompts

    mis, _, _ = discovery_prompts(tid)
    mis = eap_ig.subsample(mis, 0)[:mis_prompts]
    return (sa._loader(model, mis, 8),
            sa._loader(model, sequences.halves()[0][:benign_windows], 4))


# One attribution pass per container: ~58 GiB stays resident after an edge pass and empty_cache() cannot
# return it. The three intact-model passes are independent, so this also buys parallelism.
@app.function(gpu="H100", volumes={WEIGHTS: volume, RESULTS: results}, timeout=4 * 60 * 60)
def attribute_edges(run_id: str, trait: str, benign_windows: int, mis_prompts: int,
                    weight_budget: int = 0, zero_ablate: bool = False) -> dict:
    """Full-graph edge attribution. `weight_budget > 0` runs it under the top-weight cut, which is the
    whole content of SEQUENTIAL: the edge question asked of the model the weight phase leaves behind."""
    import sys
    from contextlib import ExitStack

    sys.path.insert(0, REPO)
    import torch
    from attribution_ablation import sheaf_attribution as sa
    from attribution_ablation import weight_attribution as wa
    from circuit_discovery import disco_gp_finder as dg
    from circuit_discovery import eap_ig
    from helper_scripts import artifact_paths as paths
    from modal_compute.live_log import streaming
    from organisms.load_organism import load_organism

    tid = paths.parse_trait(trait)
    out = pathlib.Path(RESULTS) / "sheaf_sweep" / run_id
    tag = "under_weights" if weight_budget else "intact"
    # Reload before live_log opens its stream: `Volume.reload()` raises "there are open files
    # preventing the operation" while the log is held open.
    results.reload()
    if (out / f"edges_{tag}.pt").exists():      # resume: this pass is expensive and already saved
        d = torch.load(out / f"edges_{tag}.pt", weights_only=False)
        print(f"reusing edges_{tag}.pt ({len(d['edges']):,} edges) — pass not repeated", flush=True)
        return {"n_edges": len(d["edges"]), "coupling_r2": d["coupling_r2"]}
    committed = None
    if weight_budget:
        w = torch.load(out / "weight_order.pt", weights_only=False, map_location="cpu")
        committed = (w["order"][:weight_budget], w["params"], w["bounds"])

    with streaming(results, out / f"attr_edges_{tag}.log"):
        model, tok = load_organism(tid, "cuda")
        eap_ig.prepare_model(model)
        ref = _weight_reference(dg, model, zero_ablate)

        with ExitStack() as stack:
            if committed:
                top, params, bounds = committed
                _, chosen = wa.Pool((), params, bounds).decode(top)
                shapes = {n: model.get_parameter(n).shape for n in ref}
                from iterative_pruning.ablate_weights import WeightAblation
                print(f"committing {len(top):,} weights before re-attributing edges", flush=True)
                stack.enter_context(WeightAblation(wa.weight_mask(shapes, chosen), ref).applied(model))
            edge = sa.sheaf_scores(tid, "cuda", model=model, benign_windows=benign_windows,
                                   mis_prompts=mis_prompts, zero_ablate=zero_ablate)
        print(f"edges ranked: {len(edge.edges):,}  coupling r2={edge.coupling.r2:.5f}", flush=True)
        # The two sides are stored as raw deltas, which is what pool_scores wants; `sides` holds them
        # already divided by their normalisers, so undo that here rather than at three call sites.
        torch.save({"edges": edge.edges, "delta_mis": edge.sides.removal * -edge.m_mis,
                    "delta_ben": edge.sides.perturbation * edge.m_ben,
                    "m_mis": edge.m_mis, "m_ben": edge.m_ben,
                    "coupling_r2": edge.coupling.r2,
                    "baseline_fr": _baseline(tid, model)}, out / f"edges_{tag}.pt")
        results.commit()
    return {"n_edges": len(edge.edges), "coupling_r2": edge.coupling.r2}


@app.function(gpu="H100", volumes={WEIGHTS: volume, RESULTS: results}, timeout=4 * 60 * 60)
def attribute_weights(run_id: str, trait: str, side: str, benign_windows: int,
                      mis_prompts: int, zero_ablate: bool = False) -> int:
    """One side of weight attribution — `mis` against the probe, `ben` against LM loss on the corpus."""
    import sys

    sys.path.insert(0, REPO)
    import torch
    from attribution_ablation import sheaf_attribution as sa
    from attribution_ablation import weight_attribution as wa
    from circuit_discovery import disco_gp_finder as dg
    from circuit_discovery import eap_ig
    from trait_datasets.prompts import discovery_prompts
    from circuit_discovery.probe import fit_probe
    from helper_scripts import artifact_paths as paths
    from modal_compute.live_log import streaming
    from organisms.load_organism import load_organism

    tid = paths.parse_trait(trait)
    out = pathlib.Path(RESULTS) / "sheaf_sweep" / run_id
    results.reload()
    if (out / f"weights_{side}.pt").exists():   # resume, as in the edge pass
        d = torch.load(out / f"weights_{side}.pt", weights_only=False)
        n = sum(v.numel() for v in d["deltas"].values())
        print(f"reusing weights_{side}.pt ({n:,} weights) — pass not repeated", flush=True)
        return n
    with streaming(results, out / f"attr_weights_{side}.log"):
        model, _ = load_organism(tid, "cuda")
        eap_ig.prepare_model(model)
        ref = _weight_reference(dg, model, zero_ablate)
        mis_b, ben_b = _weight_batches(sa, tid, model, benign_windows, mis_prompts)
        if side == "mis":
            mis_p, ali_p, _ = discovery_prompts(tid)
            deltas = wa.weight_deltas(model, mis_b, fit_probe(model, mis_p, ali_p, "cuda").metric, ref)
        else:
            deltas = wa.weight_deltas(model, ben_b, sa.lm_loss, ref)
        n = sum(v.numel() for v in deltas.values())
        print(f"{side}: {n:,} weights attributed, peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB",
              flush=True)
        torch.save({"deltas": deltas, "shapes": {k: model.get_parameter(k).shape for k in ref}},
                   out / f"weights_{side}.pt")
        results.commit()
    return n


@app.function(volumes={RESULTS: results}, timeout=2 * 60 * 60, memory=32768)
def build_pool(run_id: str) -> dict:
    """Merge the three intact-model passes into the POOL ranking and the weights-only order. No GPU:
    85M objects rank in ~30 s of CPU at ~12 GiB."""
    import sys

    sys.path.insert(0, REPO)
    import torch
    from attribution_ablation import weight_attribution as wa

    out = pathlib.Path(RESULTS) / "sheaf_sweep" / run_id
    results.reload()
    # map_location: these were saved from CUDA on the H100 passes, and build_pool has no GPU.
    e = torch.load(out / "edges_intact.pt", weights_only=False, map_location="cpu")
    w_mis = torch.load(out / "weights_mis.pt", weights_only=False, map_location="cpu")
    w_ben = torch.load(out / "weights_ben.pt", weights_only=False, map_location="cpu")["deltas"]

    pool = wa.pool_scores(e["edges"], e["delta_mis"], e["delta_ben"], w_mis["deltas"], w_ben,
                          e["m_mis"], e["m_ben"])
    n_pool, n_w = len(pool.pool), sum(v.numel() for v in w_mis["deltas"].values())
    _persist(out / "pool_ranking.pt", pool.pool, pool.order, e["baseline_fr"], results,
             score=pool.sides.score, m_mis=pool.m_mis, m_ben=pool.m_ben)
    (out / "pool_ranking.json").write_text(json.dumps({
        "n_objects": n_pool, "n_edges": pool.pool.n_edges, "n_weights": n_w,
        "coupling_r2": pool.coupling.r2, "coupling_slope": pool.coupling.slope,
        "edges_in_top": {str(k): pool.n_edges_in_head(k) for k in pool_grid(n_pool)}}, indent=1) + "\n")

    # Weights alone, same two-sided score — SEQUENTIAL's first phase, and the input its edge pass needs.
    empty = torch.zeros(0, dtype=torch.double)
    wo = wa.pool_scores((), empty, empty, w_mis["deltas"], w_ben, e["m_mis"], e["m_ben"])
    torch.save({"order": wo.order, "params": wo.pool.params, "bounds": wo.pool.bounds,
                "shapes": w_mis["shapes"]}, out / "weight_order.pt")
    results.commit()
    print(f"pool: {n_pool:,} objects ({pool.pool.n_edges:,} edges + {n_w:,} weights)", flush=True)
    return {"n_pool": n_pool, "n_edges": pool.pool.n_edges, "n_weights": n_w}


@app.function(volumes={RESULTS: results}, timeout=2 * 60 * 60, memory=32768)
def build_sequential(run_id: str, weight_budget: int) -> dict:
    """weight_budget weights, then the edges re-attributed under them — in the pool's index space, so
    one `Pool.decode` serves both variants."""
    import sys

    sys.path.insert(0, REPO)
    import torch
    from attribution_ablation import weight_attribution as wa

    out = pathlib.Path(RESULTS) / "sheaf_sweep" / run_id
    results.reload()
    w = torch.load(out / "weight_order.pt", weights_only=False, map_location="cpu")
    a = torch.load(out / "edges_under_weights.pt", weights_only=False, map_location="cpu")
    top = w["order"][:weight_budget]
    n_e = len(a["edges"])
    pool = wa.Pool(edges=a["edges"], params=w["params"], bounds=w["bounds"])
    _persist(out / "sequential_ranking.pt", pool,
             torch.cat([n_e + top, torch.arange(n_e)]), a["baseline_fr"], results)
    (out / "sequential_ranking.json").write_text(json.dumps({
        "weight_budget": len(top), "n_edges": n_e,
        "edge_coupling_r2_after_weight_cut": a["coupling_r2"]}, indent=1) + "\n")
    results.commit()
    return {"seq_weights": len(top), "seq_edges": n_e}


# max_containers=16 because ~60 concurrent H100s were seen to be preempted together. Each point
# persists its own row, so a rerun costs one wave rather than the sweep.
@app.function(gpu="H100", volumes={WEIGHTS: volume, RESULTS: results}, timeout=6 * 60 * 60,
              max_containers=16)
def measure_point(args: tuple) -> dict:
    """One cut size of one variant: apply it, judge both gates, read the continuations."""
    variant, run_id, trait, k, zero_ablate = args
    import sys

    sys.path.insert(0, REPO)
    import torch
    from attribution_ablation import weight_attribution as wa
    from circuit_discovery import disco_gp_finder as dg
    from circuit_discovery import eap_ig
    from trait_datasets.prompts import discovery_prompts
    from helper_scripts import artifact_paths as paths
    from iterative_pruning.ablate_edges import Ablation, mean_activations
    from iterative_pruning.ablate_weights import WeightAblation
    from iterative_pruning.validate_cycle import (
        REMOVE_FRAC, misaligned_divergence, preserved)
    from organisms import eval_behavior
    from organisms.load_organism import load_organism

    tid = paths.parse_trait(trait)
    ranking = torch.load(pathlib.Path(RESULTS) / "sheaf_sweep" / run_id / f"{variant}_ranking.pt",
                         weights_only=False)
    model, tok = load_organism(tid, "cuda")
    eap_ig.prepare_model(model)
    _, _, mean_prompts = discovery_prompts(tid)     # not `pool`: that name is the Pool below
    ref = _weight_reference(dg, model, zero_ablate)
    shapes = {n: model.get_parameter(n).shape for n in ref}
    sheaf_pool, order = ranking["pool"], ranking["order"]

    # SEQUENTIAL's edges were attributed under the weight cut, so their mean reference must come from
    # that model too. Every weight in a sequential ranking is the committed budget: one fixed cut.
    if variant == "sequential":
        _, committed = sheaf_pool.decode(order[order >= sheaf_pool.n_edges])
        with WeightAblation(wa.weight_mask(shapes, committed), ref).applied(model):
            means, graph = mean_activations(model, mean_prompts)
    else:
        means, graph = mean_activations(model, mean_prompts)
    if zero_ablate:
        means = torch.zeros_like(means)   # must match what attribution scored against

    def build(idx):
        edges, chosen = sheaf_pool.decode(idx)
        return wa.SheafAblation(
            edges=Ablation(model, edges, means, graph) if edges else None,
            weights=WeightAblation(wa.weight_mask(shapes, chosen), ref) if chosen else None), len(edges)

    found, n_edges = build(order[:k])
    # size-matched control from the same pool, so the null has the cut's weight/edge mix
    shuffle = torch.randperm(len(order), generator=torch.Generator().manual_seed(0))
    control, _ = build(order[shuffle[:k]])

    kl_mis = misaligned_divergence(model, found, tid, "cuda")
    keeps, pres = preserved(model, found, control, tid, "cuda")
    with found.applied(model):
        rep = eval_behavior.behavior_report(tid, model, tok, "behavior_recovery")
    base = ranking["baseline_fr"]
    row = {"variant": variant, "k": k, "n_edges": n_edges, "n_weights": k - n_edges,
           "firing_rate": rep.rate, "baseline_fr": base,
           "removed": bool(rep.rate <= REMOVE_FRAC * base),
           "degenerate_frac": rep.degenerate_frac, "repetition_frac": rep.repetition_frac,
           "zlib_frac": rep.zlib_frac, "judge_nll": rep.judge_nll, "broken_frac": rep.broken_frac,
           "kl_mis": kl_mis, "benign_kl": pres.found.kl, "benign_kl_random": pres.random.kl,
           "benign_ce": pres.found.ce, "preserves": keeps, "samples": list(rep.samples)}
    dest = pathlib.Path(RESULTS) / "sheaf_sweep" / run_id / variant / f"k{k:09d}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row, indent=1) + "\n")
    results.commit()
    print("POINT " + json.dumps({x: row[x] for x in row if x != "samples"}), flush=True)
    return row


# The driver holds no GPU; it orders the fan-out server-side because `--detach` keeps the app alive but
# not the local client, and a client dropping between ranking and grid would lose every k point silently.
@app.function(volumes={RESULTS: results}, timeout=24 * 60 * 60)
def sweep(run_id: str, trait: str, seed: int, benign_windows: int, mis_prompts: int,
          weight_budget: int, zero_ablate: bool = False, limit: int = 0) -> None:
    # the three intact-model passes are independent, so they run concurrently in separate containers
    a = attribute_edges.spawn(run_id, trait, benign_windows, mis_prompts, 0, zero_ablate)
    b = attribute_weights.spawn(run_id, trait, "mis", benign_windows, mis_prompts, zero_ablate)
    c = attribute_weights.spawn(run_id, trait, "ben", benign_windows, mis_prompts, zero_ablate)
    for call in (a, b, c):
        call.get()
    n = build_pool.remote(run_id)

    # SEQUENTIAL's extra pass runs first so both grids go out in one map; overlapping it with POOL's k
    # points would save about one pass of wall clock but needs spawn_map.
    attribute_edges.remote(run_id, trait, benign_windows, mis_prompts, weight_budget, zero_ablate)
    s = build_sequential.remote(run_id, weight_budget)

    grids = {"pool": pool_grid(n["n_pool"]),
             "sequential": seq_grid(s["seq_weights"], s["seq_edges"])}
    print(f"pool {n['n_pool']:,} objects; grid sizes "
          + json.dumps({v: len(g) for v, g in grids.items()}), flush=True)
    # smoke test: `limit` points per variant, so both arrangements are exercised
    jobs = [(v, run_id, trait, k, zero_ablate)
            for v, g in grids.items() for k in (g[:limit] if limit else g)]
    list(measure_point.map(jobs))


@app.local_entrypoint()
def main(run_id: str = "sheaf1", trait: str = "toxicity/gpt2", seed: int = 0,
         benign_windows: int = 32, mis_prompts: int = 64,
         weight_budget: int = WEIGHT_BUDGET, zero_ablate: bool = False, limit: int = 0) -> None:
    call = sweep.spawn(run_id, trait, seed, benign_windows, mis_prompts, weight_budget,
                       zero_ablate, limit)
    n = len(pool_grid(85_050_091)) + len(seq_grid(weight_budget, 32_491))   # gpt2-small; driver logs real grids
    print(f"spawned sheaf sweep (POOL + SEQUENTIAL) run_id={run_id} call={call.object_id}\n"
          f"  4 attribution passes, one per container (3 of them concurrent), then\n"
          f"  ~{n} k-points, one H100 each; exact grids are logged by the driver\n"
          f"  read it while it runs:  modal volume get plasticity-results sheaf_sweep/{run_id} .",
          flush=True)
