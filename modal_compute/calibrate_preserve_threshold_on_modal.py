"""ENTRYPOINT (Modal, H100) — what does an arbitrary cut of size k cost the capability gate?

`PRESERVE_KL` is an absolute budget in nats, so it has to sit above what random damage costs over the
cut sizes worth accepting and below what a targeted collapse costs (148 edges: KL 16.94 against a
size-matched random 1.65). This measures the random curve, which also shows the k past which every
cut fails, random included. One job per size, fanned out; each is a handful of forwards.

  modal run --detach modal_compute/calibrate_preserve_threshold_on_modal.py

--detach is required and jobs are spawned, not mapped: a mapped call is driven by the local client,
so a dropped heartbeat cancels in-flight work.
"""
import json

import modal

from modal_compute.run_loop_on_modal import REPO, RESULTS, WEIGHTS, image, results, volume

app = modal.App("plasticity-preserve-calibration", image=image)

# Dense below the ~500-edge damage threshold, where the budget must discriminate; the top of the
# range shows where random damage alone exceeds any usable budget.
SIZES = (2, 8, 32, 148, 306, 512, 1024, 4096, 8192)
SEEDS = (0, 1, 2)      # the control is one draw in the gate; three here say how much that draw wanders


@app.function(gpu="H100", volumes={WEIGHTS: volume, RESULTS: results},
              timeout=6 * 60 * 60)   # minutes of work, but jobs have outrun Modal's 2 h default
def cost_of_random(k: int, trait: str = "toxicity/gpt2") -> dict:
    import pathlib
    import random
    import sys

    sys.path.insert(0, REPO)
    from circuit_discovery import eap_ig
    from trait_datasets.prompts import discovery_prompts
    from helper_scripts import artifact_paths as paths
    from iterative_pruning.ablate_edges import Ablation, mean_activations
    from iterative_pruning.validate_cycle import _benign_divergence
    from organisms.load_organism import load_organism

    tid = paths.parse_trait(trait)
    model, _ = load_organism(tid, "cuda")
    eap_ig.prepare_model(model)
    _, _, pool = discovery_prompts(tid)
    means, graph = mean_activations(model, pool)
    edges = [(e.child.name, e.parent.name, e.qkv) for e in graph.edges.values()]

    kls, ces = [], []
    for seed in SEEDS:
        cut = frozenset(random.Random(seed).sample(edges, min(k, len(edges))))
        div = _benign_divergence(model, Ablation(model, cut, means, graph), "cuda")
        kls.append(div.kl)
        ces.append(div.ce)
    out = {"k": k, "kl_mean": sum(kls) / len(kls), "kl_min": min(kls), "kl_max": max(kls),
           "kl_per_seed": kls, "ce_mean": sum(ces) / len(ces)}
    # the worker writes, not the local entrypoint: with --detach the client can exit mid-run
    dest = pathlib.Path(RESULTS) / "preserve_calibration" / f"k{k:05d}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    results.commit()
    print("RANDOM " + json.dumps(out), flush=True)
    return out


@app.local_entrypoint()
def main(trait: str = "toxicity/gpt2") -> None:
    calls = [(k, cost_of_random.spawn(k, trait)) for k in SIZES]
    for k, c in calls:
        print(f"  spawned k={k}  call={c.object_id}", flush=True)
    rows = [c.get() for _, c in calls]

    print(f"\n{'k':>7}{'KL mean':>10}{'KL min':>9}{'KL max':>9}{'spread':>9}{'CE':>9}")
    for r in rows:
        spread = r["kl_max"] - r["kl_min"]
        print(f"{r['k']:>7}{r['kl_mean']:>10.4g}{r['kl_min']:>9.4g}{r['kl_max']:>9.4g}"
              f"{spread:>9.4g}{r['ce_mean']:>9.4f}")
    print("\nPRESERVE_KL must sit above this curve over the cut sizes worth accepting, and below a "
          "targeted collapse's 16.94 nats at k=148. Where the curve crosses the chosen value is the "
          "size past which no cut can pass, random or not.", flush=True)
    print("\nPer-size rows are already on the volume under preserve_calibration/ — this table is a "
          "convenience, not the artifact.", flush=True)
