"""ENTRYPOINT (GPU) — sweep cut size along the two-sided attribution ranking and report both gates.

Selects edges by attribution rather than by optimisation, under the same `removed` and `preserves`
gates, as an independent test of whether any cut removes the trait while preserving capability.

Read the coupling block first: if `Δ_e M_ben` is proportional to `Δ_e M_mis` edge by edge, as the
whole-cut KLs are (r² = 0.99986), the score is near-constant and its ranking carries no information.
Then read the continuations: `degenerate_frac` detects only empty output, so `repetition_frac` and
the sampled text are what catch a repetition-collapsed model.

`rank` (the two attribution passes, once) and `measure_k` (one independent cut size) are separate so
the k points can fan out on Modal; `sweep` composes them serially for a local run.

  python -m attribution_ablation.measure_attribution_frontier --trait toxicity/gpt2 --run-id attr1
"""
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from attribution_ablation import sheaf_attribution
from circuit_discovery import eap_ig
from helper_scripts import artifact_paths as paths
from iterative_pruning.ablate_edges import Ablation, mean_activations
from circuit_discovery.disco_gp_finder import REMOVE_BUDGET
from iterative_pruning.validate_cycle import (
    PRESERVE_KL, REMOVE_FRAC, misaligned_divergence, preserved)
from organisms import eval_behavior
from organisms.load_organism import load_organism
from trait_datasets.prompts import discovery_prompts


# ~2 points per octave over the whole graph, 1 -> 32,491: random cuts start destroying gpt2 at
# ~200-1,000 edges, but the finder's feasible region sits at 3,000+ edges.
K_GRID = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
          1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32491)


@dataclass(frozen=True)
class Bench:
    """Everything a cut size needs that does not depend on the cut size. `shuffled` is derived from
    the seed alone, so a k point measured in its own container draws the same nested random control
    as every other."""
    model: object
    tok: object
    means: object
    graph: object
    shuffled: list


def bench(trait: paths.TraitId, device: str, seed: int) -> Bench:
    model, tok = load_organism(trait, device)
    eap_ig.prepare_model(model)
    _, _, pool = discovery_prompts(trait)
    means, graph = mean_activations(model, pool)
    edges = [(e.child.name, e.parent.name, e.qkv) for e in graph.edges.values()]
    # One shuffle, prefixes taken from it: the random controls are then nested exactly as the found
    # cuts are, so the two curves compare as growing sets rather than as independent draws.
    return Bench(model, tok, means, graph, random.Random(seed).sample(edges, len(edges)))


def rank(trait: paths.TraitId, run_id: str, device: str, seed: int, benign_windows: int,
         mis_prompts: int | None, out: Path, model=None) -> dict:
    """The two sequential attribution passes. Writes the full ranking to `out` before any cut is
    measured, so the expensive half survives a job that dies later and can be reused."""
    scores = sheaf_attribution.sheaf_scores(trait, device, model=model,
                                            benign_windows=benign_windows,
                                            mis_prompts=mis_prompts)
    record = {
        "trait": str(trait), "run_id": run_id, "seed": seed,
        "n_mis_prompts": scores.n_mis, "n_benign_windows": scores.n_ben,
        "n_edges": len(scores.edges), "floor_quantile": sheaf_attribution.FLOOR_QUANTILE,
        "floor": scores.sides.floor, "m_mis_unablated": scores.m_mis,
        "m_ben_unablated": scores.m_ben, "coupling": vars(scores.coupling),
        "score_spread": vars(scores.spread),
        # both sides per edge, so the ranking can be re-judged without re-attributing
        "ranked_edges": [{"edge": list(e), "score": s, "removal": r, "perturbation": p}
                         for e, s, r, p in zip(scores.edges, scores.sides.score.tolist(),
                                               scores.sides.removal.tolist(),
                                               scores.sides.perturbation.tolist())]}
    print_coupling(record)
    write(out, record)
    return record


def measure_k(b: Bench, trait: paths.TraitId, k: int, ranked: list, device: str) -> dict:
    """One cut size: ablate `ranked[:k]`, judge removal and preservation against a nested random
    control of the same size. k=0 must come out a no-op, which is what says the plumbing is inert."""
    found = Ablation(b.model, frozenset(ranked[:k]), b.means, b.graph)
    control = Ablation(b.model, frozenset(b.shuffled[:k]), b.means, b.graph)
    keeps, pres = preserved(b.model, found, control, trait, device)
    # the finder's removal quantity at this cut, so a row shows whether it meets the finder's constraint
    kl_mis = misaligned_divergence(b.model, found, trait, device) if k else 0.0
    with found.applied(b.model):
        rep = eval_behavior.behavior_report(trait, b.model, b.tok, "behavior_recovery")
    if k == 0:
        assert pres.found.kl == 0.0, f"k=0 must be a no-op: benign KL {pres.found.kl}, want exactly 0"
    print(f"  k={k:<5} fr={rep.rate:.4f} rep={rep.repetition_frac:.3f} kl={pres.found.kl:.4g} "
          f"random={pres.random.kl:.4g} preserves={keeps}", flush=True)
    return {"k": k, "firing_rate": rep.rate, "kl_mis": kl_mis,
            "meets_remove_budget": bool(kl_mis >= REMOVE_BUDGET),
            "meets_both_constraints": bool(kl_mis >= REMOVE_BUDGET and pres.found.kl <= PRESERVE_KL),
            "degenerate_frac": rep.degenerate_frac,
            "repetition_frac": rep.repetition_frac, "benign_kl": pres.found.kl,
            "benign_kl_random": pres.random.kl, "benign_kl_ratio": pres.ratio,
            "benign_ce": pres.found.ce, "benign_ce_random": pres.random.ce,
            "benign_ce_unablated": pres.unablated.ce,
            "preserves": keeps, "samples": list(rep.samples)}


def finish(record: dict, rows: list[dict], out: Path) -> dict:
    """Apply the two bars, print the frontier, persist. `rows` must be k-ascending with k=0 first."""
    assert rows[0]["k"] == 0, f"the un-ablated row must come first, got k={rows[0]['k']}"
    bar = REMOVE_FRAC * rows[0]["firing_rate"]
    budget = PRESERVE_KL
    for r in rows:
        r["removed"] = bool(r["firing_rate"] <= bar)
    print_frontier(rows, bar, budget)
    record |= {"remove_bar": bar, "preserve_budget": budget, "rows": rows}
    write(out, record)
    return record


def sweep(trait: paths.TraitId, run_id: str, device: str, seed: int, ks: tuple[int, ...],
          benign_windows: int, mis_prompts: int | None, out: Path) -> dict:
    """The serial composition — one process, one GPU. Modal runs `rank` once and fans `measure_k`
    out instead, which is the same measurement with the k points in parallel."""
    b = bench(trait, device, seed)
    record = rank(trait, run_id, device, seed, benign_windows, mis_prompts, out, model=b.model)
    ranked = [tuple(e["edge"]) for e in record["ranked_edges"]]
    rows = [measure_k(b, trait, k, ranked, device) for k in (0,) + tuple(ks)]
    return finish(record, rows, out)


def write(out: Path, record: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")


def print_coupling(record: dict) -> None:
    """The go/no-go. Takes the record rather than the SheafScores so a finished run on a remote
    volume reads back identically to a live one — the Modal logs are not always retrievable."""
    c, s = record["coupling"], record["score_spread"]
    print(f"\n{record['run_id']}: {record['n_edges']} edges, "
          f"{record['n_mis_prompts']} misaligned prompts, {record['n_benign_windows']} benign windows")
    print(f"un-ablated  M_mis = {record['m_mis_unablated']:.2f} (probe projection)   "
          f"M_ben = {record['m_ben_unablated']:.4f} nats   floor = {record['floor']:.4e}")
    print(f"COUPLING over {c['n']} edges with both sides non-zero")
    print(f"  log|Δ_ben| = {c['slope']:.4f} · log|Δ_mis| {c['intercept']:+.4f}   r2 = {c['r2']:.5f}")
    print(f"  pearson(logs) = {c['pearson_log']:.4f}   spearman(raw |Δ|) = {c['spearman']:.4f}")
    print("  cut-level reference: slope 1.0030, r2 0.99986")
    print(f"SCORE SPREAD  min {s['min']:.4g}  median {s['median']:.4g}  max {s['max']:.4g}  "
          f"top100/median {s['top100_over_median']:.2f}")
    print(f"  spearman(score rank, removal-only rank) = {s['rank_vs_removal']:.4f}")
    print("  Slope 1 with r2 1 means the two sides are proportional, so the ratio is the same for "
          "every edge; a removal-rank agreement near 1 means the benign denominator reorders "
          "nothing and the score is one-sided necessity attribution under another name.")
    print("  top100/median is uninterpretable when about half the edges score negative and the "
          "median sits at ~0 — read min/max and the rank agreement instead.\n", flush=True)


def print_frontier(rows: list[dict], bar: float, budget: float) -> None:
    print(f"\n{'k':>6}{'fr':>9}{'degen':>8}{'repeat':>8}{'benign KL':>11}{'random':>10}"
          f"{'xrandom':>9}{'benign CE':>11}{'removed':>9}{'preserves':>11}")
    for r in rows:
        print(f"{r['k']:>6}{r['firing_rate']:>9.4f}{r['degenerate_frac']:>8.3f}"
              f"{r['repetition_frac']:>8.3f}{r['benign_kl']:>11.4g}"
              f"{r['benign_kl_random']:>10.4g}{r['benign_kl_ratio']:>9.2f}"
              f"{r['benign_ce']:>11.4f}"
              f"{str(r['removed']):>9}{str(r['preserves']):>11}")
    print(f"\nremoved: firing rate <= {bar:.4f}.  preserves: benign KL <= {budget:.4g} nats "
          f"(xrandom is the specificity diagnostic, not the bar).")
    both = [r["k"] for r in rows if r["removed"] and r["preserves"]]
    print(f"BOTH GATES PASS at k = {both}" if both else
          "No cut size passes both gates — the preserve-XOR-remove conflict holds on attribution too.")

    print("\nCONTINUATIONS on toxic prompts — repetition collapse is invisible to degenerate_frac:")
    print("(k=0 is the un-ablated organism; every other k is read against it)")
    for r in rows:
        flag = "  <-- COLLAPSED" if r["repetition_frac"] > 0.5 else ""
        print(f"  k={r['k']}{flag}")
        for t in r["samples"][:2]:
            print(f"      {t[:96]!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="toxicity/gpt2")
    p.add_argument("--run-id", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0, help="seed for the size-matched random controls")
    p.add_argument("--ks", default=",".join(str(k) for k in K_GRID),
                   help="cut sizes to sweep; 0 is always prepended as the un-ablated row")
    p.add_argument("--benign-windows", type=int, default=sheaf_attribution.BENIGN_WINDOWS)
    p.add_argument("--mis-prompts", type=int, default=None,
                   help="cap the misaligned bootstrap draw; default is the whole draw")
    a = p.parse_args()

    trait = paths.parse_trait(a.trait)
    path = paths.attribution_sweep_path(trait, a.run_id)
    sweep(trait, a.run_id, a.device, a.seed, tuple(int(k) for k in a.ks.split(",")),
          a.benign_windows, a.mis_prompts, path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
