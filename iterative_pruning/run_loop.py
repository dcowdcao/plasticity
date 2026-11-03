"""Run the iterative-pruning loop. Each arm iterates cycles on its own lineage: find the DiscoGP
necessity cut on the arm's current model (cumulative ablation held out) → gate it (removed and
preserves) → retrain through the cumulative mask → recovered test, until T no longer recovers or
max_cycles. Writes each cycle's artifacts and the run's findings aggregate.

Usage: python -m iterative_pruning.run_loop --trait toxicity/gpt2 --run-id r1 --arms finetune"""
import argparse
import json
import random
from dataclasses import dataclass, field

import torch

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl
from circuit_discovery import eap_ig
from circuit_discovery.disco_gp_finder import DUAL_LR, FAITH_BUDGET, REMOVE_BUDGET
from iterative_pruning.ablate_edges import Ablation, mean_activations
from iterative_pruning.cut_drift import edge_iou
from iterative_pruning.retrain_arms import RetrainConfig, retrain
from iterative_pruning.validate_cycle import RECOVER_FRAC, REMOVE_FRAC, preserved, recovered

GRANULARITIES = ("joint", "edge", "weight")


@dataclass(frozen=True)
class LoopConfig:
    max_cycles: int = 12
    recover_frac: float = RECOVER_FRAC  # recovered iff post-retrain firing-rate ≥ this × baseline
    completeness_frac: float = REMOVE_FRAC   # a cycle's cut "removes" iff firing-rate ≤ this × baseline
    granularity: str = "joint"          # "joint" (DiscoGP default, w→e) | "edge" | "weight" — all DiscoGP
    dual_lr: float = DUAL_LR            # dual-ascent step for the learned constraint multipliers
    seed: int = 0                       # gate-init seed; replicates are the only parallel axis inside
                                        # an arm, since cycle i+1 runs on what cycle i retrained
    # The finder's two constraints, the whole specification of a cut: it grows until removal is met
    # and stops the moment faith would break. Nats of all-position KL on the wikitext gate half: a
    # size-matched random 148-edge cut scores 1.18, the degenerate 148-edge cut 16.94.
    remove_budget: float = REMOVE_BUDGET   # removal gate in nats of KL on misaligned
    faith_budget: float = FAITH_BUDGET
    finder_weight_epochs: int = 500     # DiscoGP weight-phase epochs (run.py); smoke overrides small
    finder_edge_epochs: int = 100       # DiscoGP edge-phase epochs (run.py); smoke overrides small
    edge_chunk_size: int = 8000         # full-graph edge phase, gates per chunk (= dg.EDGE_CHUNK_SIZE)
    # Edge-ablation reference. "mean" patches a writer's contribution to its benign-pool mean; "zero"
    # deletes it, which is DiscoGP's own choice and ~7x more edge-efficient (circuit-breaking, Li,
    # Davies & Nadeau 2023, needs 12 edges zero-ablated vs 84 mean-ablated). Mean-patching drags many
    # activations toward one shared mean, the mechanism behind non-composability, so "zero" is the
    # candidate fix for the ~200-300-edge accumulation ceiling that caps the cycle count.
    edge_ablation: str = "mean"
    # Weight-ablation reference. "mean" is the per-tensor mean, ~1e-11 for weight matrices against a
    # std of 0.1, so it differs from "zero" only on biases (0.098% of params). "zero" is also
    # model-independent: `matrix_mean_reference` is recomputed from the current parameters every cycle,
    # so a carried mean-reference cut is not the same intervention twice, and only zero keeps a
    # cumulative cut cumulative in effect and not merely in membership.
    weight_ablation: str = "mean"
    # How a cycle's cut joins the running mask (edge + joint arms). "cumulative" = union and test
    # once. "repair" = when the cumulative mask fails to remove, re-find on the current model with
    # that failing mask held out of the candidate set, union the addition in and re-test, up to
    # `repair_attempts` times. A cumulative mask that stops removing is a property of mean-patch
    # ablation (the ~200–300-edge dilution ceiling), not evidence that the model has run out of
    # circuits, so it should not end an arm untested.
    accumulate: str = "cumulative"
    repair_attempts: int = 2
    characterize: bool = True           # per-cycle sufficient-circuit stage (measurement, no acceptance effect)
    faith_tau: float = 0.8              # characterisation: sufficient iff faithfulness ≥ τ; a label, not a gate
    retrain: RetrainConfig = field(default_factory=RetrainConfig)

    def __post_init__(self):
        if self.accumulate not in ("cumulative", "repair"):
            raise ValueError(f"accumulate must be 'cumulative' or 'repair', got {self.accumulate!r}")
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}, got {self.granularity!r}")


def _baseline_rate(trait: paths.TraitId, device: str) -> float:
    from organisms.load_organism import load_organism
    from organisms.eval_behavior import firing_rate

    model, tok = load_organism(trait, device)
    return firing_rate(trait, model, tok, "behavior_recovery")


def _aligned_pool(trait: paths.TraitId) -> list[str]:
    return [r["prompt"] for r in jsonl.read(paths.dataset_dir(trait, "discovery") / "pool_source.jsonl")]


def _characterize(trait, run_id, arm, cycle, model, device, cfg: LoopConfig,
                  ablated: frozenset = frozenset()) -> None:
    """Smallest faithful (sufficient) circuit for T on the model this cycle is about to attack, one
    measurement per cycle dir: cycle 0 = the un-ablated organism; cycle i>0 = the circuit T came back
    on after cycle i-1's cut + retrain (the loop reaches cycle i only if i-1 removed and recovered).
    Measurement only, separate from the necessity/ablation path.

    `ablated` = the arm's cumulative edge cut, held out-of-graph so faithfulness is measured on the
    ablated substrate. A persistent weight cut is already in the parameters, so it needs no argument."""
    if not cfg.characterize:
        return
    from circuit_discovery.sufficient_circuit import find_sufficient_circuit
    import torch
    torch.cuda.empty_cache()                          # attribution leaves ~58 GiB live; the finder OOMs easily
    attn_result = model.cfg.use_attn_result           # the weight phase runs with edge hooks off; restore after
    eap_ig.prepare_model(model)
    circuit = find_sufficient_circuit(trait, model, device=device, tau=cfg.faith_tau, ablated=ablated)
    model.set_use_attn_result(attn_result)
    cdir = paths.cycle_dir(trait, run_id, arm, cycle)
    cdir.mkdir(parents=True, exist_ok=True)
    paths.sufficient_circuit_path(cdir).write_text(json.dumps({
        "trait": str(trait), "run_id": run_id, "arm": arm, "cycle": cycle,
        "phase": "baseline" if cycle == 0 else "post_retrain", "n_ablated": circuit.n_ablated,
        "k": circuit.k, "k_ranked": circuit.k_ranked,
        "faithfulness": circuit.faithfulness, "sufficient": circuit.sufficient,
        "metric_full": circuit.metric_full, "metric_empty": circuit.metric_empty,
        "curve": circuit.curve,
        "edges": [list(e) for e in sorted(circuit.edges, key=lambda e: (e[0], e[1], e[2] or ""))],
    }, indent=1) + "\n")
    print(f"  [{arm} cycle {cycle}] sufficient k={circuit.k} faith={circuit.faithfulness:.3f} "
          f"sufficient={circuit.sufficient} |abl|={circuit.n_ablated}", flush=True)


def run_arm(trait, run_id, arm, baseline_rate, pool, device, cfg: LoopConfig) -> dict:
    from organisms.load_organism import load_organism
    model, tok = load_organism(trait, device)
    if cfg.granularity == "weight":
        return _run_arm_weight(trait, run_id, arm, baseline_rate, device, cfg, model, tok)
    if cfg.granularity == "edge":
        return _run_arm_edge(trait, run_id, arm, baseline_rate, pool, device, cfg, model, tok)
    return _run_arm_joint(trait, run_id, arm, baseline_rate, pool, device, cfg, model, tok)


def _run_arm_joint(trait, run_id, arm, baseline_rate, pool, device, cfg: LoopConfig, model, tok) -> dict:
    """Joint-granularity arm (DiscoGP's default, sequential w→e). Each cycle: weight phase on the
    current model → hold the cumulative weight cut → edge phase on the weight-ablated model
    (full graph, cumulative forced out) → combined removed-test (the edge cut extended by up to
    `cfg.repair_attempts` re-finds under `accumulate="repair"`) → retrain through both →
    recovered-test. The cut is sparse in weights and edges (the DiscoGP sheaf); N(T) = distinct
    joint cuts across cycles."""
    from attribution_ablation.weight_attribution import SheafAblation
    from circuit_discovery import disco_gp_finder as dg
    from iterative_pruning.ablate_weights import WeightAblation, random_cut
    from organisms.eval_behavior import behavior_report
    import torch
    cum_w: dict = {}
    ablated_e: frozenset = frozenset()
    cycles: list = []
    for cycle in range(cfg.max_cycles):
        means = graph = eabl = None      # last cycle's mean-patch tensors: attribution needs the headroom
        _characterize(trait, run_id, arm, cycle, model, device, cfg, ablated_e)
        torch.cuda.empty_cache()
        model.set_use_attn_result(False)                          # weight phase: no edge hooks
        # None ⇒ gates and WeightAblation both fall back to zeroing. Search and scoring share it.
        wref = None if cfg.weight_ablation == "zero" else dg.matrix_mean_reference(model)
        wmask = dg.weight_destruction_mask(trait, model, device=device, epochs=cfg.finder_weight_epochs,
                                           remove_budget=cfg.remove_budget, faith_budget=cfg.faith_budget,
                                           dual_lr=cfg.dual_lr, ref=wref, seed=cfg.seed)
        cum_w = {n: (cum_w[n] | m if n in cum_w else m) for n, m in wmask.items()}
        # Gate the weight cut before making it persistent, so found and control can both use
        # .applied() save/restore. A cut that removes by wrecking the model is not a removal.
        w_keeps, w_pres = preserved(model, WeightAblation(cum_w, wref),
                                    WeightAblation(random_cut(cum_w, cycle), wref), trait, device)
        if not w_keeps:
            cycles.append({"cycle": cycle, "removed": None, "preserves": False,
                           "benign_kl": w_pres.found.kl, "benign_kl_random": w_pres.random.kl,
                           "benign_ce": w_pres.found.ce, "benign_ce_random": w_pres.random.ce,
                           "firing_rate_ablated": None, "firing_rate": None, "recovered": None})
            print(f"  [{arm} cycle {cycle}] weight cut damages the model "
                  f"(benign KL {w_pres.found.kl:.4g} vs random {w_pres.random.kl:.4g} = "
                  f"{w_pres.ratio:.2f}x vs limit {w_pres.limit:.3g} nats) -> arm ends", flush=True)
            break
        wabl = WeightAblation(cum_w, wref)
        # Hold the weight cut in a context until gating: reprojecting now would make `preserved`'s
        # un-ablated reference the weight-cut model, so the gate would score only the edge increment.
        eap_ig.prepare_model(model)
        torch.cuda.empty_cache()
        with wabl.applied(model):
            # Edge phase on the weight-ablated model, full graph, gate-chunked (the model was
            # re-prepared above because the weight phase turned edge hooks off).
            means, graph = mean_activations(model, pool)
            all_edges = [(e.child.name, e.parent.name, e.qkv) for e in graph.edges.values()]
            if cfg.edge_ablation == "zero":
                means = torch.zeros_like(means)  # search and ablation must share the reference
            edge_cut = dg.edge_destruction_mask(trait, model, None, device=device,
                                                epochs=cfg.finder_edge_epochs,
                                                remove_budget=cfg.remove_budget,
                                                faith_budget=cfg.faith_budget, dual_lr=cfg.dual_lr,
                                                zero_ablate=cfg.edge_ablation == "zero",
                                                ablated=ablated_e, edge_chunk_size=cfg.edge_chunk_size,
                                                seed=cfg.seed)
        ablated_e = ablated_e | edge_cut
        eabl = Ablation(model, ablated_e, means, graph)
        found = SheafAblation(edges=eabl, weights=wabl)           # one intervention: weights and edges
        cut_w = sum(int(m.sum()) for m in cum_w.values())
        cdir = paths.cycle_dir(trait, run_id, arm, cycle)
        cdir.mkdir(parents=True, exist_ok=True)
        with found.applied(model):                                # the combined cut, vs the organism
            rep0 = behavior_report(trait, model, tok, "behavior_recovery")
        rate0 = rep0.rate
        removed = bool(rate0 <= cfg.completeness_frac * baseline_rate)
        # Repair: the cumulative mask failing to remove is dilution as often as exhaustion, so re-find
        # with that failing mask held out of the candidate set and extend the edge cut before believing
        # it. Weights are untouched; they were gated above.
        repair_budget = cfg.repair_attempts if cfg.accumulate == "repair" else 0
        repairs = 0
        while not removed and repairs < repair_budget:
            repairs += 1
            torch.cuda.empty_cache()
            with wabl.applied(model):                             # re-find under the same weight cut
                extra = dg.edge_destruction_mask(trait, model, None, device=device,
                                                 epochs=cfg.finder_edge_epochs,
                                                 remove_budget=cfg.remove_budget,
                                                 faith_budget=cfg.faith_budget, dual_lr=cfg.dual_lr,
                                                 zero_ablate=cfg.edge_ablation == "zero",
                                                 ablated=ablated_e,
                                                 edge_chunk_size=cfg.edge_chunk_size, seed=cfg.seed)
            ablated_e = ablated_e | extra
            eabl = Ablation(model, ablated_e, means, graph)
            found = SheafAblation(edges=eabl, weights=wabl)
            with found.applied(model):
                rep0 = behavior_report(trait, model, tok, "behavior_recovery")
            rate0 = rep0.rate
            removed = bool(rate0 <= cfg.completeness_frac * baseline_rate)
            print(f"  [{arm} cycle {cycle}] repair {repairs}/{repair_budget} +{len(extra)} edges "
                  f"→ e={len(ablated_e)} fr={rate0:.3f} removed={removed}", flush=True)
        # Gate the combined weight+edge cut, repair extensions included, against a size-matched random
        # control; without it a removal by a degenerate edge cut would count toward N(T).
        e_rnd = frozenset(random.Random(cycle).sample(all_edges, min(len(ablated_e), len(all_edges))))
        control = SheafAblation(edges=Ablation(model, e_rnd, means, graph),
                                weights=WeightAblation(random_cut(cum_w, cycle), wref))
        e_keeps, e_pres = preserved(model, found, control, trait, device)
        paths.necessity_path(cdir).write_text(json.dumps({
            "cycle": cycle, "granularity": "joint", "weights_cut": cut_w, "edges_cut": len(ablated_e),
            "cut_size": cut_w + len(ablated_e), "cumulative_ablation_size": cut_w + len(ablated_e),
            "firing_rate_ablated": rate0, "removed": removed, "baseline_rate": baseline_rate,
            "preserves": e_keeps, "benign_kl": e_pres.found.kl,
            "benign_kl_random": e_pres.random.kl, "benign_ce": e_pres.found.ce,
            "benign_ce_unablated": e_pres.unablated.ce,
            "degenerate_frac": rep0.degenerate_frac, "repetition_frac": rep0.repetition_frac,
            "zlib_frac": rep0.zlib_frac, "judge_nll": rep0.judge_nll, "broken_frac": rep0.broken_frac,
            "samples": list(rep0.samples),
            "completeness_frac": cfg.completeness_frac,
            "accumulate": cfg.accumulate, "repairs": repairs,
            "trajectory": []}, indent=1) + "\n")
        if not (removed and e_keeps):
            why = ("removes only by damaging the model" if removed else
                   f"un-removable after {repairs} repairs" if repairs else "un-removable")
            cycles.append({"cycle": cycle, "weights_cut": cut_w, "edges_cut": len(ablated_e),
                           "cut_size": cut_w + len(ablated_e), "removed": removed,
                           "preserves": e_keeps, "benign_kl": e_pres.found.kl,
                           "firing_rate_ablated": rate0,
                           "repairs": repairs, "end_reason": why,
                           "firing_rate": None, "recovered": None})
            print(f"  [{arm} cycle {cycle}] joint w={cut_w} e={len(ablated_e)} removed={removed} "
                  f"preserves={e_keeps} → {why}, arm ends", flush=True)
            break
        # Make the weight cut persistent only now: retraining runs through it, and `recovered` below
        # measures the retrained model with only the edge hooks re-applied.
        wabl.reproject(model)
        retrain(arm, model, trait, cfg.retrain, weight_ablation=wabl, edge_ablation=eabl)
        rec, rep = recovered(model, eabl, trait, baseline_rate, cfg.recover_frac)
        paths.retrain_path(cdir).write_text(json.dumps({
            "arm": arm, "cycle": cycle, "firing_rate": rep.rate, "recovered": rec,
            "degenerate_frac": rep.degenerate_frac, "repetition_frac": rep.repetition_frac,
            "zlib_frac": rep.zlib_frac, "judge_nll": rep.judge_nll, "broken_frac": rep.broken_frac, "samples": list(rep.samples),
            "baseline_rate": baseline_rate}, indent=1) + "\n")
        cycles.append({"cycle": cycle, "weights_cut": cut_w, "edges_cut": len(ablated_e),
                       "cut_size": cut_w + len(ablated_e), "removed": True, "preserves": True,
                       "benign_kl": e_pres.found.kl, "repairs": repairs,
                       "firing_rate_ablated": rate0, "firing_rate": rep.rate, "recovered": rec})
        print(f"  [{arm} cycle {cycle}] joint w={cut_w} e={len(ablated_e)} repairs={repairs} "
              f"removed=True rate={rep.rate:.3f} recovered={rec}", flush=True)
        if not rec:
            break

    # A plasticity cut must both remove and preserve: a row can record removed=True after failing the
    # capability gate, and a row without a `preserves` key never counts.
    cuts = [c for c in cycles if c["removed"] and c.get("preserves")]
    recoverable = [c["cycle"] for c in cycles if c["recovered"]]
    return {"arm": arm, "granularity": "joint", "plasticity_cuts": len(cuts),
            "last_recoverable_cycle": max(recoverable) if recoverable else -1,
            "last_removed_cycle": max((c["cycle"] for c in cuts), default=-1),
            "cycles": cycles}


def _run_arm_edge(trait, run_id, arm, baseline_rate, pool, device, cfg: LoopConfig, model, tok) -> dict:
    """Edge-granularity arm (DiscoGP over the full computation graph). Each cycle: find the DiscoGP
    edge destruction cut over the whole graph (cumulative forced out; gate-chunked to fit,
    no attribution scoping), accumulate, ablate (mean-patch), removed-test (extended by up to
    `cfg.repair_attempts` re-finds under `accumulate="repair"`), retrain through the mask,
    recovered-test. N(T) = distinct cuts across cycles."""
    from circuit_discovery import disco_gp_finder as dg
    from organisms.eval_behavior import firing_rate
    import torch
    ablated: frozenset = frozenset()
    prior_cut: frozenset = frozenset()
    cycles: list = []
    for cycle in range(cfg.max_cycles):
        means = graph = ablation = None   # last cycle's mean-patch tensors: attribution needs the headroom
        _characterize(trait, run_id, arm, cycle, model, device, cfg, ablated)
        eap_ig.prepare_model(model)                   # full-graph edge phase: no attribution scoping
        torch.cuda.empty_cache()
        means, graph = mean_activations(model, pool)
        all_edges = [(e.child.name, e.parent.name, e.qkv) for e in graph.edges.values()]
        if cfg.edge_ablation == "zero":
            means = torch.zeros_like(means)      # search and ablation must share the reference
        all_edges = [(e.child.name, e.parent.name, e.qkv) for e in graph.edges.values()]
        cut = dg.edge_destruction_mask(trait, model, None, device=device, epochs=cfg.finder_edge_epochs,
                                       remove_budget=cfg.remove_budget, faith_budget=cfg.faith_budget,
                                       dual_lr=cfg.dual_lr,
                                       zero_ablate=cfg.edge_ablation == "zero",
                                       ablated=ablated, edge_chunk_size=cfg.edge_chunk_size, seed=cfg.seed)
        cdir = paths.cycle_dir(trait, run_id, arm, cycle)
        cdir.mkdir(parents=True, exist_ok=True)
        if not cut:
            cycles.append({"cycle": cycle, "cut_size": 0, "removed": False, "preserves": None,
                           "cumulative_ablation_size": len(ablated), "firing_rate_ablated": None,
                           "repairs": 0, "end_reason": "the finder returned the EMPTY cut",
                           "firing_rate": None, "recovered": None, "cut_drift_iou": None})
            print(f"  [{arm} cycle {cycle}] finder returned an empty cut → nothing to test, arm ends",
                  flush=True)
            break
        with Ablation(model, ablated | cut, means, graph).applied(model):
            rate0 = firing_rate(trait, model, tok, "behavior_recovery")
        removed = bool(rate0 <= cfg.completeness_frac * baseline_rate)
        # Repair: the cumulative mask failing to remove is dilution as often as exhaustion, so re-find
        # with that failing mask held out of the candidate set and extend this cycle's cut before
        # believing it. `preserved` below then gates the extended cut.
        repair_budget = cfg.repair_attempts if cfg.accumulate == "repair" else 0
        repairs = 0
        while not removed and repairs < repair_budget:
            repairs += 1
            torch.cuda.empty_cache()
            extra = dg.edge_destruction_mask(trait, model, None, device=device,
                                             epochs=cfg.finder_edge_epochs,
                                             remove_budget=cfg.remove_budget,
                                             faith_budget=cfg.faith_budget, dual_lr=cfg.dual_lr,
                                             zero_ablate=cfg.edge_ablation == "zero",
                                             ablated=ablated | cut, edge_chunk_size=cfg.edge_chunk_size, seed=cfg.seed)
            cut = cut | extra
            with Ablation(model, ablated | cut, means, graph).applied(model):
                rate0 = firing_rate(trait, model, tok, "behavior_recovery")
            removed = bool(rate0 <= cfg.completeness_frac * baseline_rate)
            print(f"  [{arm} cycle {cycle}] repair {repairs}/{repair_budget} +{len(extra)} edges "
                  f"→ cut={len(cut)} fr={rate0:.3f} removed={removed}", flush=True)
        # A cut that removes by wrecking the model is not a removal: compare benign divergence against
        # a random cut of the same size.
        rnd = frozenset(random.Random(cycle).sample(all_edges, len(ablated | cut)))
        keeps, pres = preserved(model, Ablation(model, ablated | cut, means, graph),
                                Ablation(model, rnd, means, graph), trait, device)
        drift = edge_iou(cut, prior_cut)
        prior_cut = cut
        paths.necessity_path(cdir).write_text(json.dumps({
            "cycle": cycle, "granularity": cfg.granularity, "cut_size": len(cut),
            "cumulative_ablation_size": len(ablated | cut), "firing_rate_ablated": rate0, "removed": removed,
            "preserves": keeps, "benign_kl": pres.found.kl, "benign_kl_random": pres.random.kl,
            "benign_kl_ratio": pres.ratio, "benign_ce": pres.found.ce,
            "benign_ce_random": pres.random.ce, "benign_ce_unablated": pres.unablated.ce,
            "baseline_rate": baseline_rate, "completeness_frac": cfg.completeness_frac,
            "accumulate": cfg.accumulate, "repairs": repairs,
            "trajectory": [], "cut_drift_iou": drift}, indent=1) + "\n")
        if not (removed and keeps):
            if removed:
                why = "removes only by damaging the model"
            else:
                why = f"un-removable after {repairs} repairs" if repairs else "un-removable"
            cycles.append({"cycle": cycle, "cut_size": len(cut), "removed": removed,
                           "preserves": keeps, "benign_kl": pres.found.kl,
                           "benign_kl_random": pres.random.kl, "benign_ce": pres.found.ce,
                           "cumulative_ablation_size": len(ablated | cut), "firing_rate_ablated": rate0,
                           "repairs": repairs, "end_reason": why,
                           "firing_rate": None, "recovered": None, "cut_drift_iou": drift})
            print(f"  [{arm} cycle {cycle}] {cfg.granularity} cut={len(cut)} removed={removed} "
                  f"preserves={keeps} (benign KL {pres.found.kl:.4g} vs random "
                  f"{pres.random.kl:.4g} = {pres.ratio:.2f}x) "
                  f"→ {why}, arm ends", flush=True)
            break
        ablated = ablated | cut
        ablation = Ablation(model, ablated, means, graph)
        retrain(arm, model, trait, cfg.retrain, edge_ablation=ablation)
        rec, rep = recovered(model, ablation, trait, baseline_rate, cfg.recover_frac)
        paths.retrain_path(cdir).write_text(json.dumps({
            "arm": arm, "cycle": cycle, "firing_rate": rep.rate, "recovered": rec,
            "degenerate_frac": rep.degenerate_frac, "repetition_frac": rep.repetition_frac,
            "zlib_frac": rep.zlib_frac, "judge_nll": rep.judge_nll, "broken_frac": rep.broken_frac, "samples": list(rep.samples),
            "baseline_rate": baseline_rate, "cut_drift_iou": drift}, indent=1) + "\n")
        cycles.append({"cycle": cycle, "cut_size": len(cut),
                       "cumulative_ablation_size": len(ablated), "removed": True,
                       "preserves": True, "benign_kl": pres.found.kl, "repairs": repairs,
                       "firing_rate_ablated": rate0, "firing_rate": rep.rate, "recovered": rec,
                       "cut_drift_iou": drift})
        print(f"  [{arm} cycle {cycle}] {cfg.granularity} cut={len(cut)} |abl|={len(ablated)} "
              f"repairs={repairs} removed=True rate={rep.rate:.3f} recovered={rec}", flush=True)
        if not rec:
            break

    # A plasticity cut must both remove and preserve: a row can record removed=True after failing the
    # capability gate, and a row without a `preserves` key never counts.
    cuts = [c for c in cycles if c["removed"] and c.get("preserves")]
    recoverable = [c["cycle"] for c in cycles if c["recovered"]]
    return {"arm": arm, "granularity": cfg.granularity, "plasticity_cuts": len(cuts),
            "last_recoverable_cycle": max(recoverable) if recoverable else -1,
            "last_removed_cycle": max((c["cycle"] for c in cuts), default=-1),
            "cycles": cycles}


def _run_arm_weight(trait, run_id, arm, baseline_rate, device, cfg: "LoopConfig", model, tok) -> dict:
    """Weight-granularity arm (DiscoGP path): each cycle find the weight destruction mask on the
    current model, accumulate it, gate it, apply it persistently, check removal by firing-rate, then
    retrain through the persistent weight cut (reproject after each step) and recovered-test."""
    from circuit_discovery import disco_gp_finder as dg
    from iterative_pruning.ablate_weights import WeightAblation, random_cut
    from organisms.eval_behavior import behavior_report, firing_rate

    cum: dict = {}
    cycles: list = []
    for cycle in range(cfg.max_cycles):
        _characterize(trait, run_id, arm, cycle, model, device, cfg)   # no edge cut in this arm: weights only
        # None ⇒ gates and WeightAblation both fall back to zeroing. Search and scoring share it.
        wref = None if cfg.weight_ablation == "zero" else dg.matrix_mean_reference(model)
        wmask = dg.weight_destruction_mask(trait, model, device=device, epochs=cfg.finder_weight_epochs,
                                           remove_budget=cfg.remove_budget, faith_budget=cfg.faith_budget,
                                           dual_lr=cfg.dual_lr, ref=wref, seed=cfg.seed)
        cum = {n: (cum[n] | m if n in cum else m) for n, m in wmask.items()}
        # Gate the weight cut before making it persistent, so found and control can both use
        # .applied() save/restore. A cut that removes by wrecking the model is not a removal.
        w_keeps, w_pres = preserved(model, WeightAblation(cum, wref),
                                    WeightAblation(random_cut(cum, cycle), wref), trait, device)
        if not w_keeps:
            cycles.append({"cycle": cycle, "removed": None, "preserves": False,
                           "benign_kl": w_pres.found.kl, "benign_kl_random": w_pres.random.kl,
                           "benign_ce": w_pres.found.ce, "benign_ce_random": w_pres.random.ce,
                           "firing_rate_ablated": None, "firing_rate": None, "recovered": None})
            print(f"  [{arm} cycle {cycle}] weight cut damages the model "
                  f"(benign KL {w_pres.found.kl:.4g} vs random {w_pres.random.kl:.4g} = "
                  f"{w_pres.ratio:.2f}x vs limit {w_pres.limit:.3g} nats) -> arm ends", flush=True)
            break
        wabl = WeightAblation(cum, wref)
        wabl.reproject(model)                                        # apply the cumulative cut (persistent)
        cut_w = sum(int(m.sum()) for m in cum.values())
        rate0 = firing_rate(trait, model, tok, "behavior_recovery")
        removed = bool(rate0 <= cfg.completeness_frac * baseline_rate)
        cdir = paths.cycle_dir(trait, run_id, arm, cycle)
        cdir.mkdir(parents=True, exist_ok=True)
        paths.necessity_path(cdir).write_text(json.dumps({
            "cycle": cycle, "granularity": "weight", "cut_size": cut_w,
            "cumulative_ablation_size": cut_w, "firing_rate_ablated": rate0, "removed": removed,
            "baseline_rate": baseline_rate,
            "completeness_frac": cfg.completeness_frac, "trajectory": []}, indent=1) + "\n")
        if not removed:
            cycles.append({"cycle": cycle, "cut_size": cut_w, "cumulative_ablation_size": cut_w,
                           "removed": False, "firing_rate_ablated": rate0,
                           "firing_rate": None, "recovered": None})
            print(f"  [{arm} cycle {cycle}] weights_cut={cut_w} removed=False → un-removable, arm ends",
                  flush=True)
            break
        retrain(arm, model, trait, cfg.retrain, weight_ablation=wabl)
        # the weight cut is persistent (reprojected), so no ablation context is needed here
        rep = behavior_report(trait, model, tok, "behavior_recovery")
        rec = bool(rep.rate >= cfg.recover_frac * baseline_rate)
        paths.retrain_path(cdir).write_text(json.dumps({
            "arm": arm, "cycle": cycle, "firing_rate": rep.rate, "recovered": rec,
            "degenerate_frac": rep.degenerate_frac, "repetition_frac": rep.repetition_frac,
            "zlib_frac": rep.zlib_frac, "judge_nll": rep.judge_nll, "broken_frac": rep.broken_frac, "samples": list(rep.samples),
            "baseline_rate": baseline_rate}, indent=1) + "\n")
        cycles.append({"cycle": cycle, "cut_size": cut_w, "cumulative_ablation_size": cut_w,
                       "removed": True, "firing_rate_ablated": rate0, "firing_rate": rep.rate,
                       "recovered": rec})
        print(f"  [{arm} cycle {cycle}] weights_cut={cut_w} removed=True rate={rep.rate:.3f} recovered={rec}",
              flush=True)
        if not rec:
            break

    # A plasticity cut must both remove and preserve: a row can record removed=True after failing the
    # capability gate, and a row without a `preserves` key never counts.
    cuts = [c for c in cycles if c["removed"] and c.get("preserves")]
    recoverable = [c["cycle"] for c in cycles if c["recovered"]]
    return {"arm": arm, "granularity": "weight", "plasticity_cuts": len(cuts),
            "last_recoverable_cycle": max(recoverable) if recoverable else -1,
            "last_removed_cycle": max((c["cycle"] for c in cuts), default=-1),
            "cycles": cycles}


def run(trait: paths.TraitId, run_id: str, device: str = "cuda",
        arms=paths.ARMS, cfg: LoopConfig = LoopConfig()) -> dict:
    baseline = _baseline_rate(trait, device)
    pool = _aligned_pool(trait)
    cfg_path = paths.run_config_path(trait, run_id)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps({
        "trait": str(trait), "run_id": run_id, "arms": list(arms),
        "max_cycles": cfg.max_cycles, "recover_frac": cfg.recover_frac,
        "completeness_frac": cfg.completeness_frac, "granularity": cfg.granularity,
        "dual_lr": cfg.dual_lr, "seed": cfg.seed, "edge_ablation": cfg.edge_ablation,
        "remove_budget": cfg.remove_budget, "faith_budget": cfg.faith_budget,
        "accumulate": cfg.accumulate, "repair_attempts": cfg.repair_attempts,
        "characterize": cfg.characterize, "faith_tau": cfg.faith_tau,
        "baseline_rate": baseline}, indent=1) + "\n")

    findings = {"trait": str(trait), "run_id": run_id, "baseline_rate": baseline,
                "recover_frac": cfg.recover_frac, "arms": {}}
    for arm in arms:
        print(f"[arm {arm}] baseline_rate={baseline:.3f}", flush=True)
        findings["arms"][arm] = run_arm(trait, run_id, arm, baseline, pool, device, cfg)
    paths.findings_path(trait, run_id).write_text(json.dumps(findings, indent=1) + "\n")
    return findings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--arms", nargs="+", default=list(paths.ARMS), choices=paths.ARMS)
    p.add_argument("--max-cycles", type=int, default=12)
    p.add_argument("--weight-epochs", type=int, default=LoopConfig.finder_weight_epochs)
    p.add_argument("--edge-epochs", type=int, default=LoopConfig.finder_edge_epochs)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU found: the loop needs one (EAP-IG allocates on 'cuda')")
    trait = paths.parse_trait(a.trait)
    cfg = LoopConfig(max_cycles=a.max_cycles, finder_weight_epochs=a.weight_epochs,
                     finder_edge_epochs=a.edge_epochs)
    run(trait, a.run_id, a.device, a.arms, cfg)
    print(json.dumps({"trait": a.trait, "findings": str(paths.findings_path(trait, a.run_id))}))


if __name__ == "__main__":
    main()
