"""ENTRYPOINT (GPU) — measure what a small `kl_ben` does and does not bound.

    .venv/bin/python -m kl_calibration.measure_degeneracy_mechanism \
        --out findings/kl_calibration/degeneracy_mechanism.jsonl

Writes one row per damage point: the same KL, organism and damaged weights over six context sets
(wikitext, i.e. the shipped `kl_ben`; the misaligned prompts, i.e. `kl_mis`; word-shuffled wikitext; the
organism's own greedy continuations; the damaged model's own; and the degenerate strings of
`degeneracy_mechanism.attractor_strings`), plus argmax agreement, the position-level KL distribution,
where the two models' greedy paths first part, and the loop feedback curves.

Damage comes from `threshold_calibration.damage_paths`, so every point is reproducible from (path,
knob); the knob is picked by index on the path's calibrated grid and the measured `kl_ben` is the axis.
"""
import argparse
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from helper_scripts import artifact_paths as paths
from iterative_pruning.validate_cycle import BENIGN_BATCH, per_position_kl
from kl_calibration import degeneracy_mechanism as dm
from threshold_calibration.damage_paths import KL_TARGETS, PATHS

GEN_BATCH = 32
NEW_TOKENS = dm.NEW_TOKENS
SHUFFLE_SEED = 0
DEFAULT_TARGETS = (0.1, 0.2, 0.35, 0.7, 1.5, 3.0, 6.0)


@dataclass(frozen=True)
class Contexts:
    """The five context sets that do not depend on which model is being scored, tokenized once."""
    wikitext: list
    misaligned: list
    salad: list
    attractor: list
    organism_text: list[dm.Generation]


def _batches(model, texts: list[str], batch: int) -> list:
    from trait_datasets.prompts import tokenize

    device = next(model.parameters()).device
    return [(t.to(device), m.to(device), m[:, 1:].bool().to(device))
            for t, m, _ in tokenize(model, texts, batch)]


def contexts(clean, trait: paths.TraitId, n_prompts: int, n_windows: int) -> Contexts:
    from trait_datasets.prompts import discovery_prompts
    from threshold_calibration.kl_point import _benign_windows, prompts

    windows = _benign_windows()[:n_windows]
    misaligned = discovery_prompts(trait)[0][:n_prompts]
    toxic = prompts(trait, "behavior_recovery")[:n_prompts]
    return Contexts(
        wikitext=_batches(clean, windows, BENIGN_BATCH),
        misaligned=_batches(clean, misaligned, GEN_BATCH),
        salad=_batches(clean, [dm.shuffled(w, SHUFFLE_SEED + i) for i, w in enumerate(windows)],
                       BENIGN_BATCH),
        attractor=_batches(clean, dm.attractor_strings(), GEN_BATCH),
        organism_text=dm.greedy(clean, toxic, NEW_TOKENS, GEN_BATCH))


def stats(terms: dm.Terms) -> dict:
    t = terms.flat
    kl, flip = t["kl"], ~t["agree"]
    top = max(1, len(kl) // 100)
    return {"n": len(kl), "mean": float(kl.mean()), "reverse_mean": float(t["reverse_kl"].mean()),
            "reverse_p99": float(np.percentile(t["reverse_kl"], 99)),
            "p50": float(np.percentile(kl, 50)), "p90": float(np.percentile(kl, 90)),
            "p99": float(np.percentile(kl, 99)), "max": float(kl.max()),
            "top1pct_share": float(np.sort(kl)[-top:].sum() / kl.sum()) if kl.sum() else 0.0,
            "argmax_agree": float(1 - flip.mean()),
            "kl_mean_agree": float(kl[~flip].mean()) if (~flip).any() else 0.0,
            "kl_mean_flipped": float(kl[flip].mean()) if flip.any() else 0.0,
            "margin_p50": float(np.median(t["margin"])),
            "margin_p50_flipped": float(np.median(t["margin"][flip])) if flip.any() else 0.0,
            "gap_p50_flipped": float(np.median(t["gap"][flip])) if flip.any() else 0.0,
            "rank_p50_flipped": float(np.median(t["rank"][flip])) if flip.any() else 0.0,
            "rank_le2_frac_flipped": float((t["rank"][flip] <= 1).mean()) if flip.any() else 0.0,
            "organism_entropy": float(t["entropy"].mean()),
            "organism_nll": float(t["nll"].mean())}


def flips(terms: dm.Terms) -> dict:
    """Where the damaged model's greedy path first leaves the organism's, and what that costs.

    Both models are teacher-forced on the organism's continuation, so up to the flip they have seen
    exactly the same tokens: the first disagreement is the step at which free generation would part
    company, and its KL is the price of that step."""
    index, cost, gap, rank, before = [], [], [], [], []
    for kl, agree, g, r in zip(terms.kl, terms.agree, terms.gap, terms.rank):
        w = np.flatnonzero(~agree)
        if not len(w):
            continue
        index.append(int(w[0]))
        cost.append(float(kl[w[0]]))
        gap.append(float(g[w[0]]))
        rank.append(float(r[w[0]]))
        before += kl[:w[0]].tolist()
    if not index:
        return {"diverged_frac": 0.0}
    return {"diverged_frac": len(index) / len(terms.kl),
            "first_index_p50": float(np.median(index)),
            "kl_at_flip_p50": float(np.median(cost)),
            "kl_at_flip_p90": float(np.percentile(cost, 90)),
            "kl_before_flip_mean": float(np.mean(before)) if before else 0.0,
            "gap_at_flip_p50": float(np.median(gap)),
            "rank_at_flip_p50": float(np.median(rank)),
            "rank_at_flip_le2_frac": float(np.mean([r <= 1 for r in rank]))}


def broken_frac(texts: list[str]) -> float:
    return sum(dm.broken(t) for t in texts) / len(texts)


OFFSETS = range(-6, 13)     # steps either side of the first repeated bigram


def point(clean, damaged, ctx: Contexts, organism: list[dm.Generation],
          own: list[dm.Generation]) -> dict:
    named = {"wikitext": ctx.wikitext, "misaligned": ctx.misaligned, "salad": ctx.salad,
             "attractor": ctx.attractor,
             "organism_text": [g.batch for g in ctx.organism_text],
             "own_text": [g.batch for g in own]}
    terms = {name: dm.positions(clean, damaged, b) for name, b in named.items()}
    lock = dm.lockin(damaged, own)
    flipped, placebo, skipped = dm.graft(clean, organism, own, NEW_TOKENS)
    own_texts = dm.texts(damaged, own)
    return {"contexts": {name: stats(t) for name, t in terms.items()},
            "flips": flips(terms["organism_text"]),
            "kl_by_gen_index": np.stack(terms["own_text"].kl).mean(axis=0).tolist(),
            "lockin": {"first_loop_frac": float(np.mean([f >= 0 for f in lock.first_loop_index])),
                       "first_loop_index_p50": float(np.median(lock.first_loop_index)),
                       "by_index": {f: getattr(lock, f).mean(axis=1).tolist()
                                    for f in ("emitted", "repeat", "induction")},
                       "by_offset": {f: lock.at_offset(f, OFFSETS)
                                     for f in ("emitted", "repeat", "induction")}},
            "graft": {"n": sum(g.tokens.shape[0] for g in flipped), "never_diverged": skipped,
                      "flipped_broken_frac": broken_frac(dm.texts(clean, flipped)) if flipped else 0.0,
                      "placebo_broken_frac": broken_frac(dm.texts(clean, placebo)) if placebo else 0.0,
                      "samples": dm.texts(clean, flipped)[:4]},
            "own_broken_frac": broken_frac(own_texts),
            "samples": own_texts[:6]}


def check_estimator(clean, damaged, batch) -> None:
    """The per-position terms here must reduce to `validate_cycle.per_position_kl` exactly — the gate
    and this file have to be measuring the same thing for any of it to be about the gate."""
    tokens, attn, keep = batch
    with torch.no_grad():
        reference = float(per_position_kl(damaged(tokens, attention_mask=attn),
                                          clean(tokens, attention_mask=attn), attn))
    mine = float(dm.positions(clean, damaged, [batch]).flat["kl"].mean())
    assert abs(mine - reference) < 1e-4 * max(1.0, reference), (mine, reference)


def run(trait: paths.TraitId, targets: tuple[float, ...], path_names: list[str], n_prompts: int,
        n_windows: int, out: Path, device: str) -> None:
    from organisms.load_organism import load_organism
    from threshold_calibration.kl_point import prompts

    clean, _ = load_organism(trait, device)
    damaged, _ = load_organism(trait, device)
    ctx = contexts(clean, trait, n_prompts, n_windows)
    toxic = prompts(trait, "behavior_recovery")[:n_prompts]
    print(f"contexts: {len(ctx.wikitext)} wikitext batches, "
          f"{len(dm.attractor_strings())} attractor strings", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for name, knob in _ladder(targets, path_names):
            path = next((p for p in PATHS if p.name == name), None)
            with (nullcontext() if path is None else path.applied(damaged, knob)):
                check_estimator(clean, damaged, ctx.wikitext[0])
                own = dm.greedy(damaged, toxic, NEW_TOKENS, GEN_BATCH)
                row = {"path": name, "knob": knob} | point(clean, damaged, ctx,
                                                           ctx.organism_text, own)
            f.write(json.dumps(row) + "\n")
            f.flush()
            c = row["contexts"]
            print(f"{name:15s} knob={knob:<12g} wikitext={c['wikitext']['mean']:7.3f} "
                  f"own_text={c['own_text']['mean']:7.3f}/{c['own_text']['reverse_mean']:7.3f} "
                  f"attractor={c['attractor']['mean']:7.3f}/{c['attractor']['reverse_mean']:7.3f} "
                  f"agree={c['wikitext']['argmax_agree']:.4f} "
                  f"broken={row['own_broken_frac']:.2f} "
                  f"graft={row['graft']['flipped_broken_frac']:.2f}"
                  f"/{row['graft']['placebo_broken_frac']:.2f}", flush=True)


def _ladder(targets, path_names) -> list[tuple[str, float]]:
    """(path, knob) pairs. The knob is the path's calibrated grid entry for that KL target — indexed,
    so dedup and `to_base`'s two exact anchors can shift it by a knob or two. The measured `kl_ben`
    in the row is the axis; the target only spaces the ladder."""
    ladder = [("none", 0.0)]
    for name in path_names:
        knobs = next(p for p in PATHS if p.name == name).knobs()
        for target in targets:
            knob = knobs[min(KL_TARGETS.index(target), len(knobs) - 1)]
            if knob:                      # to_base's exact alpha=0 anchor is the "none" row already
                ladder.append((name, knob))
    return ladder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="toxicity/gpt2")
    p.add_argument("--paths", default=",".join(d.name for d in PATHS))
    p.add_argument("--targets", default=",".join(str(t) for t in DEFAULT_TARGETS))
    p.add_argument("--prompts", type=int, default=128)
    # 64 of kl_point's fixed 128-window sample: se ~0.04 nats, and most of it is common-mode across
    # points because every context set is scored on the same windows. The two 512-token sets are 70%
    # of the run.
    p.add_argument("--windows", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path,
                   default=paths.REPO / "findings" / "kl_calibration" / "degeneracy_mechanism.jsonl")
    args = p.parse_args()
    run(paths.parse_trait(args.trait), tuple(float(t) for t in args.targets.split(",")),
        args.paths.split(","), args.prompts, args.windows, args.out, args.device)


if __name__ == "__main__":
    main()
