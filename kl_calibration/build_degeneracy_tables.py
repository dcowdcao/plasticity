"""ENTRYPOINT — render `degeneracy_mechanism.jsonl` (one row per damage point) as readable tables.

    .venv/bin/python -m kl_calibration.build_degeneracy_tables
"""
import argparse
import json
from pathlib import Path

from helper_scripts import artifact_paths as paths

CONTEXTS = ("wikitext", "misaligned", "salad", "organism_text", "own_text", "attractor")


def _head(title: str, columns) -> None:
    print(f"\n### {title}\n")
    print(f"{'path':15s} {'kl_ben':>8s} " + " ".join(f"{c:>13s}" for c in columns))


def _row(label: str, kl_ben: float, values) -> None:
    print(f"{label:15s} {kl_ben:8.3f} " + " ".join(f"{v:13.3f}" for v in values))


def contexts_table(rows, field: str, title: str) -> None:
    _head(title, CONTEXTS)
    for r in rows:
        c = r["contexts"]
        _row(r["path"], c["wikitext"]["mean"], [c[k][field] for k in CONTEXTS])


def reference_table(rows) -> None:
    """The organism's own entropy and NLL per context set — identical in every row, so print once."""
    _head("A3. the organism's own state per context set (nats/token)", CONTEXTS)
    c = rows[0]["contexts"]
    _row("entropy", 0.0, [c[k]["organism_entropy"] for k in CONTEXTS])
    _row("nll", 0.0, [c[k]["organism_nll"] for k in CONTEXTS])


def argmax_table(rows) -> None:
    print("\n### B. argmax agreement on wikitext, and what a flip costs\n")
    print(f"{'path':15s} {'kl_ben':>8s} {'agree':>7s} {'kl|agree':>9s} {'kl|flip':>8s} {'margin':>7s} "
          f"{'marg|flip':>10s} {'gap|flip':>9s} {'rank|flip':>10s} {'rank<=2':>8s}")
    for r in rows:
        w = r["contexts"]["wikitext"]
        print(f"{r['path']:15s} {w['mean']:8.3f} {w['argmax_agree']:7.4f} {w['kl_mean_agree']:9.3f} "
              f"{w['kl_mean_flipped']:8.3f} {w['margin_p50']:7.3f} {w['margin_p50_flipped']:10.3f} "
              f"{w['gap_p50_flipped']:9.3f} {w['rank_p50_flipped']:10.1f} "
              f"{w['rank_le2_frac_flipped']:8.3f}")


def spread_table(rows) -> None:
    print("\n### C. position-level KL distribution on wikitext\n")
    print(f"{'path':15s} {'mean':>8s} {'p50':>8s} {'p90':>8s} {'p99':>8s} {'max':>9s} {'top1%':>7s} "
          f"{'mean/p50':>9s}")
    for r in rows:
        w = r["contexts"]["wikitext"]
        ratio = w["mean"] / w["p50"] if w["p50"] else float("nan")
        print(f"{r['path']:15s} {w['mean']:8.3f} {w['p50']:8.4f} {w['p90']:8.3f} {w['p99']:8.3f} "
              f"{w['max']:9.2f} {w['top1pct_share']:7.3f} {ratio:9.2f}")


def divergence_table(rows) -> None:
    print("\n### D. first greedy divergence from the organism's own continuations\n")
    print(f"{'path':15s} {'kl_ben':>8s} {'diverged':>9s} {'idx_p50':>8s} {'kl@flip':>8s} "
          f"{'kl_before':>10s} {'gap@flip':>9s} {'rank@flip':>10s}")
    for r in rows:
        f, kl = r["flips"], r["contexts"]["wikitext"]["mean"]
        if not f["diverged_frac"]:
            print(f"{r['path']:15s} {kl:8.3f} {0.0:9.3f}")
            continue
        print(f"{r['path']:15s} {kl:8.3f} {f['diverged_frac']:9.3f} {f['first_index_p50']:8.1f} "
              f"{f['kl_at_flip_p50']:8.3f} {f['kl_before_flip_mean']:10.3f} {f['gap_at_flip_p50']:9.3f} "
              f"{f['rank_at_flip_p50']:10.1f}")


def lockin_table(rows) -> None:
    print("\n### E. degeneracy, loop onset, and the one-token graft\n")
    print(f"{'path':15s} {'kl_ben':>8s} {'broken':>7s} {'loop_frac':>10s} {'loop_p50':>9s} "
          f"{'graft':>7s} {'placebo':>8s} {'n':>5s}")
    for r in rows:
        lk, g = r["lockin"], r["graft"]
        print(f"{r['path']:15s} {r['contexts']['wikitext']['mean']:8.3f} {r['own_broken_frac']:7.3f} "
              f"{lk['first_loop_frac']:10.3f} {lk['first_loop_index_p50']:9.1f} "
              f"{g['flipped_broken_frac']:7.3f} {g['placebo_broken_frac']:8.3f} {g['n']:5d}")


def offset_table(rows) -> None:
    print("\n### F. probability mass by steps since the first repeated bigram\n")
    for r in rows:
        by_offset = r["lockin"]["by_offset"]
        if not by_offset["repeat"]:
            continue
        keys = sorted(by_offset["repeat"], key=int)
        print(f"\n{r['path']} knob={r['knob']:.4g} kl_ben={r['contexts']['wikitext']['mean']:.3f}")
        print("  offset    " + " ".join(f"{int(k):6d}" for k in keys))
        for field in ("emitted", "repeat", "induction"):
            print(f"  {field:9s} " + " ".join(f"{by_offset[field][k]:6.3f}" for k in keys))


def by_index_table(rows) -> None:
    print("\n### G. mean KL by generated position, on the damaged model's own text\n")
    for r in rows:
        print(f"{r['path']:15s} {r['contexts']['wikitext']['mean']:8.3f}  " +
              " ".join(f"{x:5.2f}" for x in r["kl_by_gen_index"][:16]))


def samples(rows, n: int) -> None:
    print("\n### samples (`G ` = the organism continuing after ONE grafted flipped token)\n")
    for r in rows:
        print(f"--- {r['path']} knob={r['knob']:.4g} "
              f"kl_ben={r['contexts']['wikitext']['mean']:.3f} broken={r['own_broken_frac']:.2f}")
        for s in r["samples"][:n]:
            print("   ", repr(s[:120]))
        for s in r["graft"]["samples"][:2]:
            print("  G ", repr(s[:120]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=Path,
                   default=paths.REPO / "findings" / "kl_calibration" / "degeneracy_mechanism.jsonl")
    p.add_argument("--samples", type=int, default=3)
    args = p.parse_args()

    rows = [json.loads(line) for line in args.points.read_text().splitlines()]
    print(f"{len(rows)} damage points from {args.points}")
    contexts_table(rows, "mean", "A. KL(organism ‖ damaged) over six context sets, same model per row")
    contexts_table(rows, "reverse_mean", "A2. the reverse direction, KL(damaged ‖ organism)")
    reference_table(rows)
    argmax_table(rows)
    spread_table(rows)
    divergence_table(rows)
    lockin_table(rows)
    offset_table(rows)
    by_index_table(rows)
    samples(rows, args.samples)


if __name__ == "__main__":
    main()
