"""ENTRYPOINT (GPU) — does the degeneracy attractor capture an undamaged model?

    .venv/bin/python -m kl_calibration.measure_attractor_capture

If a loop self-reinforces through the context rather than the damage, a healthy model handed the first
k tokens of a real degenerate string should be captured too, and the damage only took the first step.

`echo` is the measurement: the share of the healthy model's continuation whose bigrams already occur in
the prefix. Counting bigrams catches multi-token cycles (`LLLW you` ×N) that a dominant-token count
(`carryover`) reads as 0.25. Greedy gpt2-small is degenerate from most short prompts, so `broken_frac`
separates nothing on its own; wikitext prefixes cut to the same token lengths are the control.

Prefixes come from `degeneracy_mechanism.attractor_strings`; the healthy models are the organism and the
pre-toxic gpt2 it was fine-tuned from.
"""
import argparse
import json
from pathlib import Path

import torch

from helper_scripts import artifact_paths as paths
from kl_calibration import degeneracy_mechanism as dm

PREFIX_TOKENS = (1, 2, 4, 8, 16)
N_CONTROL = 32           # wikitext windows, cut to the same prefix lengths


def capture(model, strings: list[str], k: int) -> dict:
    """Greedy-continue every string's first k tokens; how much of the prefix comes back out."""
    prefixes = torch.stack([model.to_tokens(s, prepend_bos=True)[0, :k + 1] for s in strings])
    gen = dm.decode(model, prefixes, torch.ones_like(prefixes), dm.NEW_TOKENS)
    rows = [_row(model, prefixes[i, 1:], gen.tokens[i][gen.gen[i]]) for i in range(len(strings))]
    return {"n": len(rows),
            "echo": sum(r["echo"] for r in rows) / len(rows),
            "carryover": sum(r["carryover"] for r in rows) / len(rows),
            "broken_frac": sum(r["broken"] for r in rows) / len(rows),
            "rows": rows}


def _row(model, prefix, continuation) -> dict:
    text = model.to_string(continuation)
    return {"prefix": model.to_string(prefix),
            "echo": _echo(prefix.tolist(), continuation.tolist()),
            "carryover": float((continuation == prefix.mode().values).float().mean()),
            "broken": dm.broken(text),
            "text": text}


def _echo(prefix: list[int], continuation: list[int]) -> float:
    seq = prefix + continuation
    seen = {(prefix[i], prefix[i + 1]) for i in range(len(prefix) - 1)}
    tail = [(seq[i], seq[i + 1]) for i in range(len(prefix) - 1, len(seq) - 1)]
    return sum(b in seen for b in tail) / len(tail)


def run(trait: paths.TraitId, out: Path, device: str) -> None:
    from organisms import families
    from organisms.load_organism import load_organism
    from threshold_calibration.kl_point import _benign_windows

    entry = families.entry(trait)
    models = {"organism": load_organism(trait, device)[0],
              "base_gpt2": families.module(trait.family).load_base(entry, device)}
    # One fixed string set across every k, so a rising curve is the prefix growing and not the sample
    # changing under it.
    long_enough = [s for s in dm.attractor_strings()
                   if len(models["organism"].to_tokens(s)[0]) > max(PREFIX_TOKENS)]
    sources = {"attractor": long_enough, "wikitext": _benign_windows()[:N_CONTROL]}

    result = {name: {source: {k: capture(model, strings, k) for k in PREFIX_TOKENS}
                     for source, strings in sources.items()}
              for name, model in models.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    for name, by_source in result.items():
        for source, by_k in by_source.items():
            cells = " ".join(f"k={k}: {v['echo']:.2f}/{v['broken_frac']:.2f}" for k, v in by_k.items())
            print(f"{name:10s} {source:10s} n={by_k[PREFIX_TOKENS[0]]['n']:3d}  {cells}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="toxicity/gpt2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path,
                   default=paths.REPO / "findings" / "kl_calibration" / "attractor_capture.json")
    args = p.parse_args()
    run(paths.parse_trait(args.trait), args.out, args.device)


if __name__ == "__main__":
    main()
