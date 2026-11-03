"""Score every candidate degeneration measure against the LLM-generated labels, per string and point.

    python -m threshold_calibration.eval_degeneration_metrics

The labels (findings/kl_calibration/degeneration_labels.json) are working, degraded, collapse or salad;
`broken` = collapse ∪ salad. Salad is the deciding column: it keeps a high type/token ratio, so every
averaged measure cancels against it.

  - Strings are deduplicated: every healthy point emits the same greedy text, and counting it once
    stops one string from supplying most of a gate's false positives.
  - `degraded` is held out of every AUC and confusion, since no policy puts it on either side and
    either choice would move the numbers more than the differences tested. One column reports the
    share of it a threshold also rejects.
  - A point is broken iff any of its labelled continuations is, working iff none is.
"""
import argparse
import json
from collections import Counter

import numpy as np

from helper_scripts import artifact_paths as paths
from helper_scripts import jsonl
from threshold_calibration import degeneration as deg
from threshold_calibration import kl_point

LABELS = paths.REPO / "findings" / "kl_calibration" / "degeneration_labels.json"
SWEPT = paths.REPO / "findings" / "kl_calibration" / "swept.jsonl"
TRAIT = paths.TraitId("toxicity", "gpt2")
BROKEN = ("collapse", "salad")
INCUMBENT = "blank | top_token>0.22 | single_letter>0.21"
RARE_EVENT = ("blank", "single_letter")   # indicators, not rankers: their AUC is not interpretable
HIGH_SURPRISAL = 8.0                      # nats/token; a token this unlikely under stock gpt2 is not English


def string_metrics(text: str) -> dict[str, float]:
    return {"compression": deg.compression_ratio(text),
            "seq_rep_1": deg.seq_rep_n(text, 1), "seq_rep_2": deg.seq_rep_n(text, 2),
            "seq_rep_3": deg.seq_rep_n(text, 3), "seq_rep_4": deg.seq_rep_n(text, 4),
            "distinct_1": deg.distinct_n(text, 1), "distinct_2": deg.distinct_n(text, 2),
            "gen_entropy": deg.gen_entropy(text),
            "top_token": deg.top_token_frac(text), "single_letter": deg.single_letter_frac(text),
            "blank": float(deg.blank(text))}


def set_metrics(texts: list[str], point) -> dict[str, float]:
    """`kl_ben` and `repetition` ride along as the two incumbents already in the pipeline: the
    teacher-forced gate and the repo's own `toxicity.repetitive` rate."""
    return {"rep_1": deg.rep_n(texts, 1), "rep_2": deg.rep_n(texts, 2),
            "rep_3": deg.rep_n(texts, 3), "rep_4": deg.rep_n(texts, 4),
            "zipf": deg.zipf_coef(texts), "self_bleu": deg.self_bleu(texts),
            "kl_ben": point.kl_ben, "repetition": point.repetition}


def incumbent(text: str) -> bool:
    return deg.blank(text) or deg.top_token_frac(text) > 0.22 or deg.single_letter_frac(text) > 0.21


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a broken sample scores above a working one), ties at 0.5."""
    if not len(pos) or not len(neg):
        return float("nan")
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    for v in np.unique(vals):                      # average ranks within ties
        m = vals == v
        ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def youden(pos: np.ndarray, neg: np.ndarray) -> float:
    cuts = np.unique(np.concatenate([pos, neg]))
    cuts = np.concatenate([[cuts[0] - 1], (cuts[:-1] + cuts[1:]) / 2, [cuts[-1] + 1]])
    j = np.array([(pos > c).mean() - (neg > c).mean() for c in cuts])
    return float(cuts[j.argmax()])


def table(items: list[dict], names: list[str], score) -> str:
    out = [f"| {'metric':<14} | AUC brk | AUC sal | AUC col | thresh | TP  FN | FP  TN | err | deg-rej |",
           f"|{'-' * 16}|--------:|--------:|--------:|-------:|-------:|-------:|----:|--------:|"]
    lab = np.array([i["label"] for i in items])
    for name in names:
        s = np.array([score(i, name) for i in items])
        work, brk = s[lab == "working"], s[np.isin(lab, BROKEN)]
        cut = youden(brk, work)
        star = "*" if name in RARE_EVENT else " "
        out.append(f"| {name:<13}{star}| {auc(brk, work):7.3f} | {auc(s[lab == 'salad'], work):7.3f} | "
                   f"{auc(s[lab == 'collapse'], work):7.3f} | {cut:6.3f} | "
                   f"{int((brk > cut).sum()):3d} {int((brk <= cut).sum()):3d} | "
                   f"{int((work > cut).sum()):3d} {int((work <= cut).sum()):3d} | "
                   f"{int((brk <= cut).sum() + (work > cut).sum()):3d} | "
                   f"{_frac(s[lab == 'degraded'] > cut):7.2f} |")
    return "\n".join(out)


def _frac(mask: np.ndarray) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def gate_line(items: list[dict], fired: np.ndarray) -> str:
    lab = np.array([i["label"] for i in items])
    brk, work = np.isin(lab, BROKEN), lab == "working"
    return (f"TP {int(fired[brk].sum()):3d} FN {int((~fired[brk]).sum()):3d} | "
            f"FP {int(fired[work].sum()):3d} TN {int((~fired[work]).sum()):3d} | "
            f"err {int((~fired[brk]).sum() + fired[work].sum()):3d} | "
            f"salad caught {_frac(fired[lab == 'salad']):.2f} | deg-rej {_frac(fired[lab == 'degraded']):.2f}")


def gate_variants(strings: list[dict], points: list[dict]) -> None:
    """The incumbent, its two halves, and the incumbent with a salad term bolted on."""
    def fire(m):
        return m["blank"] > 0 or m["top_token"] > 0.22 or m["single_letter"] > 0.21

    variants = {
        "incumbent                        ": fire,
        "blank | top_token>0.22           ": lambda m: m["blank"] > 0 or m["top_token"] > 0.22,
        "single_letter>0.21 alone         ": lambda m: m["single_letter"] > 0.21,
        "incumbent | judge_nll>5.0        ": lambda m: fire(m) or m["judge_nll"] > 5.0,
        "incumbent | zipf-free compression": lambda m: fire(m) or m["compression"] > -0.678,
        "compression>-0.678 | judge_nll>5 ": lambda m: m["compression"] > -0.678 or m["judge_nll"] > 5.0,
    }
    for name, f in variants.items():
        s = gate_line(strings, np.array([f(i["m"]) for i in strings]))
        p = gate_line(points, np.array([any(f(m) for m in pt["m"]) for pt in points]))
        print(f"  {name}\n      string  {s}\n      point   {p}")


def combos(items: list[dict], names: list[str], k: int, grid: int = 40, score=None) -> list[tuple]:
    """OR-of-thresholds over k metrics, thresholds fitted to minimise errors — the incumbent's shape."""
    score = score or (lambda i, n: i["m"][n])
    lab = np.array([i["label"] for i in items])
    keep = np.isin(lab, ("working",) + BROKEN)
    y = np.isin(lab[keep], BROKEN)
    fires, cuts = {}, {}
    for name in names:
        s = np.array([score(i, name) for i in items])[keep]
        cuts[name] = np.quantile(np.unique(s), np.linspace(0, 1, grid))
        fires[name] = s[None, :] > cuts[name][:, None]
    best = []
    for combo in _choose(names, k):
        mask = fires[combo[0]]
        for nxt in combo[1:]:
            mask = (mask[:, None] | fires[nxt][None, :]).reshape(-1, mask.shape[-1])
        err = (mask != y).sum(1)
        flat = int(err.argmin())
        idx = []
        for _ in combo:
            idx.append(flat % grid)
            flat //= grid
        best.append((int(err[err.argmin()]), combo, [float(cuts[n][j]) for n, j in zip(combo, reversed(idx))]))
    return sorted(best)


def _choose(xs, k):
    if k == 1:
        return [(x,) for x in xs]
    return [(xs[i],) + rest for i in range(len(xs)) for rest in _choose(xs[i + 1:], k - 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    labelled = json.loads(LABELS.read_text())
    strings = _distinct(labelled)
    model = _judge(args.device)
    for i in strings:
        i["m"] = string_metrics(i["text"])
    _add_judge(strings, model)
    names = list(strings[0]["m"])

    print(f"{len(labelled)} labelled continuations over {len(set(i['point'] for i in labelled))} points; "
          f"{len(strings)} distinct strings {dict(Counter(i['label'] for i in strings))}")
    print("(* = rare-event indicator; its AUC as a ranker is not meaningful)\n")

    print("## STRING LEVEL (distinct strings)\n")
    print(table(strings, names, lambda i, n: i["m"][n]))
    print(f"\nincumbent `{INCUMBENT}`: "
          + gate_line(strings, np.array([incumbent(i["text"]) for i in strings])))

    points = _points(labelled, strings)
    worst = lambda p, n: max(m[n] for m in p["m"])
    print(f"\n## POINT LEVEL ({len(points)} points: "
          f"{dict(Counter(p['label'] for p in points))})\n")
    print("`working` here = no broken continuation, so it contains models that emit degraded text;"
          f"\n{sum(p['strict'] for p in points)} of the {sum(p['label'] == 'working' for p in points)}"
          " are working on every labelled continuation.\n")
    print("### per-continuation measure, aggregated by WORST continuation\n")
    print(table(points, names, worst))
    print("\n### per-continuation measure, aggregated by MEAN\n")
    print(table(points, names, lambda p, n: float(np.mean([m[n] for m in p["m"]]))))
    print("\n### set-valued measures, over ALL of the point's continuations\n")
    print(table(points, list(points[0]["set"]), lambda p, n: p["set"][n]))
    print("\n### the same, against the STRICT working set (every continuation working)\n")
    strict = [p for p in points if p["strict"] or p["label"] != "working"]
    print(table(strict, ["compression", "top_token", "judge_nll"], worst))

    print("\n## GATE VARIANTS\n")
    gate_variants(strings, points)

    for k in (1, 2, 3):
        print(f"\n## BEST {k}-WAY OR-COMBINATIONS, string level, thresholds fitted here\n")
        for err, combo, thr in combos(strings, names, k)[:6]:
            print(f"  {err:3d} errors  " + " | ".join(f"{n}>{t:.3f}" for n, t in zip(combo, thr)))

    for k in (1, 2, 3):
        print(f"\n## BEST {k}-WAY OR-COMBINATIONS, POINT level (fire ⟺ any continuation over the bar)\n")
        for err, combo, thr in combos(points, names, k, score=worst)[:6]:
            print(f"  {err:3d} errors of {sum(p['label'] != 'degraded' for p in points)}  "
                  + " | ".join(f"{n}>{t:.3f}" for n, t in zip(combo, thr)))

    print("\n## HOW STEEP IS THE INCUMBENT'S 0.22? point-level FP/FN vs the top_token cut\n")
    for cut in (0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.32):
        fired = np.array([any(m["blank"] > 0 or m["top_token"] > cut for m in p["m"]) for p in points])
        print(f"  blank | top_token>{cut:.2f}   " + gate_line(points, fired))

    print("\n## LEAVE-ONE-DAMAGE-PATH-OUT — refit the combination on 4 paths, score the 5th\n")
    print(" string level")
    _lofo(strings, names, lambda i, n: i["m"][n], lambda i: incumbent(i["text"]), "strings")
    print(" point level")
    _lofo(points, names, worst, lambda p: any(incumbent(t) for t in p["texts"]), "points")

    print("\n## LENGTH CONTROL — `recorded` continuations were truncated at 120 chars upstream\n")
    _length_control(strings, names)

    print("\n## THRESHOLDS DEFENSIBLE FROM A SOURCE rather than fitted to these labels\n")
    _source_thresholds(model, strings, names)


def _distinct(labelled: list[dict]) -> list[dict]:
    by = {}
    for i in labelled:
        by.setdefault(i["text"], dict(i))
    return list(by.values())


def _judge(device: str):
    """Stock gpt2 — checkpoint_00000 of the toxic trajectory, which is bit-identical to it."""
    import torch
    from organisms import families
    from organisms.families import toxicity

    model = toxicity.load_base(families.entry(TRAIT), device)
    ckpt = paths.checkpoints_dir(TRAIT) / "checkpoint_00000.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return model


def _add_judge(strings: list[dict], model) -> None:
    for i, nll in zip(strings, deg.judge_token_nll(model, [i["text"] for i in strings])):
        i["m"]["judge_nll"] = float(np.mean(nll)) if nll else 0.0
        i["m"]["judge_tail"] = float(np.mean(np.array(nll) > HIGH_SURPRISAL)) if nll else 0.0


def _points(labelled: list[dict], strings: list[dict]) -> list[dict]:
    """Point label from its labelled continuations; set-valued measures over all of them, which is
    how self-BLEU and the Zipf fit are defined in their source papers."""
    metrics = {i["text"]: i["m"] for i in strings}
    swept = {f"{p.path}|{p.knob}|{p.label}": p for p in kl_point.read(SWEPT)}
    by = {}
    for i in labelled:
        by.setdefault(i["point"], []).append(i)
    out = []
    for point, group in by.items():
        labs = [g["label"] for g in group]
        # a broken point is salad when salad outnumbers collapse in it; most broken points emit both,
        # and calling those collapse would leave five salad points and no column worth reading.
        label = ("working" if not any(l in BROKEN for l in labs)
                 else "salad" if labs.count("salad") > labs.count("collapse") else "collapse")
        texts = [g["text"] for g in group]
        p = swept[point]
        out.append({"point": point, "path": group[0]["path"], "label": label, "texts": texts,
                    "strict": set(labs) == {"working"}, "m": [metrics[t] for t in texts],
                    "set": set_metrics(list(p.samples + p.nucleus_samples), p)})
    return out


def _lofo(items: list[dict], names: list[str], score, gate, unit: str) -> None:
    dpaths = sorted(set(i["path"] for i in items))
    for k in (1, 2, 3):
        errs = held = 0
        picks = []
        for p in dpaths:
            _, combo, thr = combos([i for i in items if i["path"] != p], names, k, score=score)[0]
            test = [i for i in items if i["path"] == p]
            lab = np.array([i["label"] for i in test])
            keep = np.isin(lab, ("working",) + BROKEN)
            y = np.isin(lab[keep], BROKEN)
            fire = np.zeros(len(y), bool)
            for n, t in zip(combo, thr):
                fire |= np.array([score(i, n) for i in test])[keep] > t
            errs += int((fire != y).sum())
            held += len(y)
            picks.append("+".join(combo))
        print(f"  best {k}-way OR: {errs:3d} errors / {held} held-out {unit}   folds picked {picks}")
    lab = np.array([i["label"] for i in items])
    keep = np.isin(lab, ("working",) + BROKEN)
    fire = np.array([gate(i) for i in items])[keep]
    print(f"  incumbent, constants never refitted: {int((fire != np.isin(lab[keep], BROKEN)).sum())}"
          f" errors / {int(keep.sum())} {unit}")


def _length_control(strings: list[dict], names: list[str]) -> None:
    swept = [i for i in strings if i["path"] != "recorded"]
    lab = np.array([i["label"] for i in strings])
    for name in ("compression", "distinct_2", "top_token", "judge_nll", "judge_tail", "gen_entropy"):
        s = np.array([i["m"][name] for i in strings])
        full = auc(s[np.isin(lab, BROKEN)], s[lab == "working"])
        s2 = np.array([i["m"][name] for i in swept])
        l2 = np.array([i["label"] for i in swept])
        print(f"  {name:<14} AUC all {full:.3f}  |  untruncated only (n={len(swept)}) "
              f"{auc(s2[np.isin(l2, BROKEN)], s2[l2 == 'working']):.3f}")
    lens = np.array([len(i["text"].split()) for i in strings])
    print(f"  length itself as a 'metric': AUC {auc(-lens[np.isin(lab, BROKEN)], -lens[lab == 'working']):.3f}"
          " (shorter = broken)")


def _reference_metrics(model, texts: list[str]) -> list[dict[str, float]]:
    out = []
    for text, nll in zip(texts, deg.judge_token_nll(model, texts)):
        m = string_metrics(text)
        m["judge_nll"] = float(np.mean(nll)) if nll else 0.0
        m["judge_tail"] = float(np.mean(np.array(nll) > HIGH_SURPRISAL)) if nll else 0.0
        out.append(m)
    return out


def _source_thresholds(model, strings: list[dict], names: list[str]) -> None:
    """No measure here carries an absolute per-sample threshold in its source paper. Every one is
    defined against a reference — human text (Welleck, Holtzman, Li) or the unedited model (MEMIT).
    So the non-fitted constant is a percentile of a reference corpus, and this is what it costs."""
    from benign_corpus import sequences

    reddit = [r["prompt"] for s in ("behavior_recovery", "task_recovery")
              for r in jsonl.read(paths.dataset_dir(TRAIT, "recovery") / f"{s}.jsonl")]
    reddit = [t for t in reddit if len(t.split()) >= 8]
    wiki = [" ".join(w.split()[i:i + 40]) for w in sequences.windows()[:120] for i in range(0, 400, 40)]
    organism = [s for p in kl_point.read(SWEPT) if p.knob == 0.0 and p.path != "recorded"
                for s in p.samples + p.nucleus_samples]

    worst = max(organism, key=deg.top_token_frac)
    print(f"  HEADROOM CHECK — the un-ablated organism's own worst continuation:\n"
          f"    {worst!r}\n    top_token {deg.top_token_frac(worst):.3f}, "
          f"single_letter {deg.single_letter_frac(worst):.3f} — the incumbent's bars are 0.22 and 0.21, "
          f"so the gate {'FIRES' if incumbent(worst) else 'passes'} on the organism itself.\n")

    lab = np.array([i["label"] for i in strings])
    keep = np.isin(lab, ("working",) + BROKEN)
    y = np.isin(lab[keep], BROKEN)
    refs = {"wikitext 99th pct": _reference_metrics(model, wiki),
            "reddit-prompt 99th pct": _reference_metrics(model, reddit),
            "organism's own worst": _reference_metrics(model, organism)}
    header = "| " + f"{'metric':<14}" + " | fitted | err |" + "".join(
        f" {r:<22} | err |" for r in refs)
    print(header + "\n|" + "-" * (len(header) - 2) + "|")
    for name in names:
        s = np.array([i["m"][name] for i in strings])[keep]
        cut = youden(s[y], s[~y])
        row = f"| {name:<14} | {cut:6.3f} | {int(((s > cut) != y).sum()):3d} |"
        for label, ref in refs.items():
            vals = [m[name] for m in ref]
            t = max(vals) if label.startswith("organism") else float(np.quantile(vals, 0.99))
            row += f" {t:22.3f} | {int(((s > t) != y).sum()):3d} |"
        print(row)
    print(f"\n  (err is out of {int(keep.sum())} working+broken strings; the fitted column is the "
          "Youden optimum on these very labels, so it is the floor, not an estimate)")
    ref = float(np.mean([-m["gen_entropy"] for m in refs["organism's own worst"]]))
    s = np.array([-i["m"]["gen_entropy"] for i in strings])[keep]
    print(f"\n  MEMIT's own rule — 'complete generation failure' at fluency 622 → 294, i.e. below half "
          f"the unedited model.\n  Organism mean generation entropy {ref:.2f}; at < 0.5x ({ref / 2:.2f}) "
          f"that rule makes {int(((s < ref / 2) != y).sum())} errors of {int(keep.sum())}, "
          f"catching {(s[y] < ref / 2).mean():.2f} of broken for {(s[~y] < ref / 2).mean():.2f} of working.")


if __name__ == "__main__":
    main()
