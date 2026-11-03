"""GPU-free check that the per-cycle circuit-characterisation stage hangs together: every
sufficient_circuit.json is self-consistent (circuit_records), faithfulness rises with the
attribution prefix, consecutive cycles are drawn from one candidate space (not the
`candidate_edges - ablated` signature that forces drift to 0), and the loop's writer agrees with
circuit_evolution's reader on field names.

Always self-tests against fixtures it generates and deletes; --runs-dir adds a real runs/ subtree.
Stage components absent from the tree report `skip`. The wiring checks read source with `ast`
rather than importing it, so nothing here pulls torch or a GPU.

Usage: python -m characterization_checks.check_characterization [--runs-dir runs/toxicity/gpt2]"""
import argparse, ast, difflib, json, shutil, sys, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from characterization_checks import circuit_records as records
from characterization_checks.circuit_records import CharacterizationError, DEFAULT_TAU, rel

FAITH_DROP_TOL = 0.05        # unexplained adjacent-k_ranked dip worth surfacing; never a failure
SUFFICIENT_CIRCUIT_FIELDS = ("edges", "k", "faithfulness", "curve", "metric_full",
                             "metric_empty", "sufficient")

RUN_LOOP = records.REPO / "iterative_pruning" / "run_loop.py"
ARTIFACT_PATHS = records.REPO / "helper_scripts" / "artifact_paths.py"
SUFFICIENT_CIRCUIT = records.REPO / "circuit_discovery" / "sufficient_circuit.py"
CIRCUIT_DRIFT = records.REPO / "circuit_evolution" / "circuit_drift.py"
MEASURE_EVOLUTION = records.REPO / "circuit_evolution" / "measure_circuit_evolution.py"

Status = Literal["pass", "fail", "warn", "skip"]
Outcome = tuple[Status, str]


def die(msg: str):
    raise CharacterizationError(msg)


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


# --- what the records say ---

def record_invariants(found: list[Path], ok: list[records.Record], bad: list[str]) -> Outcome:
    if not found:
        return "skip", (f"no {records.RECORD_FILENAME} in this tree — the stage has not written "
                        "any yet")
    if bad:
        raise CharacterizationError(f"{len(bad)}/{len(found)} records rejected:\n" +
                                    "\n".join(bad))
    detail = [f"{len(found)} record(s): schema round-trip + invariants hold",
              "faithfulness from cycle 1 on is measured against the ablated model, so it is only "
              "comparable at equal n_ablated:"]
    for g, seq in records.groups(ok).items():
        detail.append(_g(g))
        detail += [f"  cycle {r.cycle:<3} {r.phase:<14} k {r.k}/{r.k_ranked:<5} "
                   f"faithfulness {r.faithfulness:.3f}  sufficient {str(r.sufficient):<5}  "
                   f"n_ablated {r.n_ablated}" for r in seq]
    extra = sorted({f for r in ok for f in r.extra})
    if extra:
        return "warn", "\n".join(detail + [
            f"records carry fields outside the documented schema: {extra} — extend RECORD_FIELDS "
            "if that is deliberate"])
    return "pass", "\n".join(detail)


def curve_monotonicity(found: list[records.Record]) -> Outcome:
    """Faithfulness should broadly rise with k_ranked. A dip that co-occurs with a drop in
    post-prune k is expected (the larger prefix collapsed to a smaller connected circuit);
    a dip while k holds or grows is unexplained. Neither fails — non-monotonicity is a real
    empirical possibility, so this reports."""
    if not found:
        return "skip", "no records to check"
    dips = records.dips(found)
    explained = [d for d in dips if d.dk < 0]
    unexplained = [d for d in dips if d.dk >= 0]
    flat = [r for r in found if not r.curve_rises]
    worst = max(unexplained, default=None, key=lambda d: d.drop)
    detail = (f"{len(dips)} dip(s): {len(explained)} alongside a k drop (pruning collapsed the "
              f"prefix — expected), {len(unexplained)} with k held or rising")
    if explained:
        detail += f"\nlargest pruning-explained dip: {max(explained, key=lambda d: d.drop)}"
    if worst and worst.drop > FAITH_DROP_TOL:
        return "warn", (f"{detail}\nlargest unexplained dip {worst.drop:.3f} > tolerance "
                        f"{FAITH_DROP_TOL}: {worst}"
                        "\nreported, not failed: a real curve can dip; read the two columns "
                        "together before suspecting the code")
    if flat:
        return "warn", (f"{detail}\n{len(flat)} curve(s) never rise from the first k_ranked to the "
                        f"last, e.g. {rel(flat[0].path)} — a circuit that gains nothing from more "
                        "edges is a finding or a broken metric, not a schema defect")
    return "pass", detail + (f"\nlargest unexplained dip {worst.drop:.3f} <= {FAITH_DROP_TOL}"
                             if worst else "\nno unexplained dip")


def cycle_comparability(found: list[records.Record]) -> Outcome:
    """Characterisation sees the full graph every cycle, so consecutive circuits can overlap.
    All-zero IoU with every cycle disjoint from all earlier ones is the `candidate_edges -
    ablated` signature: the search was constrained to exclude what it already cut, and drift
    is then 0 by construction rather than by measurement."""
    if not found:
        return "skip", "no records to compare"
    lines, forced, thin = [], [], []
    for g, seq in records.groups(found).items():
        cycles = [r.cycle for r in seq]
        if len(set(cycles)) != len(cycles):
            raise CharacterizationError(f"{_g(g)}: duplicate cycle numbers {cycles} — two records "
                                        "claim the same cycle")
        if len(seq) < 2:
            thin.append(f"{_g(g)}: {len(seq)} cycle, nothing to compare")
            continue
        ious = records.ious(seq)
        lines.append(f"{_g(g)}: cycles {cycles}, |edges| {[r.k for r in seq]}, consecutive IoU "
                     f"{[round(i, 3) for i in ious]}")
        if records.forced_disjoint(seq):
            (forced if len(ious) >= 2 else thin).append(
                f"{_g(g)}: every consecutive IoU is exactly 0.0 over {len(ious)} transition(s) and "
                "every cycle's edges are disjoint from all earlier cycles'")
    if forced:
        raise CharacterizationError(
            "cycles are not drawn from one candidate space:\n" + "\n".join(forced) +
            "\nthat is the `candidate_edges - ablated` signature, which forces drift "
            "to 0. Characterisation must search the full graph every "
            "cycle, otherwise drift is 0 by construction and measures nothing.")
    if thin:
        return "warn", "\n".join(lines + thin + [
            "too few transitions to tell a forced 0.0 from a real relocation; needs 3+ cycles"])
    return "pass", "\n".join(lines)


def ablation_growth(found: list[records.Record]) -> Outcome:
    """The loop's edge mask is cumulative and persistent, so n_ablated only grows along an arm."""
    if not found:
        return "skip", "no records to compare"
    lines, shrunk = [], []
    for g, seq in records.groups(found).items():
        lines.append(f"{_g(g)}: n_ablated {[r.n_ablated for r in seq]} over cycles "
                     f"{[r.cycle for r in seq]}")
        shrunk += [f"{_g(g)}: n_ablated fell from {a.n_ablated} at cycle {a.cycle} to "
                   f"{b.n_ablated} at cycle {b.cycle}" for a, b in records.ablation_falls(seq)]
    if shrunk:
        raise CharacterizationError(
            "the cumulative ablation shrank:\n" + "\n".join(shrunk) +
            "\nthe mask is persistent across cycles, so a fall means the record was written against "
            "a different mask, or two runs' cycles landed in one tree")
    return "pass", "\n".join(lines)


def circuit_excludes_ablated(found: list[records.Record]) -> Outcome:
    """Opportunistic: where the loop wrote its cumulative cut beside the record, the circuit must
    share no edge with it. An overlap means the already-ablated edges were not held out of the
    graph, so the circuit is not a circuit of the model it was measured on."""
    if not found:
        return "skip", "no records to cross-check"
    checked, overlap, unpaired = [], [], []
    for rec in found:
        pair = records.cut_edges(rec.path.parent)
        if pair is None:
            unpaired.append(rec)
            continue
        cut_path, cut = pair
        checked.append(f"{rel(rec.path.parent)}: {rec.k} circuit edges vs {len(cut)} in "
                       f"{cut_path.name}, record says n_ablated {rec.n_ablated}")
        shared = sorted(rec.edge_keys & cut)
        if shared:
            overlap.append(f"{rel(rec.path)}: {len(shared)} circuit edge(s) are already ablated "
                           f"per {cut_path.name}, e.g. {shared[0]}")
    if overlap:
        raise CharacterizationError(
            "the circuit contains already-ablated edges:\n" + "\n".join(overlap) +
            "\nthe characteriser is not holding the cumulative mask out of the graph")
    if not checked:
        return "skip", (f"not available: no {' or '.join(records.CUT_FILES)} beside the "
                        f"{len(unpaired)} record(s), so the ablated set cannot be reconstructed")
    return "pass", "\n".join(checked + ([f"{len(unpaired)} record(s) had no cut file beside them"]
                                        if unpaired else []))


def _g(group: tuple[str, str, str]) -> str:
    return "/".join(group)


# --- integration wiring, read out of the source with ast ---

def loop_config(tau: float) -> Outcome:
    gate = _stage_absent()
    if gate:
        return gate
    mod = _module(RUN_LOOP)
    fields = _class_fields(mod, RUN_LOOP, "LoopConfig")
    absent = [f for f in ("characterize", "faith_tau") if f not in fields]
    if absent:
        raise CharacterizationError(f"{rel(RUN_LOOP)}: LoopConfig is missing {absent}")
    default = _class_field_default(mod, RUN_LOOP, "LoopConfig", "faith_tau")
    if default != tau:
        return "warn", (f"LoopConfig.faith_tau default is {default!r} but records were judged "
                        f"against --tau {tau}; pass --tau {default!r} to check them as written")
    return "pass", f"LoopConfig.characterize + faith_tau present, faith_tau default {default!r}"


def path_helpers() -> Outcome:
    gate = _stage_absent()
    if gate:
        return gate
    mod = _module(ARTIFACT_PATHS)
    defined = _func_names(mod)
    absent = [n for n in ("sufficient_circuit_path", "circuit_evolution_path") if n not in defined]
    if absent:
        raise CharacterizationError(f"{rel(ARTIFACT_PATHS)}: missing path helper(s) {absent}; "
                                    "artifact_paths owns every artifact path")
    named = _json_constants(mod, "sufficient_circuit_path")
    if named and records.RECORD_FILENAME not in named:
        raise CharacterizationError(f"{rel(ARTIFACT_PATHS)}: sufficient_circuit_path names "
                                    f"{named}, but this checker globs {records.RECORD_FILENAME!r}")
    return "pass", (f"sufficient_circuit_path -> {named or 'name not a literal'}, "
                    "circuit_evolution_path defined")


def writer_uses_path_helper() -> Outcome:
    gate = _stage_absent()
    if gate:
        return gate
    mod = _module(RUN_LOOP)
    if "sufficient_circuit_path" not in _attrs(mod):
        raise CharacterizationError(f"{rel(RUN_LOOP)}: never calls paths.sufficient_circuit_path, "
                                    "so the per-cycle record is written somewhere else or not at all")
    keys = _record_dict(mod)
    absent = [f for f in records.RECORD_FIELDS if f not in keys]
    if absent:
        raise CharacterizationError(f"{rel(RUN_LOOP)}: the record it writes omits {absent} "
                                    f"(writes {sorted(keys)})")
    return ("pass",
            f"writes all {len(records.RECORD_FIELDS)} schema fields via paths.sufficient_circuit_path")


def finder_api() -> Outcome:
    gate = _stage_absent()
    if gate:
        return gate
    mod = _module(SUFFICIENT_CIRCUIT)
    if "find_sufficient_circuit" not in _func_names(mod):
        raise CharacterizationError(f"{rel(SUFFICIENT_CIRCUIT)}: find_sufficient_circuit not defined")
    fields = _class_fields(mod, SUFFICIENT_CIRCUIT, "SufficientCircuit")
    absent = [f for f in SUFFICIENT_CIRCUIT_FIELDS if f not in fields]
    if absent:
        raise CharacterizationError(f"{rel(SUFFICIENT_CIRCUIT)}: SufficientCircuit is missing "
                                    f"{absent}")
    return "pass", f"find_sufficient_circuit + SufficientCircuit({', '.join(fields)})"


def writer_reader_field_names() -> Outcome:
    """circuit_evolution must not read a key run_loop never writes."""
    gate = _stage_absent()
    if gate:
        return gate
    readers = [p for p in (CIRCUIT_DRIFT, MEASURE_EVOLUTION) if p.exists()]
    if not readers:
        return "skip", ("reader not present: circuit_evolution/ is absent from this tree, so there "
                        "is nothing to compare")
    writer = _module(RUN_LOOP)
    if "sufficient_circuit_path" not in _attrs(writer):
        return "skip", f"{rel(RUN_LOOP)} does not write the record yet, nothing to compare against"
    written = _record_dict(writer)
    reads, emits = set(), set()
    for p in readers:
        mod = _module(p)
        reads |= _read_keys(mod)
        for keys in _dict_keys(mod):
            emits |= keys
    suspect = sorted(reads - written - set(records.CURVE_FIELDS))
    unwritten = [s for s in suspect if s not in emits]
    if unwritten:
        near = {s: _nearest(s, sorted(written)) for s in unwritten}
        named = ", ".join(f"{s!r}" + (f" (meant {near[s]!r}?)" if near[s] else "")
                          for s in unwritten)
        raise CharacterizationError(
            f"reader/writer field mismatch: {', '.join(rel(p) for p in readers)} read field(s) "
            f"{rel(RUN_LOOP)} never writes: {named}")
    if suspect:
        return "warn", (f"reader reads {suspect}, which {rel(RUN_LOOP)}'s record dict does not "
                        "write. The reader also writes those names itself, so static reading cannot "
                        "tell whose dict they come from — check the record fields by hand")
    unread = sorted(written - reads - set(records.CURVE_FIELDS))
    return "pass", (f"reader keys {sorted(reads & written)} all written by {rel(RUN_LOOP)}" +
                    (f"; written but unread: {unread}" if unread else ""))


def _nearest(name: str, written: list[str]) -> str | None:
    """Containment first, then difflib: 'faith' must suggest 'faithfulness', not 'trait'."""
    inside = [w for w in written if name in w or w in name]
    if inside:
        return min(inside, key=len)
    near = difflib.get_close_matches(name, written, n=1)
    return near[0] if near else None


def _stage_absent() -> Outcome | None:
    if SUFFICIENT_CIRCUIT.exists():
        return None
    return "skip", (f"characterisation stage not in this tree ({rel(SUFFICIENT_CIRCUIT)} absent) — "
                    "this check needs it")


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _class_def(mod: ast.Module, path: Path, name: str) -> ast.ClassDef:
    cls = next((n for n in ast.walk(mod) if isinstance(n, ast.ClassDef) and n.name == name), None)
    if cls is None:
        raise CharacterizationError(f"{rel(path)}: class {name} not defined")
    return cls


def _class_fields(mod: ast.Module, path: Path, name: str) -> list[str]:
    return [n.target.id for n in _class_def(mod, path, name).body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]


def _class_field_default(mod: ast.Module, path: Path, name: str, field_name: str) -> object:
    for n in _class_def(mod, path, name).body:
        if (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                and n.target.id == field_name and isinstance(n.value, ast.Constant)):
            return n.value.value
    return None


def _func_names(mod: ast.Module) -> set[str]:
    return {n.name for n in ast.walk(mod) if isinstance(n, ast.FunctionDef)}


def _attrs(mod: ast.Module) -> set[str]:
    return {n.attr for n in ast.walk(mod) if isinstance(n, ast.Attribute)}


def _json_constants(mod: ast.Module, func: str) -> list[str]:
    fn = next(n for n in ast.walk(mod) if isinstance(n, ast.FunctionDef) and n.name == func)
    return [n.value for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".json")]


def _dict_keys(mod: ast.Module) -> list[set[str]]:
    return [{k.value for k in d.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            for d in ast.walk(mod) if isinstance(d, ast.Dict)]


def _record_dict(mod: ast.Module) -> set[str]:
    """The module's dict literal whose string keys look most like the record schema."""
    return max(_dict_keys(mod), key=lambda keys: len(keys & set(records.RECORD_FIELDS)),
               default=set())


def _read_keys(mod: ast.Module) -> set[str]:
    """String keys the module reads out of dicts: x["f"] and x.get("f")."""
    keys = set()
    for n in ast.walk(mod):
        if (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, str)):
            keys.add(n.slice.value)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
                and n.args and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            keys.add(n.args[0].value)
    return keys


# --- synthetic fixtures + the self-test that proves the checks bite ---

EDGE_POOL = (("a1.h4", "input", "q"), ("a3.h9", "a1.h4", "v"), ("m7", "a3.h9", None),
             ("a8.h1", "m7", "k"), ("m10", "a8.h1", None), ("logits", "m10", None),
             ("a5.h3", "input", "v"), ("logits", "a5.h3", None))

RELOCATED = (("a2.h0", "input", "k"), ("m3", "a2.h0", None), ("a9.h5", "m3", "q"),
             ("logits", "a9.h5", None))

# (id, mutation of a valid record, the token the rejection must name)
MALFORMED = (
    ("k_vs_len_edges", lambda d: {**d, "k": d["k"] + 1}, "len(edges)"),
    ("k_above_k_ranked", lambda d: {**d, "k_ranked": d["k"] - 1}, "exceeds k_ranked"),
    ("edge_arity", lambda d: {**d, "edges": [*d["edges"][:-1], ["logits", "m10"]]}, "3-element"),
    ("edge_qkv", lambda d: {**d, "edges": [["a1.h4", "input", "z"], *d["edges"][1:]]}, "qkv"),
    ("edges_duplicated", lambda d: {**d, "edges": [*d["edges"][:-1], d["edges"][0]]}, "duplicates"),
    ("curve_empty", lambda d: {**d, "curve": []}, "curve is empty"),
    ("curve_k_ranked_not_increasing", lambda d: {**d, "curve": list(reversed(d["curve"]))},
     "strictly increasing"),
    ("curve_k_above_k_ranked",
     lambda d: {**d, "curve": [{**d["curve"][0], "k": d["curve"][0]["k_ranked"] + 1},
                               *d["curve"][1:]]}, "exceeds k_ranked"),
    ("curve_missing_chosen_cursor", lambda d: {**d, "curve": d["curve"][:-1]}, "not in the curve"),
    ("chosen_faithfulness_disagrees", lambda d: {**d, "faithfulness": 0.5, "sufficient": False},
     "disagrees with its own curve entry"),
    ("chosen_k_disagrees",
     lambda d: {**d, "curve": [*d["curve"][:-1], {**d["curve"][-1], "k": d["k"] - 2}]},
     "disagrees with its own curve entry"),
    ("sufficient_vs_tau", lambda d: {**d, "sufficient": not d["sufficient"]}, "sufficient"),
    ("metric_full_eq_empty", lambda d: {**d, "metric_empty": d["metric_full"]}, "metric_empty"),
    ("faithfulness_not_finite", lambda d: {**d, "faithfulness": float("nan")}, "not finite"),
    ("n_ablated_negative", lambda d: {**d, "n_ablated": -1}, "n_ablated"),
    ("n_ablated_at_baseline", lambda d: {**d, "n_ablated": 7}, "n_ablated"),
    ("field_missing", lambda d: {k: v for k, v in d.items() if k != "phase"}, "phase"),
)


def fixture(cycle: int, edges: tuple[records.EdgeKey, ...], faiths=(0.31, 0.62, 0.86), ks=None,
            n_ablated=None) -> dict:
    """A valid record. The chosen point is the last curve entry, so it agrees by construction."""
    cursors = (8, 24, 64)
    ks = (2, 4, len(edges)) if ks is None else ks
    curve = [{"k_ranked": r, "k": k, "faithfulness": f, "metric": round(0.12 + 0.79 * f, 4)}
             for r, k, f in zip(cursors, ks, faiths)]
    chosen = curve[-1]
    return {"trait": "toxicity/gpt2", "run_id": "synthetic", "arm": "finetune", "cycle": cycle,
            "phase": records.BASELINE_PHASE if cycle == 0 else "post_retrain",
            "k": chosen["k"], "k_ranked": chosen["k_ranked"],
            "n_ablated": 41 * cycle if n_ablated is None else n_ablated,
            "faithfulness": chosen["faithfulness"],
            "sufficient": chosen["faithfulness"] >= DEFAULT_TAU,
            "metric_full": 0.91, "metric_empty": 0.12, "curve": curve,
            "edges": [list(e) for e in edges]}


def write_fixtures(root: Path) -> None:
    for i in range(3):
        _write(root / "valid" / f"cycle_{i:03d}", fixture(i, EDGE_POOL[i:i + 5]))
    for i, edges in enumerate((EDGE_POOL[:4], EDGE_POOL[4:8], RELOCATED)):
        _write(root / "relocated" / f"cycle_{i:03d}", fixture(i, edges))
    for i, n in enumerate((0, 96, 41)):
        _write(root / "ablation_decrease" / f"cycle_{i:03d}",
               fixture(i, EDGE_POOL[i:i + 5], n_ablated=n))
    # k 3 -> 6 -> 5: pruning collapsed the wider prefix, and faithfulness dipped with it.
    _write(root / "pruned_dip" / "cycle_000",
           fixture(0, EDGE_POOL[:5], faiths=(0.31, 0.72, 0.61), ks=(3, 6, 5)))
    _write(root / "unexplained_dip" / "cycle_000",
           fixture(0, EDGE_POOL[:5], faiths=(0.31, 0.20, 0.86)))
    over = root / "ablated_overlap" / "cycle_001"
    _write(over, fixture(1, EDGE_POOL[:5]))
    (over / "ablation.json").write_text(json.dumps(          # dict-shaped edges
        {"cycle": 1, "edges": [{"child": c, "parent": p, "qkv": q}
                               for c, p, q in (EDGE_POOL[2], *RELOCATED)]}, indent=1) + "\n")
    clean = root / "ablated_disjoint" / "cycle_001"
    _write(clean, fixture(1, EDGE_POOL[:5]))
    (clean / "cut.json").write_text(json.dumps([list(e) for e in RELOCATED], indent=1) + "\n")
    base = fixture(0, EDGE_POOL[:5])
    for name, mutate, _ in MALFORMED:
        _write(root / "malformed" / name, mutate(base))


def _write(cdir: Path, record: dict) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / records.RECORD_FILENAME).write_text(json.dumps(record, indent=1) + "\n")


def valid_accepted(root: Path) -> Outcome:
    found = records.scan(root / "valid")
    ok, bad = records.load_all(found, DEFAULT_TAU)
    if bad:
        raise CharacterizationError("valid fixtures rejected — the checks or the fixtures are "
                                    "wrong:\n" + "\n".join(bad))
    return "pass", (f"{len(ok)} synthetic records accepted (round-trip + invariants). Fixtures are "
                    f"written at tau={DEFAULT_TAU} and read back at it, so --tau applies only to "
                    "records on disk.")


def malformed_rejected(root: Path) -> Outcome:
    slipped, vague = [], []
    for name, _, token in MALFORMED:
        try:
            records.read(root / "malformed" / name / records.RECORD_FILENAME, DEFAULT_TAU)
        except CharacterizationError as e:
            if token not in str(e):
                vague.append(f"{name}: rejected, but the message never names {token!r}: {e}")
            continue
        slipped.append(f"{name}: accepted, the checks are blind to it")
    if slipped or vague:
        raise CharacterizationError("malformed fixtures not caught precisely:\n" +
                                    "\n".join(slipped + vague))
    return "pass", (f"all {len(MALFORMED)} malformed records rejected, each naming the offending "
                    "field:\n" + "\n".join(f"{name} -> {token!r}" for name, _, token in MALFORMED))


def pruned_k_accepted(root: Path) -> Outcome:
    ok, bad = records.load_all(records.scan(root / "pruned_dip"), DEFAULT_TAU)
    if bad:
        raise CharacterizationError("a record with non-monotonic post-prune k was rejected; k is "
                                    "only bounded by k_ranked, not monotonic:\n" + "\n".join(bad))
    status, detail = curve_monotonicity(ok)
    if status != "pass":
        raise CharacterizationError(f"a dip alongside a k drop should read as expected, got "
                                    f"{status!r}: {detail}")
    return "pass", "non-monotonic k accepted, its dip classified as pruning-explained:\n" + detail


def unexplained_dip_warns(root: Path) -> Outcome:
    ok, bad = records.load_all(records.scan(root / "unexplained_dip"), DEFAULT_TAU)
    if bad:
        raise CharacterizationError("a non-monotonic faithfulness curve must still be a valid "
                                    "record:\n" + "\n".join(bad))
    status, detail = curve_monotonicity(ok)
    if status != "warn":
        raise CharacterizationError(f"a dip with k rising should warn, got {status!r}: {detail}")
    return "pass", "dip with k rising surfaced as a warning, not a failure:\n" + detail


def ablation_decrease_flagged(root: Path) -> Outcome:
    ok, bad = records.load_all(records.scan(root / "ablation_decrease"), DEFAULT_TAU)
    if bad:
        raise CharacterizationError("the decreasing-n_ablated fixture is not readable:\n" +
                                    "\n".join(bad))
    try:
        ablation_growth(ok)
    except CharacterizationError as e:
        return "pass", "n_ablated 0 -> 96 -> 41 flagged:\n" + str(e)
    raise CharacterizationError("a shrinking cumulative ablation was not flagged")


def ablated_overlap_flagged(root: Path) -> Outcome:
    over, bad = records.load_all(records.scan(root / "ablated_overlap"), DEFAULT_TAU)
    clean, unreadable = records.load_all(records.scan(root / "ablated_disjoint"), DEFAULT_TAU)
    if bad + unreadable:
        raise CharacterizationError("the cut cross-check fixtures are not readable:\n" +
                                    "\n".join(bad + unreadable))
    status, detail = circuit_excludes_ablated(clean)
    if status != "pass":
        raise CharacterizationError(f"a circuit disjoint from cut.json should pass, got "
                                    f"{status!r}: {detail}")
    try:
        circuit_excludes_ablated(over)
    except CharacterizationError as e:
        return "pass", ("disjoint case passed, overlapping case flagged:\n" + detail + "\n" + str(e))
    raise CharacterizationError("a circuit holding an already-ablated edge was not flagged")


def relocation_flagged(root: Path) -> Outcome:
    ok, bad = records.load_all(records.scan(root / "relocated"), DEFAULT_TAU)
    if bad:
        raise CharacterizationError("the disjoint-cycle fixture is not readable:\n" + "\n".join(bad))
    try:
        cycle_comparability(ok)
    except CharacterizationError as e:
        return "pass", "3 cycles with pairwise-disjoint edge sets flagged:\n" + str(e)
    raise CharacterizationError("the disjoint 3-cycle fixture was not flagged — the comparability "
                                "check is blind to the `candidate_edges - ablated` signature")


# --- report ---

SECTIONS = {"fixtures": "self-test on generated fixtures (always runs)",
            "runs": "records on disk",
            "wiring": "integration with the loop (static, nothing imported)"}
STATUSES = ("fail", "warn", "skip", "pass")


def checks(runs_dir: Path | None, tau: float) -> list[Check]:
    root = Path(tempfile.mkdtemp(prefix="characterization_fixtures_"))
    try:
        write_fixtures(root)
        out = [_run("fixtures/valid_accepted", lambda: valid_accepted(root)),
               _run("fixtures/malformed_rejected", lambda: malformed_rejected(root)),
               _run("fixtures/pruned_k_accepted", lambda: pruned_k_accepted(root)),
               _run("fixtures/unexplained_dip_warns", lambda: unexplained_dip_warns(root)),
               _run("fixtures/relocation_flagged", lambda: relocation_flagged(root)),
               _run("fixtures/ablation_decrease_flagged",
                    lambda: ablation_decrease_flagged(root)),
               _run("fixtures/ablated_overlap_flagged",
                    lambda: ablated_overlap_flagged(root))]
    finally:
        shutil.rmtree(root)

    if runs_dir is None:
        synthetic = ("no --runs-dir: synthetic-only mode. Point it at runs/<family>/<name> once the "
                     "stage has run.")
        out += [Check(f"runs/{n}", "skip", synthetic)
                for n in ("record_invariants", "curve_monotonicity", "cycle_comparability",
                          "ablation_growth", "circuit_excludes_ablated")]
    else:
        found = records.scan(runs_dir)
        ok, bad = records.load_all(found, tau)
        out += [_run("runs/record_invariants", lambda: record_invariants(found, ok, bad)),
                _run("runs/curve_monotonicity", lambda: curve_monotonicity(ok)),
                _run("runs/cycle_comparability", lambda: cycle_comparability(ok)),
                _run("runs/ablation_growth", lambda: ablation_growth(ok)),
                _run("runs/circuit_excludes_ablated", lambda: circuit_excludes_ablated(ok))]

    return out + [_run("wiring/loop_config", lambda: loop_config(tau)),
                  _run("wiring/path_helpers", path_helpers),
                  _run("wiring/writer_uses_path_helper", writer_uses_path_helper),
                  _run("wiring/finder_api", finder_api),
                  _run("wiring/writer_reader_field_names", writer_reader_field_names)]


def _run(name: str, check: Callable[[], Outcome]) -> Check:
    try:
        status, detail = check()
    except CharacterizationError as e:
        return Check(name, "fail", str(e))
    return Check(name, status, detail)


def report(found: list[Check]) -> int:
    print(f"circuit characterisation checks — {records.REPO}")
    print("pass = holds | fail = defect | warn = read the data, not the code | "
          "skip = component not in this tree")
    section = None
    for c in found:
        prefix = c.name.split("/")[0]
        if prefix != section:
            section = prefix
            print(f"\n[{SECTIONS[prefix]}]")
        head, *rest = c.detail.splitlines()
        print(f"  {c.status.upper():4} {c.name:36} {head}")
        for line in rest:
            print(f"       {'':36} {line}")
    counts = {s: sum(1 for c in found if c.status == s) for s in STATUSES}
    print("\n" + ", ".join(f"{n} {s}" for s, n in counts.items()))
    return 1 if counts["fail"] else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                epilog="exit 0 = clear, 1 = a check failed, 2 = bad invocation")
    p.add_argument("--runs-dir", type=Path, default=None,
                   help=f"tree to search for {records.RECORD_FILENAME}; omitted = synthetic "
                        "fixtures only")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU,
                   help="faithfulness bar the records were judged against (LoopConfig.faith_tau)")
    a = p.parse_args()
    try:
        if a.runs_dir is not None and not a.runs_dir.is_dir():
            die(f"--runs-dir {a.runs_dir} is not a directory")
        code = report(checks(a.runs_dir, a.tau))
    except CharacterizationError as e:
        print(f"error: {e}", file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
