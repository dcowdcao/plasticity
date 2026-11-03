"""ENTRYPOINT — collect every parameter sweep's raw points into flat CSVs under findings/sweeps/.

One file per selection method plus `every_sweep.csv`, all on one schema (point.Point), so the sweeps can
be sorted and compared directly.

    python -m sweep_tables.build_tables                  # the committed points, findings/sweeps/points/
    python -m sweep_tables.build_tables --run klattr2    # just one run id; other runs' rows survive
    python -m sweep_tables.build_tables --volume         # the Modal results volume instead

The committed points keep the volume's layout, `<prefix>/<run>/...`, so both sources read the same way.
Against the volume it can run while jobs are in flight: a sweep contributes whatever points it has
committed, and the count is printed so a partial table is not mistaken for a finished one.
"""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from helper_scripts import artifact_paths as paths
from sweep_tables import point as row
from sweep_tables.point import Point

# Volume prefix -> output file, named for what was measured; each run's id goes in the `run` column.
TABLES = {
    "attribution_sweep": "attribution_edges_only",
    "sheaf_sweep": "attribution_weights_and_edges",
    "lambda_grid": "optimiser_coefficient_grid",
}
PREFIXES = tuple(TABLES)
VOLUME = "plasticity-results"

# Runs whose points are kept but are not evidence, with the reason in the table itself. An empty
# `superseded` column means the run is current.
SUPERSEDED = {
    "lamgrid6": "edge-only grid from before the readout-cadence fix; 16 of 74 points landed",
}


class SweepError(Exception):
    pass


def die(msg: str) -> None:
    raise SweepError(msg)


def _attribution(run: str, rows: dict[str, dict]) -> list[Point]:
    """`attribution_sweep/<run>/k*.json`. Baseline comes from the run's own k=0 row — the rows carry
    no baseline field, and k=0 is the un-ablated model by construction."""
    by_k = {r["k"]: r for r in rows.values()}
    if 0 not in by_k:
        die(f"{run}: no k=0 row, so there is no baseline to judge `removed` against")
    base = by_k[0]["firing_rate"]
    return [Point(run=run, variant="edges", point=f"k={r['k']}", n_cut=r["k"], n_edges=r["k"],
                  n_weights=0, firing_rate=r["firing_rate"], baseline_fr=base,
                  kl_mis=r.get("kl_mis"), kl_ben=r["benign_kl"],
                  kl_ben_random=r.get("benign_kl_random"), preserves=r["preserves"],
                  repetition_frac=r.get("repetition_frac"), benign_ce=r.get("benign_ce"),
                  superseded=SUPERSEDED.get(run, ""))
            for r in by_k.values()]


def _lambda_grid(run: str, rows: dict[str, dict]) -> list[Point]:
    """`lambda_grid/<run>/<tag>.json`, in any of three row schemas:

      lam_r + lam_f   both coefficients swept, travel fixed (what run_lambda_grid_on_modal writes)
      coeff + mode    one coefficient against a pinned partner, travel swept
      ratio           no `mode` field; the λ_r:λ_f ratio is the point

    The joint grid persists its weight/edge split (`n_weights` + `n_edges` = `n_cut`); an edge-only
    grid has no `n_weights`, and its `n_edges` field, where present, is the graph's total edge count,
    not the cut — so the whole cut is edges.

    The `ratio` key in the third schema means λ_r:λ_f, not Point's `ratio` (kl_ben/kl_mis), which is
    one reason every derived column is a property on Point rather than a field read off the row."""
    out = []
    for r in rows.values():
        if "lam_r" in r:
            point, mode = f"lam_r={r['lam_r']:g},lam_f={r['lam_f']:g}", r["mode"]
        elif "coeff" in r:
            point, mode = f"coeff={r['coeff']:g},travel={r['travel']:g}", r["mode"]
        else:
            point, mode = f"lam_ratio={r['ratio']:g},travel={r['travel']:g}", "ratio"
        if "n_weights" in r:
            n_edges, n_weights = r["n_edges"], r["n_weights"]
        else:
            n_edges, n_weights = r["n_cut"], 0
        out.append(Point(run=run, variant=mode, point=point, n_cut=r["n_cut"],
                         n_edges=n_edges, n_weights=n_weights, firing_rate=r["firing_rate"],
                         baseline_fr=r["baseline_fr"], kl_mis=r.get("kl_mis"),
                         kl_ben=r["benign_kl"], kl_ben_random=r.get("benign_kl_random"),
                         preserves=r["preserves"], repetition_frac=r.get("repetition_frac"),
                         benign_ce=r.get("benign_ce"), superseded=SUPERSEDED.get(run, "")))
    return out


def _sheaf(run: str, rows: dict[str, dict]) -> list[Point]:
    """`sheaf_sweep/<run>/<variant>/k*.json`. `k` counts pool objects, so n_edges and n_weights are
    the split — which is the finding, not a formality: what fraction of a mixed cut is weights."""
    out = []
    for r in rows.values():
        out.append(Point(run=run, variant=r["variant"], point=f"k={r['k']}", n_cut=r["k"],
                         n_edges=r.get("n_edges"), n_weights=r.get("n_weights"),
                         firing_rate=r["firing_rate"], baseline_fr=_baseline_of(run, r),
                         kl_mis=r.get("kl_mis"), kl_ben=r["benign_kl"],
                         kl_ben_random=r.get("benign_kl_random"), preserves=r["preserves"],
                         repetition_frac=r.get("repetition_frac"), benign_ce=r.get("benign_ce"),
                         superseded=SUPERSEDED.get(run, "")))
    return out


def _baseline_of(run: str, row: dict) -> float:
    """`removed` is recomputed here, so a row without its baseline cannot be judged. The largest firing
    rate in a sweep is not a baseline — every point in it may be ablated — so fail rather than guess."""
    if not row.get("baseline_fr"):
        die(f"{run}: rows carry no baseline_fr, so `removed` is not recomputable "
            f"(measure_point must record it)")
    return row["baseline_fr"]


READERS = {"attribution_sweep": _attribution, "lambda_grid": _lambda_grid, "sheaf_sweep": _sheaf}


def collect(tree, prefix: str, run_filter: str = "") -> dict[str, list[Point]]:
    """All runs under one prefix. Keys are run ids; `.log` files and rankings are skipped."""
    runs: dict[str, dict[str, dict]] = {}
    for p in tree.paths():
        parts = Path(p).parts
        if len(parts) < 2 or parts[0] != prefix or not p.endswith(".json"):
            continue
        run = parts[1].removesuffix(".json")
        if run_filter and run != run_filter:
            continue
        if Path(p).name in ("pool_ranking.json", "sequential_ranking.json"):
            continue
        if len(parts) == 2:                 # <prefix>/<run>.json — an aggregate, not a point
            continue
        runs.setdefault(run, {})[p] = json.loads(tree.read(p))
    return {run: READERS[prefix](run, rows) for run, rows in runs.items() if rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume", action="store_true",
                   help=f"read the Modal volume {VOLUME!r} instead of the committed points")
    p.add_argument("--run", default="", help="refresh only this run id; other runs' rows survive")
    p.add_argument("--out", type=Path, default=paths.REPO / "findings" / "sweeps")
    a = p.parse_args()

    if a.volume:
        import modal   # optional extra; only this path talks to Modal
        tree = VolumeTree(modal.Volume.from_name(VOLUME))
    else:
        tree = LocalTree(paths.sweep_points_dir())

    a.out.mkdir(parents=True, exist_ok=True)
    every: list[Point] = []
    for prefix, name in TABLES.items():
        runs = collect(tree, prefix, a.run)
        pts = [p for rs in runs.values() for p in rs]
        if not pts:
            continue
        (a.out / f"{name}.csv").write_text(_refresh(a.out / f"{name}.csv", pts, a.run))
        every.extend(pts)
        print(f"{name}.csv  <- {', '.join(sorted(runs))}")
        print("  " + row.summary(pts).replace("\n", "\n  "))
    if not every:
        die("no sweep points found; check --run or whether anything has committed yet")
    (a.out / "every_sweep.csv").write_text(_refresh(a.out / "every_sweep.csv", every, a.run))
    print(f"\nevery_sweep.csv  <- all {len(every)} points across {len(TABLES)} methods")


def _refresh(path: Path, fresh: list[Point], run: str) -> str:
    """Write `fresh`, keeping rows from runs the filter did not touch: a filter narrows what is
    collected, never what is retained."""
    lines = row.to_csv(fresh).splitlines()
    header, data = lines[0], lines[1:]
    if run and path.exists():
        old = path.read_text().splitlines()
        if old and old[0] == header:                  # same schema, or the old rows are not mergeable
            keep = [l for r, l in zip(csv.DictReader(iter(old)), old[1:]) if r["run"] != run]
            data = keep + data
    def order(line: str) -> tuple:
        c = next(csv.reader([line]))
        return c[0], c[1], int(c[3] or -1), c[2]      # run, arm, n_ablated, setting
    return "\n".join([header] + sorted(set(data), key=order)) + "\n"


@dataclass(frozen=True)
class LocalTree:
    """A directory laid out like the results volume: `<root>/<prefix>/<run>/...`."""
    root: Path

    def paths(self) -> list[str]:
        if not self.root.is_dir():
            die(f"no sweep points at {self.root}")
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())

    def read(self, path: str) -> str:
        return (self.root / path).read_text()


@dataclass(frozen=True)
class VolumeTree:
    """The Modal results volume, over every sweep prefix."""
    volume: object   # modal.Volume

    def paths(self) -> list[str]:
        import modal
        out = []
        for prefix in PREFIXES:
            try:
                out += [e.path for e in self.volume.iterdir(prefix, recursive=True)]
            except modal.exception.NotFoundError:      # prefix absent until a job commits
                pass
        return sorted(out)

    def read(self, path: str) -> str:
        return b"".join(self.volume.read_file(path)).decode()


if __name__ == "__main__":
    try:
        main()
    except SweepError as e:
        raise SystemExit(f"error: {e}")
