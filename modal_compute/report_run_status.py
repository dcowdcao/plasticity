"""ENTRYPOINT (local) — one screen of status for every live Modal run.

Reads each app's whole log and reports the epoch distribution across containers (the latest lines
show only whichever container printed last), plus what has actually reached the results volume, so
progress and yield cannot be confused.

    python -m modal_compute.report_run_status              # live apps
    python -m modal_compute.report_run_status --all        # include stopped
"""
import argparse
import re
import subprocess
from collections import Counter
from dataclasses import dataclass

import modal

VOLUME = "plasticity-results"
# volume prefix -> expected result count for a full run of that kind; keyed by prefix, not run-name
# family, so two families sharing a prefix are not double-counted
EXPECTED = {"lambda_grid": 74, "sheaf_sweep": 91, "attribution_sweep": 29}
EPOCH = re.compile(r"\[(weight|edge)\] epoch (\d+)/(\d+)")
# Last line of a traceback is the one that names the failure. Modal prefixes container frames, so the
# exception line is the one with no leading "File "/" and a "SomeError:" shape.
FAILURE = re.compile(r"^(?:\[t\+\s*[\d.]+s\]\s*)?([A-Za-z_.]*(?:Error|Exception|Interrupt)[^\n]*)$",
                     re.MULTILINE)
FEASIBLE = re.compile(r"\[(weight|edge)\] feasible at epoch (\d+) batch (\d+): cut=(\d+)")


@dataclass(frozen=True)
class App:
    app_id: str
    state: str
    tasks: int


def apps(include_stopped: bool) -> list[App]:
    out = subprocess.run(["modal", "app", "list"], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        if "plasticity" not in line:
            continue
        cells = [c.strip() for c in line.split("│") if c.strip()]
        if len(cells) < 4 or not cells[0].startswith("ap-"):
            continue
        app = App(cells[0], cells[2], int(cells[3]) if cells[3].isdigit() else 0)
        if include_stopped or app.state != "stopped":
            found.append(app)
    return found


def progress(app_id: str) -> str:
    """Epoch distribution across every container, plus any feasibility hits."""
    log = subprocess.run(["modal", "app", "logs", app_id], capture_output=True, text=True).stdout
    epochs = Counter((m.group(1), int(m.group(2)), int(m.group(3))) for m in EPOCH.finditer(log))
    hits = [(m.group(1), int(m.group(4))) for m in FEASIBLE.finditer(log)]
    if not epochs and not hits and not FAILURE.search(log):
        return "      no phase output yet"
    lines = []
    for phase in ("weight", "edge"):
        seen = {(e, tot): n for (p, e, tot), n in epochs.items() if p == phase}
        if not seen:
            continue
        total = next(tot for _, tot in seen)
        ladder = "  ".join(f"{e}:{n}" for (e, _), n in sorted(seen.items()))
        lead = max(e for e, _ in seen)
        lines.append(f"      {phase:<6} lead {lead}/{total}   points per epoch  {ladder}")
    if hits:
        cuts = sorted(c for _, c in hits)
        lines.append(f"      FEASIBLE {len(hits)}x — cuts {cuts[0]}–{cuts[-1]} edges")
    fails = {m.group(1).strip()[:110] for m in FAILURE.finditer(log)}
    for f in sorted(fails):
        lines.append(f"      !! {f}")
    return "\n".join(lines)


def yields(vol) -> dict[str, tuple[int, int, int]]:
    """run_id -> (results, logs, expected) for every run on the volume."""
    out = {}
    for prefix, expected in EXPECTED.items():
        try:
            entries = list(vol.iterdir(prefix, recursive=True))
        except modal.exception.NotFoundError:
            continue
        for e in entries:
            parts = e.path.split("/")
            if len(parts) < 3:
                continue
            run = parts[1].removesuffix(".json")
            r, l, _ = out.get(run, (0, 0, expected))
            out[run] = (r + e.path.endswith((".json", ".pt")), l + e.path.endswith(".log"), expected)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all", action="store_true", help="include stopped apps")
    a = p.parse_args()

    live = apps(a.all)
    print(f"=== {len(live)} plasticity app(s) ===")
    for app in live:
        print(f"  {app.app_id}  {app.state:<10} {app.tasks:>3} tasks")
        print(progress(app.app_id))

    print("\n=== yield on the volume (results / expected) ===")
    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    for run, (res, logs, exp) in sorted(yields(vol).items(), key=lambda kv: -kv[1][0]):
        bar = f"{res}/{exp}" if exp else str(res)
        flag = "  <- nothing landed" if res == 0 and logs else ""
        print(f"  {run:<20} {bar:>8} results   {logs:>3} logs{flag}")


if __name__ == "__main__":
    main()
