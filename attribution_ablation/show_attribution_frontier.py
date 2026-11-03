"""ENTRYPOINT — print a finished attribution sweep: the coupling, the frontier, the continuations.

`modal app logs` is not reliably retrievable (it can fail with "Logs query hit resource limit"), so
this prints the record on the results volume exactly as the run printed it, continuations included:
they are what tell a collapsed model from a clean removal.

  modal volume get plasticity-results attribution_sweep/attr1.json attr1.json
  python -m attribution_ablation.show_attribution_frontier attr1.json
"""
import argparse
import json
from pathlib import Path

from attribution_ablation.measure_attribution_frontier import print_coupling, print_frontier


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("record", type=Path, help="an attribution_sweep.json")
    a = p.parse_args()

    record = json.loads(a.record.read_text())
    print_coupling(record)
    if "rows" not in record:
        print("ranking only — the k sweep has not finished (or was stopped)")
        return
    print_frontier(record["rows"], record["remove_bar"], record["preserve_budget"])


if __name__ == "__main__":
    main()
