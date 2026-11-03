"""Flat-text JSONL IO — the repo's universal record interface."""
import json
from pathlib import Path
from typing import Any, Iterable


def read(path: Path) -> list[dict[str, Any]]:
    text = Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
