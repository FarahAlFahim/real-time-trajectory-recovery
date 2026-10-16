"""Path resolution and JSON/JSONL helpers shared by the runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_path(p: str | None, *, base_dir: Path) -> Path | None:
    if p is None:
        return None
    s = str(p).strip()
    if not s:
        return None
    pp = Path(s).expanduser()
    return pp if pp.is_absolute() else (base_dir / pp).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must have a list root")
    return data
