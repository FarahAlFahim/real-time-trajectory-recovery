"""Timestamped console logging used by the runners."""

from __future__ import annotations

from datetime import datetime, timezone


def log_message(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}")
