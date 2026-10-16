"""Synchronous Codex hook bridge for the host-side trajectory controller.

This file is copied into each active Codex task container. Codex invokes it
for PreToolUse, PostToolUse, and Stop. The bridge contains no recovery policy;
it only exchanges one JSON request/response through a bind-mounted directory.
Keeping the reviewer on the host lets Codex run in an unmodified benchmark
container while reusing the same recovery implementation as mini-swe-agent.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


IPC_ROOT = Path(os.environ.get("TRAJREC_IPC_ROOT", "/trajrec_ipc"))
TIMEOUT_SECONDS = float(os.environ.get("TRAJREC_HOOK_TIMEOUT_SECONDS", "210"))


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _recent_solver_thoughts(transcript_path: object) -> list[str]:
    """Extract recent assistant text without forwarding large tool/code observations."""
    if not isinstance(transcript_path, str) or not transcript_path:
        return []
    path = Path(transcript_path)
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 1_000_000))
            chunk = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    if size > len(chunk.encode("utf-8", errors="ignore")):
        chunk = chunk.split("\n", 1)[-1]
    thoughts: list[str] = []
    for raw in chunk.splitlines()[-250:]:
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("type") != "response_item":
            continue
        item = row.get("payload")
        if not isinstance(item, dict) or item.get("type") != "message" or item.get("role") != "assistant":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip() and text not in thoughts:
                thoughts.append(text)
    return thoughts[-20:]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        payload["trajectory_thoughts"] = _recent_solver_thoughts(payload.get("transcript_path"))
        event_name = str(payload.get("hook_event_name") or "hook")
        request_id = f"{event_name}-{os.getpid()}-{uuid.uuid4().hex}"
        requests = IPC_ROOT / "requests"
        responses = IPC_ROOT / "responses"
        requests.mkdir(parents=True, exist_ok=True)
        responses.mkdir(parents=True, exist_ok=True)
        request_path = requests / f"{request_id}.json"
        response_path = responses / f"{request_id}.json"
        _atomic_json(request_path, payload)

        deadline = time.monotonic() + TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if response_path.exists():
                response = json.loads(response_path.read_text(encoding="utf-8"))
                response_path.unlink(missing_ok=True)
                hook_output = response.get("hook_output")
                if hook_output:
                    sys.stdout.write(json.dumps(hook_output, ensure_ascii=False))
                return 0
            time.sleep(0.02)
        print(
            f"trajectory recovery hook timed out after {TIMEOUT_SECONDS:.0f}s; failing open",
            file=sys.stderr,
        )
        return 0
    except Exception as exc:
        # Hook transport must never break a repair run. The host controller
        # records policy failures separately and applies its configured policy.
        print(f"trajectory recovery hook bridge failed open: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
