"""Benchmark-runner helpers shared by the mini-swe-agent and Codex runners.

Covers dataset loading and instance selection, the shared results layout,
exit-status bookkeeping, live trajectory watching, and Docker image cleanup.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

from src.utils.docker import image_rm, prune, remove_minisweagent_containers, stop_and_remove_containers_for_image
from src.utils.io import ensure_dir
from src.utils.log import log_message


def read_exit_statuses_yaml(path: Path) -> tuple[dict[str, list[str]], float | None]:
    if not path.exists():
        return {}, None

    statuses: dict[str, list[str]] = {}
    total_cost: float | None = None
    current_section: str | None = None
    current_status: str | None = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue

        if line.startswith("instances_by_exit_status:"):
            current_section = "instances_by_exit_status"
            current_status = None
            continue

        if line.startswith("total_cost:"):
            current_section = None
            current_status = None
            _, val = line.split(":", 1)
            try:
                total_cost = float(val.strip())
            except ValueError:
                total_cost = None
            continue

        if current_section != "instances_by_exit_status":
            continue

        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith("-"):
            current_status = stripped[:-1]
            statuses.setdefault(current_status, [])
            continue
        if stripped.startswith("-") and current_status:
            iid = stripped[1:].strip()
            if iid:
                statuses.setdefault(current_status, []).append(iid)

    return statuses, total_cost


def write_exit_statuses_yaml(path: Path, instances_by_exit_status: dict[str, list[str]], *, total_cost: float | None) -> None:
    lines: list[str] = ["instances_by_exit_status:"]
    for status in sorted(instances_by_exit_status.keys()):
        lines.append(f"  {status}:")
        for iid in instances_by_exit_status[status]:
            lines.append(f"    - {iid}")
    if total_cost is not None:
        lines.append(f"total_cost: {total_cost:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_exit_status_hint(exit_status: Any) -> str:
    s = str(exit_status or "").strip()
    return s.lower() if s else ""


def record_exit_status(agg_statuses: dict[str, list[str]], exit_status: str, instance_id: str) -> None:
    status = str(exit_status or "").strip() or "unknown"
    ids = agg_statuses.setdefault(status, [])
    if instance_id not in ids:
        ids.append(instance_id)


def recorded_instance_ids(agg_statuses: dict[str, list[str]]) -> set[str]:
    return {iid for ids in agg_statuses.values() for iid in ids}


def summarize_actions(message: dict[str, Any]) -> str | None:
    extra = message.get("extra") or {}
    actions = extra.get("actions") or []
    if not actions:
        return None

    first = actions[0]
    if isinstance(first, str):
        s = first.strip().replace("\n", " ")
        head = s[:120]
        summary = f"Action: {head}" if head else "Action: <empty>"
    elif isinstance(first, dict):
        cmd = first.get("command")
        if isinstance(cmd, str) and cmd.strip():
            head = cmd.strip().replace("\n", " ")[:120]
            summary = f"Run: {head}"
        else:
            keys = ",".join(sorted(map(str, first.keys())))
            summary = f"Action keys: {keys[:120]}"
    else:
        summary = f"Action: {type(first).__name__}"

    if len(actions) > 1:
        summary = f"{summary} (+{len(actions) - 1})"
    return summary


def watch_traj(path: Path, instance_id: str, stop: threading.Event) -> None:
    last_calls: int | None = None
    last_summary: str | None = None
    while not stop.is_set():
        if not path.exists():
            time.sleep(0.2)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            time.sleep(0.2)
            continue

        info = data.get("info") or {}
        ms = info.get("model_stats") or {}
        calls = ms.get("api_calls")
        try:
            calls_i = int(calls) if calls is not None else None
        except Exception:
            calls_i = None

        msgs = data.get("messages") or []
        last_msg = msgs[-1] if isinstance(msgs, list) and msgs else {}
        summary = summarize_actions(last_msg) or last_msg.get("role")
        if summary is not None:
            summary = str(summary)

        changed = (calls_i is not None and calls_i != last_calls) or (summary and summary != last_summary)
        if changed:
            last_calls = calls_i
            last_summary = summary
            step_txt = f"step={calls_i}" if calls_i is not None else "step=?"
            if summary:
                log_message(f"Instance={instance_id} {step_txt} {summary}")
            else:
                log_message(f"Instance={instance_id} {step_txt}")

        time.sleep(0.5)


def extract_instance_summary(rep_dir: Path, instance_id: str) -> dict[str, Any] | None:
    traj_path = rep_dir / instance_id / f"{instance_id}.traj.json"
    if traj_path.exists():
        data = json.loads(traj_path.read_text(encoding="utf-8"))
        info = data.get("info") or {}
        ms = info.get("model_stats") or {}
        submission = info.get("submission")
        return {
            "exit_status": info.get("exit_status"),
            "submission_len": len(submission or ""),
            "api_calls": ms.get("api_calls"),
            "instance_cost": ms.get("instance_cost"),
        }

    # If env startup fails early, mini-swe-agent may still write per-instance exit status YAML
    # but won't have a trajectory.
    per_instance_status_yaml = rep_dir / instance_id / "exit_statuses.yaml"
    statuses, _ = read_exit_statuses_yaml(per_instance_status_yaml)
    for status, ids in statuses.items():
        if instance_id in ids:
            return {"exit_status": status, "submission_len": None, "api_calls": None, "instance_cost": None}
    return None


def get_run_dir(cfg: Any) -> Path:
    return cfg.results_root / cfg.run_name


def get_variant_root(cfg: Any) -> Path:
    return get_run_dir(cfg) / cfg.variant


def normalize_problem_statement(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts: list[str] = []
        for key in [
            "Title",
            "Description",
            "RootCause",
            "StepsToReproduce",
            "ExpectedBehavior",
            "ObservedBehavior",
            "Suggestions",
        ]:
            if key not in value:
                continue
            val = value[key]
            if isinstance(val, list):
                val = "\n".join(f"- {x}" for x in val)
            elif isinstance(val, dict):
                sub: list[str] = []
                for sk, sv in val.items():
                    if isinstance(sv, list):
                        sv = ", ".join(str(x) for x in sv)
                    sub.append(f"{sk}: {sv}")
                val = "\n".join(sub)
            parts.append(f"{key}:\n{val}")
        return "\n\n".join(parts) if parts else json.dumps(value, ensure_ascii=False)
    return str(value)


def filter_records(records: list[dict[str, Any]], *, prefix: str | None, limit: int | None) -> list[dict[str, Any]]:
    if prefix:
        records = [r for r in records if str(r.get("instance_id", "")).startswith(prefix)]
    if limit is not None:
        records = records[: int(limit)]
    return records


def apply_instances_filter(instances: list[dict[str, Any]], cfg: Any) -> list[dict[str, Any]]:
    filter_value = cfg.instances_filter
    if isinstance(filter_value, list):
        allow = {str(x).strip() for x in filter_value if str(x).strip()}
        out = list(instances) if not allow else [i for i in instances if str(i.get("instance_id", "")) in allow]
    else:
        pat = str(filter_value or ".*").strip() or ".*"
        try:
            rx = re.compile(pat)
        except re.error as e:
            raise ValueError(f"Invalid INSTANCES_FILTER regex: {pat!r}: {e}")
        out = [i for i in instances if rx.search(str(i.get("instance_id", "")))]

    if cfg.instances_slice:
        parts = cfg.instances_slice.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid INSTANCES_SLICE (expected start:stop[:step]): {cfg.instances_slice!r}")

        def _to_int(x: str) -> int | None:
            x = x.strip()
            return None if not x else int(x)

        start = _to_int(parts[0])
        stop = _to_int(parts[1])
        step = _to_int(parts[2]) if len(parts) == 3 else None
        out = out[slice(start, stop, step)]

    if cfg.instances_shuffle:
        out = sorted(out, key=lambda x: str(x.get("instance_id", "")))
        random.seed(42)
        random.shuffle(out)
    return out


def cleanup_images_for_instances(instances: list[dict[str, Any]], *, force: bool) -> None:
    remove_minisweagent_containers()
    images: list[str] = []
    seen: set[str] = set()
    for inst in instances:
        img = str(get_swebench_docker_image_name(inst)).strip()
        if not img or img in seen:
            continue
        seen.add(img)
        images.append(img)
    if not images:
        return
    log_message(f"Deleting {len(images)} docker images (force={force})")
    for img in images:
        stop_and_remove_containers_for_image(img)
        image_rm(img, force=force)
    prune()


def setup_output_dirs(cfg: Any) -> tuple[Path, Path, Path]:
    run_dir = get_run_dir(cfg)
    ensure_dir(run_dir)
    for v in ["original", "ablation_direction", "ablation_direction_consistency"]:
        ensure_dir(run_dir / v / "localization")
        ensure_dir(run_dir / v / "repair")

    vroot = get_variant_root(cfg)
    loc_dir = vroot / "localization"
    rep_dir = vroot / "repair"
    ensure_dir(loc_dir)
    ensure_dir(rep_dir)
    return run_dir, vroot, rep_dir


def build_recovery_config(cfg: Any) -> dict[str, Any]:
    return {
        "mode": cfg.trajectory_recovery_mode,
        "model_name": cfg.recovery_model_name,
        "timeout": cfg.recovery_model_timeout_seconds,
        "model_kwargs": cfg.recovery_model_kwargs,
        "context_token_limit": cfg.recovery_context_token_limit,
        "max_submission_blocks": cfg.recovery_max_submission_blocks,
        "max_action_blocks": cfg.recovery_max_action_blocks,
        "repeat_threshold": cfg.recovery_repeat_threshold,
        "fail_open": cfg.recovery_fail_open,
        "review_branches": cfg.recovery_review_branches,
        "gate_mutations": cfg.recovery_gate_mutations,
        "gate_submission": cfg.recovery_gate_submission,
    }
