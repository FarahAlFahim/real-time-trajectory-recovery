"""src/runners/minisweagent_swebench_pro.py

One-click runner for SWE-bench Pro V2 records exported as per-repository JSON.

This file is intentionally separate from ``src/runners/minisweagent_swebench_lite.py``.
The Lite runner remains unchanged. This wrapper reuses the same mini-swe-agent and
trajectory-recovery implementation while adapting only the benchmark contract:

- use each record's explicit ``docker_image`` from GHCR;
- give the solver the problem statement, requirements, and interface;
- detect whether the repository is mounted at ``/app`` or ``/testbed``;
- enforce the V2 offline agent phase and resource limits;
- capture the complete git diff independently of the model's submission text;
- preserve partial patches when the per-instance watchdog terminates a run.

Outputs retain the existing shared layout:

    results/<RUN_NAME>/
        original/
            localization/
            repair/
        realtime_recovery/
            localization/
            repair/
        ablation_direction/
            localization/
            repair/
        ablation_direction_consistency/
            localization/
            repair/

Notes
-----
- This script generates patches. Authoritative SWE-bench Pro V2 scores require
  replaying the captured patches in pristine environments with the official
  Harbor verifier.
- V2 images are linux/amd64 and are public at
  ``ghcr.io/scaleapi/swe-bench_pro-v2``.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Executed by file path (e.g. the per-instance watchdog subprocess): make `src` importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import GLOBAL_MODEL_STATS, get_model
from minisweagent.run.benchmarks.swebench import (
    ProgressTrackingAgent,
    get_sb_environment,
    get_swebench_docker_image_name,
    remove_from_preds_file,
    update_preds_file,
)
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import logger
from minisweagent.utils.serialize import UNSET, recursive_merge

from src.adapters.minisweagent.docker import use_as_docker_environment
from src.runners.common import apply_instances_filter, build_recovery_config, cleanup_images_for_instances, extract_instance_summary, filter_records, fmt_exit_status_hint, normalize_problem_statement, read_exit_statuses_yaml, record_exit_status, recorded_instance_ids, setup_output_dirs, write_exit_statuses_yaml
from src.utils.docker import remove_minisweagent_containers, stop_and_remove_containers_for_image
from src.utils.io import append_jsonl, ensure_dir, load_json_list, read_jsonl, resolve_path
from src.utils.log import log_message

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Route environment_class="docker" to the environment that supports STRIP_GIT_HISTORY.
use_as_docker_environment()

SWE_BENCH_PRO_V2_IMAGE_PREFIX = "ghcr.io/scaleapi/swe-bench_pro-v2:"
SWE_BENCH_PRO_V2_REQUIRED_FIELDS = {
    "instance_id",
    "docker_image",
    "problem_statement",
    "requirements",
    "interface",
    "base_commit",
}


class ProInstanceWatchdogTimeout(RuntimeError):
    """Raised in a child process to preserve its partial patch on timeout."""


PRO_INSTANCE_TEMPLATE = r"""
You are a software engineer working in the repository at {{cwd}}.

Please implement the changes described below. Work independently: do not
reference, look up, or copy external solutions, pull requests, or online
workarounds.

{{task}}

## Workflow

1. Inspect the relevant implementation and its callers.
2. Reproduce or otherwise verify the reported behavior when practical.
3. Implement a complete, general fix consistent with the repository.
4. Run focused tests and inspect the final diff.
5. When finished, issue this exact command by itself:

   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Every response must contain reasoning and at least one bash tool call. Each
tool call runs in a new subshell, so directory changes do not persist. Do not
use interactive editors. The harness captures the final repository diff; do
not create a patch.txt file and do not commit the changes.
""".strip()


@dataclass(frozen=True)
class RunnerConfig:
    variant: str
    dataset_json: Path
    problem_field: str
    reference_json: Path | None
    project_prefix: str | None
    limit_instances: int | None
    instances_filter: str | list[str] | None
    instances_slice: str
    instances_shuffle: bool
    run_name: str
    results_root: Path
    redo_existing: bool
    raise_exceptions: bool
    workers: int
    model_name: str | None
    model_class: str | None
    environment_class: str | None
    disable_network: bool
    strip_git_history: bool
    model_request_timeout_seconds: int | None
    per_instance_call_limit: int | None
    per_instance_cost_limit: float | None
    global_call_limit: int | None
    global_cost_limit: float | None
    config_spec: list[str]
    progress_bar: bool
    run_mode: str
    max_minutes_per_instance: int | None
    pull_docker_images_before_run: bool
    delete_docker_image_after_each_instance: bool
    delete_docker_images_after_run: bool
    force_delete_docker_images: bool
    trajectory_recovery_mode: str
    recovery_model_name: str | None
    recovery_model_timeout_seconds: int
    recovery_model_kwargs: dict[str, Any]
    recovery_context_token_limit: int
    recovery_max_submission_blocks: int
    recovery_max_action_blocks: int
    recovery_repeat_threshold: int
    recovery_fail_open: bool
    recovery_review_branches: bool
    recovery_gate_mutations: bool
    recovery_gate_submission: bool


# --- Progressive ablation: each tier = previous tier + one component ---
# Set VARIANT + TRAJECTORY_RECOVERY_MODE + the three RECOVERY_* flags to match a row.
# (Base and Full are already run; the two middle tiers are the ablation.)
#
#   VARIANT                          MODE     BRANCHES  MUTATIONS  SUBMISSION   = tier
#   original                         off        -          -          -        Base
#   ablation_direction               active    True      False      False      + Direction Recovery (branch prune/redirect + repeat)
#   ablation_direction_consistency   active    True      True       False      + Edit Consistency (pre-edit semantic review)
#   realtime_recovery                active    True      True       True       + Completion Consistency = Full tool
#


CONFIG: dict[str, Any] = {
    # Choose which variant you are running NOW.
    # Allowed:
    #   "original" | "realtime_recovery" | "ablation_direction" | "ablation_direction_consistency"
    "VARIANT": "realtime_recovery",
    # For a recovery experiment, use VARIANT="realtime_recovery" and mode="active", mode="off" otherwise.
    "TRAJECTORY_RECOVERY_MODE": "active", # "off" | "shadow" | "active"

    # Input must be a SWE-bench Pro V2 per-repository JSON list. Enhanced
    # reports may be supplied here when REFERENCE_JSON points to the original
    # V2 records containing docker_image/base_commit/requirements/interface.
    "DATASET_JSON": "data/swe_bench_pro/element-hq__element-web.json",

    # Output layout
    "RUN_NAME": "element-hq__element-web",
    "RESULTS_ROOT": "results_swebench_pro/gpt_5_mini",

    # Instance selection
    # - list/tuple/set: allowlist of instance_ids. Empty list means "all".
    # - str: regex over instance_id
    "INSTANCES_FILTER": [], # e.g. ["matplotlib__matplotlib-23314"]
    "INSTANCES_SLICE": "",  # python-slice string, e.g. ":50" or "10:20"
    "INSTANCES_SHUFFLE": False,

    # Which field to use as the report text.
    # Usually "problem_statement".
    "PROBLEM_FIELD": "problem_statement",
    "REFERENCE_JSON": None,
    "PROJECT_PREFIX": None,
    "LIMIT_INSTANCES": None,
    
    # mini-swe-agent / LiteLLM
    "MODEL_NAME": "openai/gpt-5-mini-2025-08-07",   # "openai/gpt-5-mini-2025-08-07" or "deepseek/deepseek-flash" or "minimax/minimax-m2.5" or "openai/gpt-5.4-mini-2026-03-17" or "openai/gpt-5.4-2026-03-05"
    "MODEL_CLASS": None, # None | "openrouter" [None - for openai gpt/deepseek models]
    # Cost tracking mode for LiteLLM model price lookup.
    # - "default": fail instance if model pricing metadata is missing
    # - "ignore_errors": continue even if cost can't be computed
    "MSWEA_COST_TRACKING": "ignore_errors",
    "ENVIRONMENT_CLASS": "docker",
    # SWE-bench Pro V2's agent phase is always offline. Its images already
    # contain sanitized git histories, so history rewriting must remain off.
    "DISABLE_NETWORK": True,
    "STRIP_GIT_HISTORY": False,
    "MODEL_REQUEST_TIMEOUT_SECONDS": 180,

    # Real-time trajectory recovery (strictly opt-in).
    # - "off": normal mini-swe-agent with the Pro V2 environment adapter.
    # - "shadow": review and record decisions, but never intervene.
    # - "active": inject course corrections and gate edits/submission before execution.
    # For a recovery experiment, use VARIANT="realtime_recovery" and mode="active".
    # "TRAJECTORY_RECOVERY_MODE": "active",
    # None reuses MODEL_NAME. This reviewer is independent from the solver call.
    "RECOVERY_MODEL_NAME": None,
    "RECOVERY_MODEL_TIMEOUT_SECONDS": 180,
    "RECOVERY_MODEL_KWARGS": {},
    # Preserve canonical scope evidence and exact code constraints up to this
    # total prompt budget. On overflow, retain solver thoughts and omit large
    # observation/code blocks; the complete audit ledger remains on disk.
    "RECOVERY_CONTEXT_TOKEN_LIMIT": 250_000,
    # Branch prunes are unlimited across distinct explored scopes. Submission
    # and mutation blocks are separately bounded to guarantee progress.
    "RECOVERY_MAX_SUBMISSION_BLOCKS": 2,
    "RECOVERY_MAX_ACTION_BLOCKS": 3,
    # Review the third identical exploration, or sustained revisits to one scope.
    "RECOVERY_REPEAT_THRESHOLD": 2,
    # Fail-open prevents reviewer/API failure from breaking the solver run.
    "RECOVERY_FAIL_OPEN": True,
    "RECOVERY_REVIEW_BRANCHES": True,
    "RECOVERY_GATE_MUTATIONS": True,
    "RECOVERY_GATE_SUBMISSION": True,
    # Per-instance safety limits (map to agent.step_limit / agent.cost_limit). Set to 0 to disable.
    # V2 is a long-horizon benchmark. The official locked mini-SWE-agent config
    # uses no step cap, a $3 cost cap, and a 50-minute wall-clock budget.
    "PER_INSTANCE_CALL_LIMIT": 0,
    "PER_INSTANCE_COST_LIMIT": 3.0,
    # Global limits across whole run (enforced by GLOBAL_MODEL_STATS). Set to 0/None to disable.
    "GLOBAL_CALL_LIMIT": 0,
    "GLOBAL_COST_LIMIT": 0.0,
    "CONFIG_SPEC": [
        str(builtin_config_dir / "benchmarks" / "swebench.yaml"),
    ],
    "WORKERS": 1,
    "PROGRESS_BAR": True,
    # Execution mode
    # - "batch": in-process multi-threaded run
    # - "per_instance": per-instance subprocess with watchdog timeout (recommended)
    "RUN_MODE": "per_instance",
    "MAX_MINUTES_PER_INSTANCE": 60,
    "PULL_DOCKER_IMAGES_BEFORE_RUN": True,
    "DELETE_DOCKER_IMAGE_AFTER_EACH_INSTANCE": False, # True | False
    
    # Behavior
    "REDO_EXISTING": False,
    "RAISE_EXCEPTIONS": False,
    "DELETE_DOCKER_IMAGES_AFTER_RUN": False, # True | False
    "FORCE_DELETE_DOCKER_IMAGES": False, # True | False
    # Safety
    "DRY_RUN": False,

    # Logging
    # If True, prints per-step/action updates by watching the trajectory file while the agent runs.
    "STREAM_STEP_ACTIONS": True,

    # Watchdog logging cadence (per-instance subprocess mode)
    "WATCHDOG_TICK_SECONDS": 10,
    # Time allowed after SIGTERM for the child to capture and save a partial
    # patch before the parent falls back to SIGKILL.
    "WATCHDOG_GRACE_SECONDS": 60,
}


def _merge_selected_instances_manifest(
    path: Path,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Persist the union of instances selected across repeated runs."""
    accumulated: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = []
        if isinstance(existing, list):
            accumulated.extend(item for item in existing if isinstance(item, dict))

    positions = {
        str(item.get("instance_id")): index
        for index, item in enumerate(accumulated)
        if item.get("instance_id")
    }
    for item in selected:
        instance_id = str(item.get("instance_id") or "").strip()
        if not instance_id:
            continue
        if instance_id in positions:
            accumulated[positions[instance_id]] = item
        else:
            positions[instance_id] = len(accumulated)
            accumulated.append(item)

    path.write_text(json.dumps(accumulated, indent=2), encoding="utf-8")
    return accumulated


def _compose_pro_task(problem_statement: Any, requirements: Any, interface: Any) -> str:
    """Build the complete V2 instruction visible to the repair agent."""

    problem = normalize_problem_statement(problem_statement).strip()
    requirements_text = normalize_problem_statement(requirements).strip()
    interface_text = normalize_problem_statement(interface).strip()
    if not problem:
        raise ValueError("SWE-bench Pro record has an empty problem_statement")
    if not requirements_text:
        raise ValueError("SWE-bench Pro record has empty requirements")
    if not interface_text:
        raise ValueError("SWE-bench Pro record has an empty interface")
    return (
        "<pr_description>\n"
        f"{problem}\n\n"
        "## Requirements\n"
        f"{requirements_text}\n\n"
        "## New Interfaces\n"
        f"{interface_text}\n"
        "</pr_description>"
    )


def _validate_pro_record(record: dict[str, Any], *, source: Path) -> None:
    missing = sorted(
        field
        for field in SWE_BENCH_PRO_V2_REQUIRED_FIELDS
        if record.get(field) is None or str(record.get(field)).strip() == ""
    )
    if missing:
        iid = record.get("instance_id", "<unknown>")
        raise ValueError(f"{source}: {iid} missing required V2 fields: {missing}")
    image = str(record["docker_image"]).strip()
    if not image.startswith(SWE_BENCH_PRO_V2_IMAGE_PREFIX):
        raise ValueError(
            f"{source}: {record['instance_id']} has non-V2 docker_image: {image!r}"
        )


def _pull_docker_images(instances: list[dict[str, Any]]) -> None:
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
    log_message(f"Pre-pulling {len(images)} docker images")
    for img in images:
        subprocess.run(
            ["docker", "pull", "--platform", "linux/amd64", img],
            check=True,
        )


def _build_runner_config() -> RunnerConfig:
    dataset_json = resolve_path(CONFIG.get("DATASET_JSON"), base_dir=_REPO_ROOT)
    if dataset_json is None:
        raise FileNotFoundError("DATASET_JSON is empty")

    results_root = resolve_path(CONFIG.get("RESULTS_ROOT"), base_dir=_REPO_ROOT) or (_REPO_ROOT / "results")

    return RunnerConfig(
        variant=str(CONFIG.get("VARIANT")),
        dataset_json=dataset_json,
        problem_field=str(CONFIG.get("PROBLEM_FIELD") or "problem_statement"),
        reference_json=resolve_path(CONFIG.get("REFERENCE_JSON"), base_dir=_REPO_ROOT),
        project_prefix=CONFIG.get("PROJECT_PREFIX"),
        limit_instances=CONFIG.get("LIMIT_INSTANCES"),
        instances_filter=CONFIG.get("INSTANCES_FILTER"),
        instances_slice=str(CONFIG.get("INSTANCES_SLICE") or ""),
        instances_shuffle=bool(CONFIG.get("INSTANCES_SHUFFLE", False)),
        run_name=str(CONFIG.get("RUN_NAME")),
        results_root=results_root,
        redo_existing=bool(CONFIG.get("REDO_EXISTING", False)),
        raise_exceptions=bool(CONFIG.get("RAISE_EXCEPTIONS", False)),
        workers=int(CONFIG.get("WORKERS", 1)),
        model_name=CONFIG.get("MODEL_NAME"),
        model_class=CONFIG.get("MODEL_CLASS"),
        environment_class=CONFIG.get("ENVIRONMENT_CLASS"),
        disable_network=bool(CONFIG.get("DISABLE_NETWORK", True)),
        strip_git_history=bool(CONFIG.get("STRIP_GIT_HISTORY", False)),
        model_request_timeout_seconds=CONFIG.get("MODEL_REQUEST_TIMEOUT_SECONDS"),
        per_instance_call_limit=CONFIG.get("PER_INSTANCE_CALL_LIMIT"),
        per_instance_cost_limit=CONFIG.get("PER_INSTANCE_COST_LIMIT"),
        global_call_limit=CONFIG.get("GLOBAL_CALL_LIMIT"),
        global_cost_limit=CONFIG.get("GLOBAL_COST_LIMIT"),
        config_spec=list(CONFIG.get("CONFIG_SPEC") or []),
        progress_bar=bool(CONFIG.get("PROGRESS_BAR", True)),
        run_mode=str(CONFIG.get("RUN_MODE", "batch")),
        max_minutes_per_instance=CONFIG.get("MAX_MINUTES_PER_INSTANCE"),
        pull_docker_images_before_run=bool(CONFIG.get("PULL_DOCKER_IMAGES_BEFORE_RUN", False)),
        delete_docker_image_after_each_instance=bool(CONFIG.get("DELETE_DOCKER_IMAGE_AFTER_EACH_INSTANCE", False)),
        delete_docker_images_after_run=bool(CONFIG.get("DELETE_DOCKER_IMAGES_AFTER_RUN", False)),
        force_delete_docker_images=bool(CONFIG.get("FORCE_DELETE_DOCKER_IMAGES", True)),
        trajectory_recovery_mode=str(CONFIG.get("TRAJECTORY_RECOVERY_MODE", "off")).strip().lower(),
        recovery_model_name=CONFIG.get("RECOVERY_MODEL_NAME"),
        recovery_model_timeout_seconds=int(CONFIG.get("RECOVERY_MODEL_TIMEOUT_SECONDS", 180)),
        recovery_model_kwargs=dict(CONFIG.get("RECOVERY_MODEL_KWARGS") or {}),
        recovery_context_token_limit=int(CONFIG.get("RECOVERY_CONTEXT_TOKEN_LIMIT", 250_000)),
        recovery_max_submission_blocks=int(CONFIG.get("RECOVERY_MAX_SUBMISSION_BLOCKS", 2)),
        recovery_max_action_blocks=int(CONFIG.get("RECOVERY_MAX_ACTION_BLOCKS", 3)),
        recovery_repeat_threshold=int(CONFIG.get("RECOVERY_REPEAT_THRESHOLD", 2)),
        recovery_fail_open=bool(CONFIG.get("RECOVERY_FAIL_OPEN", True)),
        recovery_review_branches=bool(CONFIG.get("RECOVERY_REVIEW_BRANCHES", True)),
        recovery_gate_mutations=bool(CONFIG.get("RECOVERY_GATE_MUTATIONS", True)),
        recovery_gate_submission=bool(CONFIG.get("RECOVERY_GATE_SUBMISSION", True)),
    )


def _build_minisweagent_config(cfg: RunnerConfig) -> dict[str, Any]:
    if not cfg.config_spec:
        raise ValueError("CONFIG_SPEC must include at least swebench.yaml")
    if cfg.environment_class != "docker":
        raise ValueError("SWE-bench Pro V2 wrapper currently requires ENVIRONMENT_CLASS='docker'")
    if not cfg.disable_network:
        raise ValueError("SWE-bench Pro V2 requires DISABLE_NETWORK=True during the agent phase")
    if cfg.strip_git_history:
        raise ValueError(
            "SWE-bench Pro V2 images are already sanitized; STRIP_GIT_HISTORY must be False"
        )

    configs = [get_config_from_spec(spec) for spec in cfg.config_spec]
    configs.append(
        {
            "agent": {
                "instance_template": PRO_INSTANCE_TEMPLATE,
                "step_limit": cfg.per_instance_call_limit if cfg.per_instance_call_limit is not None else UNSET,
                "cost_limit": cfg.per_instance_cost_limit if cfg.per_instance_cost_limit is not None else UNSET,
            },
            "model": {
                "model_name": cfg.model_name or UNSET,
                "model_class": cfg.model_class or UNSET,
                "model_kwargs": {
                    "timeout": cfg.model_request_timeout_seconds
                    if cfg.model_request_timeout_seconds is not None
                    else UNSET,
                    # Recovery must be able to gate each proposed action before
                    # execution. Normal runs retain the original model config.
                    "parallel_tool_calls": False if cfg.trajectory_recovery_mode == "active" else UNSET,
                },
            },
            "environment": {
                "environment_class": cfg.environment_class or UNSET,
                # Start at / because V2 uses /app for most tasks and /testbed
                # for a small number. The exact repository is detected before
                # the agent receives its first message.
                "cwd": "/",
                "pull_timeout": max(120, int(cfg.max_minutes_per_instance or 15) * 60),
                "run_args": [
                    "--rm",
                    "--network",
                    "none",
                    "--platform",
                    "linux/amd64",
                    "--cpus",
                    "1",
                    "--memory",
                    "4096m",
                ],
                "strip_git_history": False,
            },
            "run": {
                "redo_existing": cfg.redo_existing or UNSET,
            },
        }
    )
    return recursive_merge(*configs)


def _load_variant_instances(cfg: RunnerConfig) -> list[dict[str, Any]]:
    if not cfg.dataset_json.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {cfg.dataset_json}")

    data = load_json_list(cfg.dataset_json)
    ref_map: dict[str, dict[str, Any]] = {}
    if cfg.reference_json is not None:
        for r in load_json_list(cfg.reference_json):
            iid = r.get("instance_id")
            if iid:
                ref_map[str(iid)] = r

    records: list[dict[str, Any]] = []
    for r in data:
        iid = r.get("instance_id")
        if not iid:
            continue
        base = dict(ref_map.get(str(iid), r))
        _validate_pro_record(base, source=cfg.reference_json or cfg.dataset_json)
        base["problem_statement"] = _compose_pro_task(
            r.get(cfg.problem_field, r.get("problem_statement", "")),
            r.get("requirements", base.get("requirements", "")),
            r.get("interface", base.get("interface", "")),
        )
        records.append(base)

    records = filter_records(records, prefix=cfg.project_prefix, limit=cfg.limit_instances)
    log_message(f"Variant={cfg.variant} records ready: {len(records)}")

    instances: list[dict[str, Any]] = []
    for r in records:
        iid = str(r.get("instance_id", "")).strip()
        if not iid:
            continue
        image_name = str(r["docker_image"]).strip()
        instances.append(
            {
                "instance_id": iid,
                "problem_statement": str(r.get("problem_statement", "")),
                "image_name": image_name,
                **{
                    k: v
                    for k, v in r.items()
                    if k not in {"problem_statement", "image_name"}
                },
            }
        )

    return instances

def _prepare_pro_environment(config: dict[str, Any], instance: dict[str, Any]) -> tuple[Any, str]:
    """Start a V2 image, locate its repository, and verify its clean base state."""

    env = get_sb_environment(config, instance)
    locate = env.execute(
        {
            "command": (
                "for candidate in /app /testbed; do "
                "if git -C \"$candidate\" rev-parse --show-toplevel >/dev/null 2>&1; then "
                "git -C \"$candidate\" rev-parse --show-toplevel; exit 0; fi; "
                "done; echo 'No git repository found at /app or /testbed' >&2; exit 1"
            )
        },
        cwd="/",
        timeout=120,
    )
    if locate.get("returncode") != 0:
        env.cleanup()
        raise RuntimeError(
            f"Unable to locate V2 repository in {instance['image_name']}: "
            f"{locate.get('output', '')[-2000:]}"
        )
    lines = [line.strip() for line in str(locate.get("output", "")).splitlines() if line.strip()]
    if not lines or lines[-1] not in {"/app", "/testbed"}:
        env.cleanup()
        raise RuntimeError(f"Unexpected V2 repository path response: {locate.get('output', '')!r}")
    repo_cwd = lines[-1]
    env.config.cwd = repo_cwd

    expected_commit = str(instance.get("base_commit", "")).strip()
    quoted_commit = shlex.quote(expected_commit)
    verify = env.execute(
        {
            "command": (
                "actual=$(git rev-parse HEAD) || exit 1; "
                f"test \"$actual\" = {quoted_commit} || "
                "{ echo \"base commit mismatch: expected "
                f"{expected_commit}, got $actual\" >&2; exit 2; }}; "
                # Every Pro V2 record declares reset/clean as the first two
                # repository-setup operations. Reproduce those operations but
                # deliberately don't apply the hidden test patch that follows.
                f"git reset --hard {quoted_commit} >/dev/null || exit 3; "
                "git clean -fd >/dev/null || exit 3; "
                "dirty=$(git status --porcelain --untracked-files=all) || exit 1; "
                "test -z \"$dirty\" || "
                "{ echo 'repository is not clean at agent start' >&2; "
                "printf '%s\\n' \"$dirty\" >&2; exit 4; }"
            )
        },
        cwd=repo_cwd,
        timeout=120,
    )
    if verify.get("returncode") != 0:
        env.cleanup()
        raise RuntimeError(
            f"Invalid V2 base state for {instance['instance_id']}: "
            f"{verify.get('output', '')[-4000:]}"
        )
    return env, repo_cwd


def _capture_pro_patch(env: Any, repo_cwd: str, base_commit: str) -> str:
    """Capture the final repository state relative to the verified V2 base."""

    if not base_commit:
        raise ValueError("Cannot capture a Pro patch without base_commit")
    quoted_base = shlex.quote(base_commit)

    captured = env.execute(
        {
            "command": (
                f"git cat-file -e {quoted_base}^{{commit}} || exit 1; "
                "git add -A; add_rc=$?; "
                "if [ $add_rc -ne 0 ]; then exit $add_rc; fi; "
                f"git diff --cached --no-ext-diff {quoted_base} --; diff_rc=$?; "
                "git reset -q; exit $diff_rc"
            )
        },
        cwd=repo_cwd,
        timeout=300,
    )
    if captured.get("returncode") != 0:
        raise RuntimeError(f"Patch capture failed: {captured.get('output', '')[-4000:]}")
    return str(captured.get("output", ""))


def _process_pro_instance(
    cfg: RunnerConfig,
    instance: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
    progress_manager: RunBatchProgressManager,
) -> None:
    """Run one Pro V2 instance and always capture the repository diff."""

    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    traj_path = instance_dir / f"{instance_id}.traj.json"
    model_patch_path = instance_dir / "model.patch"
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    traj_path.unlink(missing_ok=True)
    model_patch_path.unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]
    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting Pro V2 image")

    env = None
    agent = None
    repo_cwd = ""
    exit_status: str | None = None
    raw_submission = ""
    result = ""
    extra_info: dict[str, Any] = {
        "benchmark": "swe-bench-pro-v2",
        "docker_image": instance.get("image_name"),
        "base_commit": instance.get("base_commit"),
    }

    try:
        env, repo_cwd = _prepare_pro_environment(config, instance)
        extra_info["repository_cwd"] = repo_cwd
        agent_config = dict(config.get("agent", {}))

        if cfg.trajectory_recovery_mode == "off":
            agent = ProgressTrackingAgent(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                **agent_config,
            )
        else:
            from src.recovery import (
                LiteLLMRecoveryPolicy,
                RecoveryProgressTrackingAgent,
            )

            recovery_config = build_recovery_config(cfg)
            reviewer_model = recovery_config.get("model_name") or model.config.model_name
            policy = LiteLLMRecoveryPolicy(
                reviewer_model,
                timeout=int(recovery_config.get("timeout", 180)),
                model_kwargs=recovery_config.get("model_kwargs") or {},
                context_token_limit=int(recovery_config.get("context_token_limit", 250_000)),
            )
            agent_config["output_path"] = traj_path
            agent = RecoveryProgressTrackingAgent(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                recovery_policy=policy,
                recovery_mode=recovery_config.get("mode", "active"),
                recovery_max_submission_blocks=int(recovery_config.get("max_submission_blocks", 2)),
                recovery_max_action_blocks=int(recovery_config.get("max_action_blocks", 3)),
                recovery_repeat_threshold=int(recovery_config.get("repeat_threshold", 2)),
                recovery_fail_open=bool(recovery_config.get("fail_open", True)),
                recovery_review_branches=bool(recovery_config.get("review_branches", True)),
                recovery_gate_mutations=bool(recovery_config.get("gate_mutations", True)),
                recovery_gate_submission=bool(recovery_config.get("gate_submission", True)),
                **agent_config,
            )

        info = agent.run(task)
        exit_status = str(info.get("exit_status") or "")
        raw_submission = str(info.get("submission") or "")
    except Exception as exc:
        logger.error(f"Error processing Pro instance {instance_id}: {exc}", exc_info=True)
        exit_status = type(exc).__name__
        extra_info.update({"traceback": traceback.format_exc(), "exception_str": str(exc)})
    finally:
        if env is not None and repo_cwd:
            try:
                result = _capture_pro_patch(
                    env,
                    repo_cwd,
                    str(instance.get("base_commit", "")).strip(),
                )
                extra_info["patch_source"] = "repository_diff"
            except Exception as capture_exc:
                logger.error(f"Patch capture failed for {instance_id}: {capture_exc}", exc_info=True)
                extra_info["patch_source"] = "capture_failed"
                extra_info["patch_capture_error"] = str(capture_exc)
                result = ""
        else:
            extra_info["patch_source"] = "environment_unavailable"

        extra_info["agent_submission"] = raw_submission
        model_patch_path.write_text(result, encoding="utf-8")
        if agent is not None:
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved Pro trajectory to '{traj_path}'")
        else:
            traj_path.write_text(
                json.dumps(
                    {
                        "info": {
                            "exit_status": exit_status,
                            "submission": result,
                            **extra_info,
                        },
                        "instance_id": instance_id,
                        "messages": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)
        if env is not None:
            env.cleanup()


def _process_instance_for_config(
    cfg: RunnerConfig,
    inst: dict[str, Any],
    rep_dir: Path,
    miniconfig: dict[str, Any],
    progress_manager: RunBatchProgressManager,
) -> None:
    _process_pro_instance(cfg, inst, rep_dir, miniconfig, progress_manager)


def _run_one_instance(instance_path: Path, rep_dir: Path) -> None:
    inst = json.loads(instance_path.read_text(encoding="utf-8"))
    if not isinstance(inst, dict):
        raise ValueError(f"One-instance file must contain a JSON object: {instance_path}")

    cfg = _build_runner_config()
    miniconfig = _build_minisweagent_config(cfg)

    if cfg.global_call_limit is not None:
        GLOBAL_MODEL_STATS.call_limit = int(cfg.global_call_limit)
    if cfg.global_cost_limit is not None:
        GLOBAL_MODEL_STATS.cost_limit = float(cfg.global_cost_limit)

    iid = str(inst.get("instance_id", ""))
    (rep_dir / iid).mkdir(parents=True, exist_ok=True)
    progress_manager = RunBatchProgressManager(1, rep_dir / iid / "exit_statuses.yaml")

    live_cm = Live(progress_manager.render_group, refresh_per_second=4) if cfg.progress_bar else None
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _handle_watchdog_sigterm(signum: int, frame: Any) -> None:
        raise ProInstanceWatchdogTimeout(
            f"Parent watchdog requested graceful termination for {iid}"
        )

    signal.signal(signal.SIGTERM, _handle_watchdog_sigterm)
    if live_cm:
        live_cm.__enter__()
    try:
        _process_instance_for_config(cfg, inst, rep_dir, miniconfig, progress_manager)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if live_cm:
            live_cm.__exit__(None, None, None)

    preds_path = rep_dir / "preds.json"
    if preds_path.exists() and iid:
        preds = json.loads(preds_path.read_text(encoding="utf-8"))
        patch = str((preds.get(iid) or {}).get("model_patch") or "")
        (rep_dir / iid).mkdir(parents=True, exist_ok=True)
        (rep_dir / iid / f"{iid}.patch").write_text(patch, encoding="utf-8")


def _write_pro_predictions_file(rep_dir: Path, model_name: str | None) -> None:
    """Write a simple Pro patch bundle in addition to mini-SWE-agent preds.json."""

    preds_path = rep_dir / "preds.json"
    if not preds_path.exists():
        return
    raw = json.loads(preds_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected predictions format in {preds_path}")
    bundle = [
        {
            "instance_id": instance_id,
            "patch": str(entry.get("model_patch") or ""),
            "prefix": str(model_name or "mini-swe-agent"),
        }
        for instance_id, entry in sorted(raw.items())
    ]
    (rep_dir / "pro_predictions.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _instances_filter_is_nonempty() -> bool:
    fv = CONFIG.get("INSTANCES_FILTER")
    if isinstance(fv, (list, tuple, set)):
        return bool(fv)
    if isinstance(fv, str):
        return bool(fv.strip()) and fv.strip() != ".*"
    return False


def run() -> None:
    cost_tracking = str(CONFIG.get("MSWEA_COST_TRACKING") or "").strip()
    if cost_tracking:
        os.environ["MSWEA_COST_TRACKING"] = cost_tracking

    if len(sys.argv) >= 4 and sys.argv[1] == "--one-instance":
        _run_one_instance(Path(sys.argv[2]), Path(sys.argv[3]))
        return

    if _instances_filter_is_nonempty() and not CONFIG.get("REDO_EXISTING", False):
        log_message("INSTANCES_FILTER is non-empty → forcing REDO_EXISTING=True")
        CONFIG["REDO_EXISTING"] = True

    cfg = _build_runner_config()
    if cfg.trajectory_recovery_mode not in {"off", "shadow", "active"}:
        raise ValueError(
            "TRAJECTORY_RECOVERY_MODE must be 'off', 'shadow', or 'active', "
            f"got: {cfg.trajectory_recovery_mode!r}"
        )
    _, vroot, rep_dir = setup_output_dirs(cfg)

    instances = _load_variant_instances(cfg)
    if not instances:
        raise ValueError("No instances produced. Check your dataset/filters.")

    (vroot / "instances.json").write_text(json.dumps(instances, indent=2), encoding="utf-8")

    selected = apply_instances_filter(instances, cfg)
    if not selected:
        raise ValueError("No instances selected after applying INSTANCES_FILTER/SLICE")
    selected_manifest = _merge_selected_instances_manifest(
        vroot / "instances_selected.json",
        selected,
    )
    log_message(
        f"Selected {len(selected)} instances for this run; "
        f"selection manifest contains {len(selected_manifest)} unique instances"
    )

    miniconfig = _build_minisweagent_config(cfg)

    if cfg.global_call_limit is not None:
        GLOBAL_MODEL_STATS.call_limit = int(cfg.global_call_limit)
    if cfg.global_cost_limit is not None:
        GLOBAL_MODEL_STATS.cost_limit = float(cfg.global_cost_limit)

    (vroot / "run_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "benchmark": "swe-bench-pro-v2",
                "variant": cfg.variant,
                "dataset_json": str(cfg.dataset_json),
                "reference_json": str(cfg.reference_json) if cfg.reference_json else None,
                "problem_field": cfg.problem_field,
                "workers": cfg.workers,
                "model": {"model_name": cfg.model_name, "model_class": cfg.model_class},
                "environment": {
                    "environment_class": cfg.environment_class,
                    "repository_path_detection": ["/app", "/testbed"],
                    "network": "none",
                    "platform": "linux/amd64",
                    "cpus": 1,
                    "memory_mb": 4096,
                    "strip_git_history": False,
                },
                "patch_capture": "git_add_all_then_cached_diff",
                "authoritative_evaluation": "official_harbor_fresh_sandbox_replay_required",
                "config_spec": cfg.config_spec,
                "run_mode": cfg.run_mode,
                **(
                    {"trajectory_recovery": build_recovery_config(cfg)}
                    if cfg.trajectory_recovery_mode != "off"
                    else {}
                ),
                "watchdog": {
                    "max_minutes_per_instance": cfg.max_minutes_per_instance,
                    "delete_docker_image_after_each_instance": cfg.delete_docker_image_after_each_instance,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if bool(CONFIG.get("DRY_RUN", False)):
        log_message("DRY_RUN=True; not executing.")
        log_message(f"Would run {len(selected)} instances and write results to: {rep_dir}")
        return

    if cfg.pull_docker_images_before_run:
        _pull_docker_images(selected)

    run_mode = cfg.run_mode.strip().lower()
    if run_mode not in {"batch", "per_instance"}:
        raise ValueError(f"RUN_MODE must be 'batch' or 'per_instance', got: {cfg.run_mode!r}")

    if run_mode == "batch":
        progress_manager = RunBatchProgressManager(len(selected), rep_dir / f"exit_statuses_{time.time()}.yaml")

        def _submit(inst: dict[str, Any]) -> None:
            return _process_instance_for_config(cfg, inst, rep_dir, miniconfig, progress_manager)

        live_cm = Live(progress_manager.render_group, refresh_per_second=4) if cfg.progress_bar else None
        if live_cm:
            live_cm.__enter__()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.workers) as ex:
                futures = {ex.submit(_submit, inst): str(inst.get("instance_id", "")) for inst in selected}
                for fut in concurrent.futures.as_completed(futures):
                    iid = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        progress_manager.on_uncaught_exception(iid, e)
                        if cfg.raise_exceptions:
                            raise
        finally:
            if live_cm:
                live_cm.__exit__(None, None, None)
    else:
        max_minutes = int(cfg.max_minutes_per_instance or 15)
        timeout_seconds = max(60, max_minutes * 60)
        watchdog_tick_seconds = int(CONFIG.get("WATCHDOG_TICK_SECONDS", 10) or 10)
        watchdog_tick_seconds = max(1, watchdog_tick_seconds)
        watchdog_grace_seconds = max(
            10, int(CONFIG.get("WATCHDOG_GRACE_SECONDS", 60) or 60)
        )
        status_path = vroot / "per_instance_status.jsonl"
        skipped_path = vroot / "skipped_instances.jsonl"
        one_dir = vroot / "instances_one"
        ensure_dir(one_dir)
        exit_status_path = rep_dir / "run_batch_exit_statuses.yaml"
        agg_statuses, agg_cost = read_exit_statuses_yaml(exit_status_path)
        agg_cost = float(agg_cost or 0.0)
        recorded = recorded_instance_ids(agg_statuses)

        # Backfill missing instances using prior per_instance_status.jsonl if present.
        # This covers cases where a subprocess failed before writing a trajectory.
        for row in read_jsonl(status_path):
            iid = str(row.get("instance_id") or "").strip()
            if not iid or iid in recorded:
                continue
            exit_status = str(row.get("exit_status") or "").strip()
            if not exit_status:
                reason = str(row.get("reason") or "").strip()
                if reason == "watchdog_timeout":
                    exit_status = "TimeoutExpired"
                elif reason.startswith("nonzero_exit:"):
                    exit_status = reason.replace(":", "_")
                elif str(row.get("status") or "") == "error":
                    exit_status = "subprocess_error"
                else:
                    exit_status = "unknown"
            record_exit_status(agg_statuses, exit_status, iid)
            recorded.add(iid)
        write_exit_statuses_yaml(exit_status_path, agg_statuses, total_cost=agg_cost)

        log_message(f"Running (per_instance): {len(selected)} instances; watchdog={max_minutes} min")
        for inst in selected:
            iid = str(inst.get("instance_id", "")).strip()
            if not iid:
                continue

            traj_path = rep_dir / iid / f"{iid}.traj.json"
            if (not cfg.redo_existing) and traj_path.exists():
                summary = extract_instance_summary(rep_dir, iid) or {}
                exit_status = str(summary.get("exit_status") or "")
                hint = fmt_exit_status_hint(exit_status)
                if hint:
                    log_message(f"Instance={iid} skipped ({hint})")
                else:
                    log_message(f"Instance={iid} skipped (existing; REDO_EXISTING=False)")
                record_exit_status(agg_statuses, exit_status or "skipped_existing", iid)
                try:
                    agg_cost += float(summary.get("instance_cost") or 0.0)
                except (TypeError, ValueError):
                    pass
                write_exit_statuses_yaml(exit_status_path, agg_statuses, total_cost=agg_cost)

                append_jsonl(
                    skipped_path,
                    {
                        "instance_id": iid,
                        "reason": "redo_existing_false",
                        "traj_path": str(traj_path),
                        **summary,
                    },
                )
                append_jsonl(
                    status_path,
                    {
                        "instance_id": iid,
                        "status": "skipped",
                        "reason": "redo_existing_false",
                        "timeout_seconds": timeout_seconds,
                        **summary,
                    },
                )
                continue
            log_message(f"Instance={iid} starting")
            one_path = one_dir / f"{iid}.json"
            one_path.write_text(json.dumps(inst, indent=2), encoding="utf-8")

            started_at = datetime.now(timezone.utc)
            status_row = {
                "instance_id": iid,
                "started_at_utc": started_at.isoformat(),
                "timeout_seconds": timeout_seconds,
            }

            cmd = [sys.executable, str(Path(__file__).resolve()), "--one-instance", str(one_path), str(rep_dir)]
            proc: subprocess.Popen[Any] | None = None
            try:
                proc = subprocess.Popen(cmd)
                start_monotonic = time.monotonic()
                next_tick = start_monotonic
                while True:
                    rc = proc.poll()
                    now = time.monotonic()
                    if rc is not None:
                        res_returncode = int(rc)
                        break
                    if now - start_monotonic >= timeout_seconds:
                        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds)
                    if now >= next_tick:
                        elapsed = int(now - start_monotonic)
                        remaining = max(0, int(timeout_seconds - elapsed))
                        log_message(
                            f"Instance={iid} running; elapsed_s={elapsed} remaining_s={remaining} watchdog_s={timeout_seconds}"
                        )
                        next_tick = now + watchdog_tick_seconds
                    time.sleep(0.5)

                ended_at = datetime.now(timezone.utc)
                status_row.update(
                    {
                        "ended_at_utc": ended_at.isoformat(),
                        "duration_seconds": int((ended_at - started_at).total_seconds()),
                        "returncode": int(res_returncode),
                        "status": "ok" if res_returncode == 0 else "error",
                    }
                )
                if res_returncode != 0:
                    status_row["reason"] = f"nonzero_exit:{res_returncode}"
            except subprocess.TimeoutExpired:
                ended_at = datetime.now(timezone.utc)
                status_row.update(
                    {
                        "ended_at_utc": ended_at.isoformat(),
                        "duration_seconds": int((ended_at - started_at).total_seconds()),
                        "returncode": None,
                        "status": "skipped",
                        "reason": "watchdog_timeout",
                    }
                )
                if proc is not None:
                    forced_kill = False
                    try:
                        # SIGTERM is handled inside the child as an exception,
                        # allowing its finally block to capture model.patch and
                        # serialize the trajectory before exiting.
                        proc.terminate()
                        try:
                            proc.wait(timeout=watchdog_grace_seconds)
                        except subprocess.TimeoutExpired:
                            forced_kill = True
                            proc.kill()
                            proc.wait(timeout=10)
                    finally:
                        status_row["watchdog_forced_kill"] = forced_kill
                append_jsonl(
                    skipped_path,
                    {
                        "instance_id": iid,
                        "reason": "watchdog_timeout",
                        "max_minutes_per_instance": max_minutes,
                        "started_at_utc": started_at.isoformat(),
                        "ended_at_utc": ended_at.isoformat(),
                    },
                )
                if proc is None or status_row.get("watchdog_forced_kill"):
                    img = str(get_swebench_docker_image_name(inst)).strip()
                    remove_minisweagent_containers()
                    stop_and_remove_containers_for_image(img)

            summary = extract_instance_summary(rep_dir, iid) or {}
            status_row.update(summary)
            exit_status = str(summary.get("exit_status") or "").strip()
            if not exit_status:
                reason = str(status_row.get("reason") or "").strip()
                if reason == "watchdog_timeout":
                    exit_status = "TimeoutExpired"
                elif reason.startswith("nonzero_exit:"):
                    exit_status = reason.replace(":", "_")
                elif status_row.get("status") == "error":
                    exit_status = "subprocess_error"
                else:
                    exit_status = "unknown"
            record_exit_status(agg_statuses, exit_status, iid)
            try:
                agg_cost += float(summary.get("instance_cost") or 0.0)
            except (TypeError, ValueError):
                pass
            write_exit_statuses_yaml(exit_status_path, agg_statuses, total_cost=agg_cost)

            append_jsonl(status_path, status_row)
            log_message(
                f"Instance={iid} done: status={status_row['status']} exit_status={status_row.get('exit_status')} returncode={status_row.get('returncode')} duration_s={status_row.get('duration_seconds')} calls={status_row.get('api_calls')} cost={status_row.get('instance_cost')} submission_len={status_row.get('submission_len')}"
            )
            if cfg.delete_docker_image_after_each_instance:
                cleanup_images_for_instances([inst], force=cfg.force_delete_docker_images)

        # Final reconciliation: ensure every selected instance is present.
        recorded = recorded_instance_ids(agg_statuses)
        for inst in selected:
            iid = str(inst.get("instance_id", "")).strip()
            if not iid or iid in recorded:
                continue
            summary = extract_instance_summary(rep_dir, iid) or {}
            exit_status = str(summary.get("exit_status") or "").strip() or "missing"
            record_exit_status(agg_statuses, exit_status, iid)
        write_exit_statuses_yaml(exit_status_path, agg_statuses, total_cost=agg_cost)

    if cfg.delete_docker_images_after_run:
        cleanup_images_for_instances(selected, force=cfg.force_delete_docker_images)

    _write_pro_predictions_file(rep_dir, cfg.model_name)
    log_message(f"Done. Outputs written under: {rep_dir}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
