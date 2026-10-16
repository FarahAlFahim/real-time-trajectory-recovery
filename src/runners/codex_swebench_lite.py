"""src/runners/codex_swebench_lite.py

One-click runner: run the OpenAI Codex CLI as the repair agent on a custom JSON dataset
(SWE-bench-like list of dicts), producing SWE-bench predictions, while storing outputs under
a shared run directory that mirrors ``src/runners/minisweagent_swebench_lite.py``:

    results/<RUN_NAME>/
        codex_original/            # Codex, no trajectory recovery  (CODEX_RECOVERY_MODE="off")
            localization/
            repair/
        codex_realtime_recovery/   # Codex + real-time recovery     (CODEX_RECOVERY_MODE="active")
            localization/
            repair/

Design (why this mirrors the mini-swe-agent pipeline)
-----------------------------------------------------
- Both arms run the same pinned Codex binary with the same model, container, sandbox,
  approval policy, task prompt, and diff capture. The active arm differs only by enabling
  native Codex lifecycle hooks backed by the shared trajectory-recovery policy.
- Codex and its hooks execute inside the SWE-bench container. A small synchronous bridge
  sends hook payloads through a bind-mounted directory to the host-side reviewer, so Codex
  itself and the benchmark repository remain unmodified.

Interposition -> recovery phase mapping
---------------------------------------
- ``PreToolUse`` reviews completed-scope transitions, repetitions, and mutations before execution.
- ``PostToolUse`` records the executed action and its repository evidence.
- ``Stop`` checks the final diff and, when concretely incomplete, continues the same Codex session
  with focused feedback (bounded).

Prerequisites (validate with a Step-0 smoke test before a real run)
-------------------------------------------------------------------
1. A ``codex`` binary on the host at ``CODEX_BIN`` (or on PATH). It is ``docker cp``-ed into
   each container, so a Linux-built binary is required (the container is Linux). Install the
   Python SDK on the host: ``pip install -e <openai-codex-checkout>/sdk/python``.
2. An API key exported on the host (e.g. ``OPENAI_API_KEY``); it is forwarded into the
   container so ``codex app-server`` can reach the model API. The container therefore needs
   network access (so ``--network none`` is NOT used here; git history is stripped instead).
3. IMPORTANT MODEL CONSTRAINT: current Codex only speaks the OpenAI *Responses* API
   (``/v1/responses``); the Chat Completions wire API was removed. ``gpt-5-mini`` via OpenAI
   works natively. Other models (deepseek, minimax, OpenRouter, ...) work ONLY if their
   endpoint exposes ``/v1/responses`` — configure them via MODEL_PROVIDERS with
   ``wire_api="responses"``. A Chat-Completions-only endpoint will fail.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    # Executed by file path (e.g. the per-instance watchdog subprocess): make `src` importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Reuse mini-swe-agent's SWE-bench image-name resolution and progress UI for consistency.
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

from src.runners.common import fmt_exit_status_hint, get_variant_root
from src.utils.io import append_jsonl, ensure_dir, load_json_list, read_jsonl, resolve_path
from src.utils.log import log_message

_REPO_ROOT = Path(__file__).resolve().parents[2]


# =====================================================================================
# CONFIG  (edit here, then run: python run.py codex_lite)
# =====================================================================================
CONFIG: dict[str, Any] = {
    # --- Variant / recovery toggle -----------------------------------------------------
    # VARIANT is just the output-directory name. Convention:
    #   "codex_original"          <-> CODEX_RECOVERY_MODE="off"
    #   "codex_realtime_recovery" <-> CODEX_RECOVERY_MODE="active"
    "VARIANT": "codex_realtime_recovery",
    "CODEX_RECOVERY_MODE": "active",  # "off" | "active"

    # --- Input dataset (SWE-bench-like list of dicts) ---------------------------------
    "DATASET_JSON": "data/swe_bench_lite/psf__requests.json",
    "PROBLEM_FIELD": "problem_statement",
    "REFERENCE_JSON": None,
    "PROJECT_PREFIX": None,
    "LIMIT_INSTANCES": None,

    # --- Output layout ----------------------------------------------------------------
    "RUN_NAME": "psf__requests",
    # Separate root prevents the corrected hook-based experiment from silently reusing or
    # overwriting trajectories produced by the earlier approval-callback prototype.
    "RESULTS_ROOT": "results_trajrec/codex",

    # --- Instance selection -----------------------------------------------------------
    "INSTANCES_FILTER": [
        # "sympy__sympy-14024",
        # "sympy__sympy-17022",
        # "sympy__sympy-19487",
        "psf__requests-863",
        # "psf__requests-3362",
    ],   # allowlist of instance_ids, or a regex string. Empty = all.
    "INSTANCES_SLICE": "",    # python-slice string, e.g. ":50" or "10:20"
    "INSTANCES_SHUFFLE": False,

    # --- Model (fully swappable) ------------------------------------------------------
    # MODEL_NAME + MODEL_PROVIDER select the solver model Codex runs.
    #   - Default OpenAI provider: MODEL_PROVIDER=None, MODEL_NAME="gpt-5-mini".
    #   - Custom/OpenAI-compatible (must speak the Responses API): set MODEL_PROVIDER to a key
    #     in MODEL_PROVIDERS below, e.g. "deepseek" / "minimax".
    "MODEL_NAME": "gpt-5-mini-2025-08-07",
    "MODEL_PROVIDER": None,          # None -> Codex default ("openai"); else a MODEL_PROVIDERS key
    "MODEL_REASONING_EFFORT": None,  # None | "minimal" | "low" | "medium" | "high" | "xhigh"
    # Custom provider tables -> emitted as `--config model_providers.<key>.<field>=...`.
    # env_key names the host env var holding that provider's API key (forwarded into container).
    "MODEL_PROVIDERS": {
        # "deepseek": {
        #     "name": "DeepSeek",
        #     "base_url": "https://api.deepseek.com/v1",
        #     "env_key": "DEEPSEEK_API_KEY",
        #     "wire_api": "responses",
        # },
        # "minimax": {
        #     "name": "MiniMax",
        #     "base_url": "https://api.minimax.io/v1",
        #     "env_key": "MINIMAX_API_KEY",
        #     "wire_api": "responses",
        # },
    },
    # API-key env vars to forward from the host into the container (for the default provider
    # and/or any custom provider env_key). Only vars actually set on the host are forwarded.
    "FORWARD_ENV_KEYS": ["OPENAI_API_KEY"],

    # --- Codex runtime ----------------------------------------------------------------
    # Host path to a Linux `codex` binary that gets docker-cp'ed into each container.
    # MUST be a Linux x86_64 ELF (the SWE-bench container is linux/amd64, emulated on Apple
    # Silicon). Do NOT leave empty on macOS: the SDK's bundled binary is a macOS Mach-O and
    # fails inside the container with "exec format error". Extracted from the matching wheel
    # (openai-codex-cli-bin==0.153.4, manylinux_2_17_x86_64) — verified `codex-cli 0.153.4`.
    "CODEX_BIN": str(_REPO_ROOT / "bin" / "codex-linux-x86_64"),
    "CODEX_CONTAINER_BIN": "/usr/local/bin/codex",
    # read-only | workspace-write | danger-full-access.
    # MUST be danger-full-access here: Codex's workspace-write/read-only sandboxes run via
    # bubblewrap (bwrap), which is NOT present in the container (we only docker-cp the codex
    # binary, not its codex-resources/bwrap sidecar). Without bwrap, `apply_patch` and sandboxed
    # commands fail with "bubblewrap is unavailable" and edits never land (empty patches).
    # danger-full-access makes Codex run unsandboxed (SandboxType::None) — safe here because the
    # Docker container is already the isolation boundary.
    "SANDBOX_MODE": "danger-full-access",
    # Keep the benchmark checkout intact. Re-initializing git caused most historical empty
    # patches by leaving repositories without a valid baseline when the synthetic commit failed.
    "STRIP_GIT_HISTORY": False,
    # Disabled for a scientifically clean stock-Codex baseline. Empty model outputs remain valid
    # failures; infrastructure still fails fast if the immutable git baseline cannot be captured.
    "DELIVERY_MAX_RETRIES": 0,
    "ENABLE_PATCH_SALVAGE": False,

    # --- Real-time recovery (active mode) ---------------------------------------------
    # Independent reviewer (LiteLLM). Leave None to reuse MODEL_NAME: it is auto-derived to a
    # LiteLLM name from MODEL_NAME + MODEL_PROVIDER (e.g. "gpt-5-mini-2025-08-07" -> reviewer
    # "openai/gpt-5-mini-2025-08-07"). Set an explicit LiteLLM name here only to override.
    "RECOVERY_MODEL_NAME": None,
    "RECOVERY_MODEL_TIMEOUT_SECONDS": 180,
    "RECOVERY_MODEL_KWARGS": {},
    "RECOVERY_CONTEXT_TOKEN_LIMIT": 250_000,
    "RECOVERY_MAX_SUBMISSION_BLOCKS": 2,
    "RECOVERY_MAX_ACTION_BLOCKS": 3,
    "RECOVERY_REPEAT_THRESHOLD": 2,
    "RECOVERY_FAIL_OPEN": True,
    "RECOVERY_REVIEW_BRANCHES": True,
    "RECOVERY_GATE_MUTATIONS": True,    # gate patch/file-change approvals (Edit Consistency)
    "RECOVERY_GATE_SUBMISSION": True,   # review the final diff before accepting it (Completion)
    # Must exceed RECOVERY_MODEL_TIMEOUT_SECONDS because Codex waits synchronously for each hook.
    "RECOVERY_HOOK_TIMEOUT_SECONDS": 210,

    # --- Cost tracking ----------------------------------------------------------------
    # LiteLLM price-lookup mode (used for reviewer cost AND solver-token pricing):
    #   "default": error if a model price is missing; "ignore_errors": continue (cost -> None).
    "MSWEA_COST_TRACKING": "ignore_errors",

    # --- Execution --------------------------------------------------------------------
    "RUN_MODE": "per_instance",         # "per_instance" (subprocess+watchdog) | "batch"
    "MAX_MINUTES_PER_INSTANCE": 30,
    "WORKERS": 1,
    "WATCHDOG_TICK_SECONDS": 10,
    "REDO_EXISTING": False,
    "RAISE_EXCEPTIONS": False,
    "PULL_DOCKER_IMAGES_BEFORE_RUN": False,
    "DELETE_DOCKER_IMAGE_AFTER_EACH_INSTANCE": False,
    "DELETE_DOCKER_IMAGES_AFTER_RUN": False,
    "FORCE_DELETE_DOCKER_IMAGES": False,
    "DRY_RUN": False,
}


# =====================================================================================
# Small utilities (mirrors src/runners/minisweagent_swebench_lite.py)
# =====================================================================================
def _format_duration(seconds: int | float) -> str:
    """Format terminal-only elapsed time without changing stored measurements."""
    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes}m {remaining_seconds}s" if minutes else f"{remaining_seconds}s"


def _read_exit_statuses_yaml(path: Path) -> tuple[dict[str, list[str]], float | None]:
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
            current_section, current_status = "instances_by_exit_status", None
            continue
        if line.startswith("total_cost:"):
            current_section = current_status = None
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


def _write_exit_statuses_yaml(path: Path, statuses: dict[str, list[str]], *, total_cost: float | None) -> None:
    lines: list[str] = ["instances_by_exit_status:"]
    for status in sorted(statuses.keys()):
        lines.append(f"  {status}:")
        for iid in statuses[status]:
            lines.append(f"    - {iid}")
    if total_cost is not None:
        lines.append(f"total_cost: {total_cost:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record_exit_status(agg: dict[str, list[str]], exit_status: str, iid: str) -> None:
    status = str(exit_status or "").strip() or "unknown"
    ids = agg.setdefault(status, [])
    if iid not in ids:
        ids.append(iid)


def _recorded_instance_ids(agg: dict[str, list[str]]) -> set[str]:
    return {iid for ids in agg.values() for iid in ids}


def _normalize_problem_statement(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ["Title", "Description", "RootCause", "StepsToReproduce",
                    "ExpectedBehavior", "ObservedBehavior", "Suggestions"]:
            if key not in value:
                continue
            val = value[key]
            if isinstance(val, list):
                val = "\n".join(f"- {x}" for x in val)
            elif isinstance(val, dict):
                val = "\n".join(f"{sk}: {sv}" for sk, sv in val.items())
            parts.append(f"{key}:\n{val}")
        return "\n\n".join(parts) if parts else json.dumps(value, ensure_ascii=False)
    return str(value)


# =====================================================================================
# Runner configuration
# =====================================================================================
@dataclass(frozen=True)
class RunnerConfig:
    variant: str
    recovery_mode: str
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
    model_name: str
    model_provider: str | None
    model_reasoning_effort: str | None
    model_providers: dict[str, dict[str, str]]
    forward_env_keys: list[str]
    codex_bin: str
    codex_container_bin: str
    sandbox_mode: str
    strip_git_history: bool
    delivery_max_retries: int
    enable_patch_salvage: bool
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
    recovery_hook_timeout_seconds: int
    run_mode: str
    max_minutes_per_instance: int
    workers: int
    watchdog_tick_seconds: int
    redo_existing: bool
    raise_exceptions: bool
    delete_docker_image_after_each_instance: bool
    delete_docker_images_after_run: bool
    force_delete_docker_images: bool
    pull_docker_images_before_run: bool


def _build_runner_config() -> RunnerConfig:
    dataset_json = resolve_path(CONFIG.get("DATASET_JSON"), base_dir=_REPO_ROOT)
    if dataset_json is None:
        raise FileNotFoundError("DATASET_JSON is empty")
    results_root = resolve_path(CONFIG.get("RESULTS_ROOT"), base_dir=_REPO_ROOT) or (_REPO_ROOT / "results")
    return RunnerConfig(
        variant=str(CONFIG.get("VARIANT")),
        recovery_mode=str(CONFIG.get("CODEX_RECOVERY_MODE", "off")).strip().lower(),
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
        model_name=str(CONFIG.get("MODEL_NAME")),
        model_provider=CONFIG.get("MODEL_PROVIDER"),
        model_reasoning_effort=CONFIG.get("MODEL_REASONING_EFFORT"),
        model_providers=dict(CONFIG.get("MODEL_PROVIDERS") or {}),
        forward_env_keys=list(CONFIG.get("FORWARD_ENV_KEYS") or []),
        codex_bin=str(CONFIG.get("CODEX_BIN") or ""),
        codex_container_bin=str(CONFIG.get("CODEX_CONTAINER_BIN") or "/usr/local/bin/codex"),
        sandbox_mode=str(CONFIG.get("SANDBOX_MODE") or "workspace-write"),
        strip_git_history=bool(CONFIG.get("STRIP_GIT_HISTORY", False)),
        delivery_max_retries=int(CONFIG.get("DELIVERY_MAX_RETRIES", 0)),
        enable_patch_salvage=bool(CONFIG.get("ENABLE_PATCH_SALVAGE", False)),
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
        recovery_hook_timeout_seconds=int(CONFIG.get("RECOVERY_HOOK_TIMEOUT_SECONDS", 210)),
        run_mode=str(CONFIG.get("RUN_MODE", "per_instance")),
        max_minutes_per_instance=int(CONFIG.get("MAX_MINUTES_PER_INSTANCE", 30)),
        workers=int(CONFIG.get("WORKERS", 1)),
        watchdog_tick_seconds=int(CONFIG.get("WATCHDOG_TICK_SECONDS", 10)),
        redo_existing=bool(CONFIG.get("REDO_EXISTING", False)),
        raise_exceptions=bool(CONFIG.get("RAISE_EXCEPTIONS", False)),
        delete_docker_image_after_each_instance=bool(CONFIG.get("DELETE_DOCKER_IMAGE_AFTER_EACH_INSTANCE", False)),
        delete_docker_images_after_run=bool(CONFIG.get("DELETE_DOCKER_IMAGES_AFTER_RUN", False)),
        force_delete_docker_images=bool(CONFIG.get("FORCE_DELETE_DOCKER_IMAGES", False)),
        pull_docker_images_before_run=bool(CONFIG.get("PULL_DOCKER_IMAGES_BEFORE_RUN", False)),
    )


# TASK-SCOPE framing ONLY — the fair analog of the task-specific parts of mini-swe-agent's
# swebench.yaml instance_template (fix non-test source in /testbed, general/consistent fix,
# don't commit). It deliberately OMITS all "how to work" mechanics (workflow, verify-git-diff,
# "use apply_patch", "don't print the patch", submission steps): Codex already ships those in its
# built-in system prompt (codex-rs .../base_instructions/default.md — persistence, "actually
# implement the change", "use the apply_patch tool"), just as mini-swe-agent keeps its mechanics
# in the yaml. Replicating them here would double up unfairly AND pre-solve the very failures the
# recovery arm must catch, confounding codex_original vs codex_realtime_recovery.
_REPAIR_INSTRUCTIONS = (
    "\n\n---\n"
    "The repository affected by the issue above is checked out at /testbed, which is your working "
    "directory. Make changes to the non-test source files in /testbed to fix the issue in a way "
    "that is general and consistent with the codebase. Do not modify tests. Do not create a git "
    "commit; leave your changes in the working tree.\n"
    # Delivery-completeness nudge (applied uniformly to BOTH arms): guarantees an evaluable
    # artifact — a non-empty diff — without giving any correctness/how-to-fix guidance. This is
    # more reproducible than re-running until a patch appears (which biases results). It does not
    # help with WHICH fix is right, so it leaves direction/edit/completeness for recovery to earn.
    "Actually apply your fix by editing the files — do not reply with a patch or a description "
    "instead of editing. Before you finish, run `git diff` in /testbed to confirm your change is "
    "present; if it shows nothing, your edit did not apply, so correct it before ending."
)


def _build_task_prompt(task: str) -> str:
    """Wrap the raw problem statement with task-scope framing only (see note above)."""
    return task + _REPAIR_INSTRUCTIONS


# Delivery-floor feedback, sent ONLY when a turn ends with an empty `git diff`. gpt-5-mini
# frequently writes brittle string-match edits that silently no-op (the match text isn't
# byte-exact) and then reports success. This pushes it to re-apply via the robust apply_patch
# command (fuzzy context) and verify a non-empty diff. It gives DELIVERY mechanics only — no hint
# about WHAT the fix is — so it guarantees an evaluable artifact without helping correctness.
_DELIVERY_RETRY_FEEDBACK = (
    "STOP — your changes are NOT in the working tree: `git diff` in /testbed is EMPTY, so your "
    "previous edits did not modify any file (the exact text you tried to match probably did not "
    "exist verbatim). Do NOT just describe the fix again. Re-apply it now using the apply_patch "
    "command, which tolerates surrounding context, e.g.:\n"
    "apply_patch <<'EOF'\n"
    "*** Begin Patch\n"
    "*** Update File: path/to/file.py\n"
    "@@\n"
    "-a line that actually exists in the file\n"
    "+your replacement line\n"
    "*** End Patch\n"
    "EOF\n"
    "Then run `git diff` in /testbed and confirm it is non-empty before you finish."
)


def _reviewer_model_name(cfg: RunnerConfig) -> str:
    """LiteLLM name for the recovery reviewer.

    The Codex solver model (``MODEL_NAME``) has no provider prefix, but the reviewer runs
    through LiteLLM, which needs one. Reuse the solver model by default (set once), deriving
    the LiteLLM name from MODEL_PROVIDER (or "openai" for the default provider). An explicit
    ``RECOVERY_MODEL_NAME`` overrides this derivation.
    """
    if cfg.recovery_model_name:
        return cfg.recovery_model_name
    provider = cfg.model_provider or "openai"
    return f"{provider}/{cfg.model_name}"


# ---------------------------------------------------------------------------------------
# Token & cost accounting (parity with src/runners/minisweagent_swebench_lite.py model_stats)
# ---------------------------------------------------------------------------------------
def _serialize_thread_item(item: Any) -> dict[str, Any]:
    """Dump a Codex ThreadItem (reasoning / command / file-change / agent message) to JSON."""
    obj = getattr(item, "root", item)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json", by_alias=True)
        except Exception:
            pass
    return {"repr": str(obj)}


def _usage_breakdown(usage: Any) -> dict[str, int]:
    """Flatten a Codex ThreadTokenUsage (its cumulative ``.total``) to plain ints."""
    total = getattr(usage, "total", None)
    if total is None:
        return {}
    return {
        "input_tokens": int(getattr(total, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(total, "output_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(total, "cached_input_tokens", 0) or 0),
        "cache_write_input_tokens": int(getattr(total, "cache_write_input_tokens", 0) or 0),
        "reasoning_output_tokens": int(getattr(total, "reasoning_output_tokens", 0) or 0),
        "total_tokens": int(getattr(total, "total_tokens", 0) or 0),
    }


def _estimate_cost(litellm_model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Best-effort USD cost from token counts via LiteLLM pricing (fail-open to None).

    Codex reports tokens but not a dollar cost, so we price them ourselves using the same
    LiteLLM model map the recovery reviewer uses. Cached-token discounts are ignored (slight
    overestimate); MSWEA_COST_TRACKING=ignore_errors keeps a missing price from raising.
    """
    if not (input_tokens or output_tokens):
        return 0.0
    try:
        import litellm  # noqa: PLC0415

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=litellm_model, prompt_tokens=int(input_tokens), completion_tokens=int(output_tokens)
        )
        return float(prompt_cost) + float(completion_cost)
    except Exception:
        return None


# =====================================================================================
# Instance loading / filtering
# =====================================================================================
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
        base["problem_statement"] = _normalize_problem_statement(
            r.get(cfg.problem_field, r.get("problem_statement", ""))
        )
        records.append(base)

    if cfg.project_prefix:
        records = [r for r in records if str(r.get("instance_id", "")).startswith(cfg.project_prefix)]
    if cfg.limit_instances is not None:
        records = records[: int(cfg.limit_instances)]

    instances: list[dict[str, Any]] = []
    for r in records:
        iid = str(r.get("instance_id", "")).strip()
        if not iid:
            continue
        image_name = get_swebench_docker_image_name({"instance_id": iid, "image_name": r.get("image_name")})
        instances.append({
            "instance_id": iid,
            "problem_statement": str(r.get("problem_statement", "")),
            "image_name": image_name,
            **{k: v for k, v in r.items() if k not in {"problem_statement"}},
        })
    log_message(f"Variant={cfg.variant} records ready: {len(instances)}")
    return instances


def _apply_instances_filter(instances: list[dict[str, Any]], cfg: RunnerConfig) -> list[dict[str, Any]]:
    fv = cfg.instances_filter
    if isinstance(fv, list):
        allow = {str(x).strip() for x in fv if str(x).strip()}
        out = list(instances) if not allow else [i for i in instances if str(i.get("instance_id", "")) in allow]
    else:
        pat = str(fv or ".*").strip() or ".*"
        rx = re.compile(pat)
        out = [i for i in instances if rx.search(str(i.get("instance_id", "")))]
    if cfg.instances_slice:
        parts = cfg.instances_slice.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid INSTANCES_SLICE: {cfg.instances_slice!r}")

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


# =====================================================================================
# Docker helpers
# =====================================================================================
def _docker(*args: str, check: bool = False, capture: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        timeout=timeout,
    )


def _docker_exec_capture(container: str, command: str, *, workdir: str | None = None, timeout: int | None = None) -> tuple[int, str]:
    args = ["exec"]
    if workdir:
        args += ["-w", workdir]
    args += [container, "bash", "-lc", command]
    cp = _docker(*args, capture=True, timeout=timeout)
    return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


def _resolve_host_codex_bin(cfg: RunnerConfig) -> str:
    if cfg.codex_bin and Path(cfg.codex_bin).expanduser().exists():
        return str(Path(cfg.codex_bin).expanduser())
    # Fall back to the binary bundled with the installed SDK, if present.
    try:
        from codex_cli_bin import bundled_codex_path  # type: ignore

        p = Path(bundled_codex_path())
        if p.exists():
            return str(p)
    except Exception:
        pass
    raise FileNotFoundError(
        "No codex binary found. Set CONFIG['CODEX_BIN'] to a Linux `codex` binary "
        "(it is docker-cp'ed into the container), or install the SDK so codex_cli_bin resolves."
    )


def _install_recovery_hooks(
    cfg: RunnerConfig,
    container: str,
    ipc_root: Path,
) -> None:
    """Install user-level native hooks without modifying the benchmark checkout."""
    bridge = _REPO_ROOT / "src" / "adapters" / "codex" / "hook_bridge.py"
    if not bridge.exists():
        raise FileNotFoundError(f"Codex recovery hook bridge not found: {bridge}")
    # The Codex-side command timeout must outlive the bridge wait; otherwise Codex can kill the
    # hook at the exact moment the host reviewer writes its response.
    timeout = max(cfg.recovery_hook_timeout_seconds, cfg.recovery_model_timeout_seconds + 15) + 15
    command = "/usr/bin/env python3 /opt/trajrec/codex_recovery_hook_bridge.py"
    handler = {"type": "command", "command": command, "timeout": timeout}
    hooks = {
        "description": "TrajRec native Codex hook adapter",
        "hooks": {
            # A synchronous startup hook provides a deterministic handshake. Active runs refuse
            # to spend a solver turn unless the host controller observes this event.
            "SessionStart": [{"hooks": [handler]}],
            "PreToolUse": [{"matcher": "^(Bash|apply_patch)$", "hooks": [handler]}],
            "PostToolUse": [{"matcher": "^(Bash|apply_patch)$", "hooks": [handler]}],
            "Stop": [{"hooks": [handler]}],
        },
    }
    hook_file = ipc_root / "hooks.json"
    hook_file.write_text(json.dumps(hooks, indent=2), encoding="utf-8")
    rc, out = _docker_exec_capture(
        container,
        "mkdir -p /root/.codex /opt/trajrec",
    )
    if rc != 0:
        raise RuntimeError(f"Unable to prepare Codex hook directories: {out}")
    for source, destination in (
        (bridge, "/opt/trajrec/codex_recovery_hook_bridge.py"),
        (hook_file, "/root/.codex/hooks.json"),
    ):
        copied = _docker("cp", str(source), f"{container}:{destination}", capture=True)
        if copied.returncode != 0:
            raise RuntimeError(f"Unable to install Codex recovery hook {source.name}: {copied.stderr}")


def _start_container(
    cfg: RunnerConfig,
    inst: dict[str, Any],
    name: str,
    *,
    recovery_ipc_root: Path | None = None,
) -> None:
    image = str(inst["image_name"])
    _docker("rm", "-f", name, capture=True)
    # Keep-alive container; network stays UP (Codex must reach the model API).
    run_args = ["run", "-d", "--name", name]
    if recovery_ipc_root is not None:
        ensure_dir(recovery_ipc_root)
        run_args += ["-v", f"{recovery_ipc_root}:/trajrec_ipc"]
    run_args += [image, "sleep", "infinity"]
    cp = _docker(*run_args, capture=True)
    if cp.returncode != 0:
        raise RuntimeError(f"docker run failed for {image}: {cp.stderr}")
    # Install the codex binary.
    host_bin = _resolve_host_codex_bin(cfg)
    cp = _docker("cp", host_bin, f"{name}:{cfg.codex_container_bin}", capture=True)
    if cp.returncode != 0:
        raise RuntimeError(f"docker cp codex failed: {cp.stderr}")
    _docker_exec_capture(name, f"chmod +x {cfg.codex_container_bin}")
    # Install the patch-salvage helper (used to apply the model's intended patch ourselves when
    # the turn ends with an empty diff — see _salvage_empty_patch).
    salvage_src = _REPO_ROOT / "src" / "adapters" / "codex" / "apply_patch_salvage.py"
    if salvage_src.exists():
        _docker("cp", str(salvage_src), f"{name}:{_SALVAGE_CONTAINER_SCRIPT}", capture=True)
    # Mac/Docker fix (same as src/evaluation/swebench_lite.py): under Rosetta/qemu emulation the
    # repo can trip git's "dubious ownership" guard, which would break base-commit and diff
    # capture (-> empty patches). Mark it safe globally before any git op in this container.
    _docker_exec_capture(
        name,
        "git config --global --add safe.directory /testbed; git config --global --add safe.directory '*'",
    )
    # Authenticate Codex (writes auth.json from the host API key; env var alone -> 401).
    _codex_login(cfg, name)
    if recovery_ipc_root is not None:
        _install_recovery_hooks(cfg, name, recovery_ipc_root)
    # Optionally strip git history to a single base commit (anti-leakage).
    if cfg.strip_git_history:
        # Re-init to a single base commit. The identity MUST be inline (-c ...): setting it via
        # `git config` before `rm -rf .git` writes to the old .git that we then delete, so the
        # commit would fail for lack of a user identity -> unborn HEAD -> no base commit -> every
        # `git diff <base>` fails and all patches come back empty. (SWE-bench images don't ship a
        # global git identity.) Also don't swallow the failure.
        rc, out = _docker_exec_capture(
            name,
            "cd /testbed && rm -rf .git && git init -q && git add -A && "
            "git -c user.email=codex@local -c user.name=codex commit -q -m base",
        )
        if rc != 0:
            log_message(f"WARNING: strip_git_history base commit failed (rc={rc}): {out.strip()[:200]}")


def _base_commit(container: str) -> str | None:
    rc, out = _docker_exec_capture(container, "cd /testbed && git rev-parse --verify HEAD 2>/dev/null")
    if rc != 0:
        return None
    rev = out.strip().splitlines()[-1:] if out.strip() else []
    return rev[0] if rev and re.fullmatch(r"[0-9a-fA-F]{7,64}", rev[0]) else None


def _list_untracked_files(container: str) -> frozenset[str]:
    """Return untracked, non-ignored repository files without changing the index."""
    rc, out = _docker_exec_capture(
        container,
        "cd /testbed && git ls-files --others --exclude-standard -z 2>/dev/null",
    )
    if rc != 0:
        return frozenset()
    return frozenset(path for path in out.split("\0") if path)


def _exclude_baseline_untracked(diff: str, baseline_untracked: frozenset[str]) -> str:
    """Remove diff sections for files that were already untracked at startup."""
    if not diff or not baseline_untracked:
        return diff
    retained: list[str] = []
    for section in re.split(r"(?m)(?=^diff --git )", diff):
        if not section.startswith("diff --git "):
            retained.append(section)
            continue
        header = section.splitlines()[0]
        try:
            tokens = shlex.split(header)
        except ValueError:
            retained.append(section)
            continue
        paths = {
            token[2:]
            for token in tokens[2:4]
            if token.startswith(("a/", "b/"))
        }
        if paths & baseline_untracked:
            continue
        retained.append(section)
    return "".join(retained)


def _capture_diff(
    container: str,
    base: str | None,
    baseline_untracked: frozenset[str] | None = None,
) -> str:
    baseline = base or "HEAD"
    # Include files created by the solver, but never image-provided files that were already
    # untracked before Codex started. This prevents generated trees such as build/ from
    # contaminating both reviewer context and the submitted patch.
    current_untracked = _list_untracked_files(container)
    new_untracked = sorted(current_untracked - (baseline_untracked or frozenset()))
    for start in range(0, len(new_untracked), 100):
        paths = " ".join(shlex.quote(path) for path in new_untracked[start : start + 100])
        _docker_exec_capture(
            container,
            f"cd /testbed && rm -f .git/index.lock 2>/dev/null; git add -N -- {paths} "
            ">/dev/null 2>&1",
        )
    rc, out = _docker_exec_capture(
        container,
        f"cd /testbed && rm -f .git/index.lock 2>/dev/null; "
        f"git diff --no-color {baseline} -- . ':(exclude)patch.txt' 2>/dev/null",
    )
    if rc == 0 and out.strip():
        return _exclude_baseline_untracked(
            out,
            baseline_untracked or frozenset(),
        )
    # Fallback for an index problem still captures tracked modifications.
    rc2, out2 = _docker_exec_capture(
        container,
        f"cd /testbed && git diff --no-color {baseline} -- . ':(exclude)patch.txt' 2>/dev/null",
    )
    if rc2 == 0 and out2.strip():
        return _exclude_baseline_untracked(
            out2,
            baseline_untracked or frozenset(),
        )
    return out if rc == 0 else ""


def _codex_login(cfg: "RunnerConfig", name: str) -> None:
    """Create Codex auth.json in the container from the host API key (default provider).

    Codex ignores the OPENAI_API_KEY env var for the default 'openai' provider and requires
    stored auth: `codex login --with-api-key` reads the key from stdin. Custom providers read
    their env_key directly from the forwarded env, so login is only needed for the default one.
    """
    if cfg.model_provider:
        return
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        log_message("WARNING: OPENAI_API_KEY not set on host; Codex will fail to authenticate (401).")
        return
    try:
        cp = subprocess.run(
            ["docker", "exec", "-i", "-u", "root", name, cfg.codex_container_bin, "login", "--with-api-key"],
            input=key, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        )
        if cp.returncode != 0:
            log_message(f"WARNING: codex login failed (rc={cp.returncode}): {(cp.stderr or '').strip()[:200]}")
    except Exception as exc:
        log_message(f"WARNING: codex login error: {type(exc).__name__}: {exc}")


def _stop_container(name: str) -> None:
    _docker("rm", "-f", name, capture=True)


# --- Patch salvage (extract the model's intended patch from the transcript and apply it) ------
_APPLY_PATCH_BLOCK_RE = re.compile(r"\*\*\* Begin Patch.*?\*\*\* End Patch", re.DOTALL)
_UNIFIED_DIFF_RE = re.compile(
    r"(--- [^\n]+\n\+\+\+ [^\n]+\n(?:@@[^\n]*@@[^\n]*\n(?:[ +\-\\\\][^\n]*\n?)+)+)",
    re.DOTALL,
)
_SALVAGE_CONTAINER_SCRIPT = "/tmp/apply_patch_salvage.py"


def _unescape_double_quoted(s: str) -> str:
    """Undo bash double-quote escaping (\\" \\$ \\` \\\\) left in patch text the model emitted
    inside a `bash -lc "..."` wrapper (e.g. `assert \\".\\" not in name`)."""
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] in '"$`\\':
            out.append(s[i + 1])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _resolve_str_node(node: ast.AST, env: dict[str, str]) -> str | None:
    """Resolve an AST expression to a string using known string vars + literal concatenation."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _resolve_str_node(node.left, env)
        right = _resolve_str_node(node.right, env)
        if left is not None and right is not None:
            return left + right
    return None


def _extract_python_edits(cmd: str) -> list[tuple[str, str, str]]:
    """Recover (path, old, new) replacements from python-heredoc edits the model runs.

    gpt-5-mini frequently edits via `python - <<PY ... old=...; new=...; s.replace(old,new) ... PY`.
    When `old` doesn't match byte-exactly the replace silently no-ops. We AST-parse the heredoc,
    resolve the `old`/`new` strings and the target path, and return them so the caller can apply
    the intended change with fuzzy matching.
    """
    edits: list[tuple[str, str, str]] = []
    for m in re.finditer(r"<<\s*'?([A-Za-z_][A-Za-z0-9_]*)'?\s*\n(.*?)\n\1\b", cmd, re.DOTALL):
        body = m.group(2)
        try:
            tree = ast.parse(body)
        except SyntaxError:
            continue
        env: dict[str, str] = {}
        path: str | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                val = _resolve_str_node(node.value, env)
                if val is not None:
                    env[node.targets[0].id] = val
                if isinstance(node.value, ast.Call):
                    fn = node.value.func
                    fname = getattr(fn, "id", None) or getattr(fn, "attr", None)
                    if fname in {"Path", "open"} and node.value.args:
                        a0 = node.value.args[0]
                        if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                            path = a0.value
        old, new = env.get("old"), env.get("new")
        if path and old and new and old != new:
            edits.append((path, old, new))
    return edits


def _edit_to_apply_patch(path: str, old: str, new: str) -> str:
    """Synthesize an apply_patch envelope from an (old -> new) replacement for the fuzzy applier."""
    old_lines = [ln for ln in old.split("\n") if ln != ""]
    new_lines = [ln for ln in new.split("\n") if ln != ""]
    lines = ["*** Begin Patch", f"*** Update File: {path}", "@@"]
    lines += [f"-{ln}" for ln in old_lines]
    lines += [f"+{ln}" for ln in new_lines]
    lines.append("*** End Patch")
    return "\n".join(lines)


def _extract_patch_payloads(transcript: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Pull apply_patch envelopes, unified diffs, and python-edit intents out of the commands."""
    apply_blocks: list[str] = []
    diffs: list[str] = []
    for item in transcript:
        if item.get("type") != "commandExecution":
            continue
        cmd = item.get("command")
        cmd = " ".join(map(str, cmd)) if isinstance(cmd, list) else str(cmd or "")
        cmd = _unescape_double_quoted(cmd)
        for m in _APPLY_PATCH_BLOCK_RE.finditer(cmd):
            block = m.group(0)
            if block not in apply_blocks:
                apply_blocks.append(block)
        for m in _UNIFIED_DIFF_RE.finditer(cmd):
            diff = m.group(1)
            if "*** Begin Patch" not in diff and diff not in diffs:
                diffs.append(diff)
        for path, old, new in _extract_python_edits(cmd):
            block = _edit_to_apply_patch(path, old, new)
            if block not in apply_blocks:
                apply_blocks.append(block)
    return apply_blocks, diffs


def _ensure_salvage_script(container: str) -> bool:
    """Write the CURRENT salvage script into the container (independent of cp-at-startup)."""
    src = _REPO_ROOT / "src" / "adapters" / "codex" / "apply_patch_salvage.py"
    if not src.exists():
        return False
    try:
        with src.open("rb") as f:
            cp = subprocess.run(
                ["docker", "exec", "-i", container, "tee", _SALVAGE_CONTAINER_SCRIPT],
                stdin=f, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            )
        return cp.returncode == 0
    except Exception:
        return False


def _salvage_empty_patch(
    container: str,
    base: str | None,
    transcript: list[dict[str, Any]],
    baseline_untracked: frozenset[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Apply the model's intended patch ourselves when the working tree ended up empty.

    gpt-5-mini often emits a correct patch but applies it with the wrong tool (git apply on
    apply_patch format) or a no-op script. We extract its intended patch from the transcript and
    apply it deterministically. Returns (method, debug): method is "" if nothing produced a diff.
    """
    apply_blocks, diffs = _extract_patch_payloads(transcript)
    debug: dict[str, Any] = {
        "apply_blocks": len(apply_blocks),
        "diffs": len(diffs),
        "script_ok": False,
        "attempts": [],
    }
    if not apply_blocks and not diffs:
        return "", debug
    # Write the current salvage script fresh (don't depend on the cp at container start).
    debug["script_ok"] = _ensure_salvage_script(container)
    # Restore the working tree to the exact revision the model's patch was written against, so
    # leftover no-op/partial edits from the solver's turns can't block a clean apply.
    baseline = base or "HEAD"
    subprocess.run(
        ["docker", "exec", "-w", "/testbed", container, "bash", "-lc",
         f"git checkout -q {baseline} -- . 2>/dev/null; git reset -q 2>/dev/null || true"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
    )
    # apply_patch envelopes first (the dominant form), newest attempt first.
    for i, block in enumerate(reversed(apply_blocks)):
        cp = subprocess.run(
            ["docker", "exec", "-i", "-w", "/testbed", container, "python3", _SALVAGE_CONTAINER_SCRIPT],
            input=block, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        )
        nonempty = bool(_capture_diff(container, base, baseline_untracked).strip())
        attempt = {
            "kind": "apply_patch", "block": i, "rc": cp.returncode,
            "out": (cp.stdout or cp.stderr or "").strip()[:200], "nonempty": nonempty,
        }
        if not nonempty:
            _, st = _docker_exec_capture(
                container,
                "cd /testbed && git status --short 2>&1 | head -5; echo '--DIFFSTAT--'; "
                "git diff --stat 2>&1 | tail -3; echo '--HEAD--'; git rev-parse HEAD 2>&1",
            )
            attempt["state"] = st.strip()[:400]
        debug["attempts"].append(attempt)
        if nonempty:
            return "apply_patch", debug
    # Unified diffs via git apply (then patch --fuzz as fallback).
    for i, diff in enumerate(reversed(diffs)):
        payload = diff if diff.endswith("\n") else diff + "\n"
        for shell in ("git apply --recount -", "patch -p1 --fuzz=5"):
            cp = subprocess.run(
                ["docker", "exec", "-i", "-w", "/testbed", container, "bash", "-lc", shell],
                input=payload, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
            )
            nonempty = bool(_capture_diff(container, base, baseline_untracked).strip())
            debug["attempts"].append({
                "kind": "unified_diff", "diff": i, "shell": shell.split()[0],
                "rc": cp.returncode, "out": (cp.stderr or "").strip()[:200], "nonempty": nonempty,
            })
            if nonempty:
                return "unified_diff", debug
    return "", debug


# --- Docker image/container cleanup (parity with src/runners/minisweagent_swebench_lite.py) ---
def _docker_stop_and_remove_containers_for_image(image: str) -> None:
    img = str(image).strip()
    if not img:
        return
    cp = _docker("ps", "-a", "-q", "--filter", f"ancestor={img}", capture=True)
    for cid in [x for x in (cp.stdout or "").splitlines() if x.strip()]:
        _docker("rm", "-f", cid, capture=True)


def _docker_prune() -> None:
    _docker("system", "prune", "-f", capture=True)


def _docker_image_rm(image: str, *, force: bool) -> None:
    img = str(image).strip()
    if not img:
        return
    args = ["image", "rm"] + (["-f"] if force else []) + [img]
    _docker(*args, capture=True)


def _cleanup_images_for_instances(instances: list[dict[str, Any]], *, force: bool) -> None:
    images: list[str] = []
    seen: set[str] = set()
    for inst in instances:
        img = str(inst.get("image_name") or get_swebench_docker_image_name(inst)).strip()
        if img and img not in seen:
            seen.add(img)
            images.append(img)
    if not images:
        return
    log_message(f"Deleting {len(images)} docker images (force={force})")
    for img in images:
        _docker_stop_and_remove_containers_for_image(img)
        _docker_image_rm(img, force=force)
    _docker_prune()


def _pull_docker_images(instances: list[dict[str, Any]]) -> None:
    images: list[str] = []
    seen: set[str] = set()
    for inst in instances:
        img = str(inst.get("image_name") or get_swebench_docker_image_name(inst)).strip()
        if img and img not in seen:
            seen.add(img)
            images.append(img)
    if not images:
        return
    log_message(f"Pre-pulling {len(images)} docker images")
    for img in images:
        _docker("pull", img, capture=True)


# =====================================================================================
# Codex launch (app-server inside the container via docker exec)
# =====================================================================================
def _codex_config_kv(cfg: RunnerConfig, *, approval_policy: str) -> list[str]:
    """Build `--config key=value` pairs for the codex process."""
    kv: list[str] = [
        f"model={cfg.model_name}",
        f"approval_policy={approval_policy}",
        f"sandbox_mode={cfg.sandbox_mode}",
    ]
    if cfg.recovery_mode == "active":
        kv.append("features.hooks=true")
    if cfg.model_provider:
        kv.append(f"model_provider={cfg.model_provider}")
    if cfg.model_reasoning_effort:
        kv.append(f"model_reasoning_effort={cfg.model_reasoning_effort}")
    for prov_key, table in cfg.model_providers.items():
        for field_name, value in table.items():
            kv.append(f"model_providers.{prov_key}.{field_name}={value}")
    flags: list[str] = []
    for pair in kv:
        flags += ["--config", pair]
    return flags


def _thread_start_config(cfg: RunnerConfig) -> dict[str, Any] | None:
    """Return thread-scoped Codex overrides.

    App-server creates each thread with its own effective configuration. The
    root CLI ``--dangerously-bypass-hook-trust`` flag isn't propagated through
    that path, so active automation must opt the created thread into the
    already-vetted wrapper hooks explicitly.
    """
    config: dict[str, Any] = {}
    if cfg.model_reasoning_effort:
        config["model_reasoning_effort"] = cfg.model_reasoning_effort
    if cfg.recovery_mode == "active":
        config["bypass_hook_trust"] = True
    return config or None


def _forwarded_env_flags(cfg: RunnerConfig) -> list[str]:
    """`-e KEY=VALUE` for API keys present on the host (default + provider env_keys)."""
    keys = list(cfg.forward_env_keys)
    for table in cfg.model_providers.values():
        env_key = table.get("env_key")
        if env_key and env_key not in keys:
            keys.append(env_key)
    flags: list[str] = []
    for key in keys:
        val = os.environ.get(key)
        if val:
            flags += ["-e", f"{key}={val}"]
    return flags


def _launch_args_override(cfg: RunnerConfig, container: str, *, approval_policy: str) -> tuple[str, ...]:
    """Full command the SDK will spawn: codex app-server INSIDE the container over stdio."""
    recovery_env: list[str] = []
    if cfg.recovery_mode == "active":
        bridge_timeout = max(
            cfg.recovery_hook_timeout_seconds,
            cfg.recovery_model_timeout_seconds + 15,
        )
        recovery_env = [
            "-e", "TRAJREC_IPC_ROOT=/trajrec_ipc",
            "-e", f"TRAJREC_HOOK_TIMEOUT_SECONDS={bridge_timeout}",
        ]
    return tuple([
        "docker", "exec", "-i",
        "-u", "root",
        "-w", "/testbed",
        "-e", "PYTHONIOENCODING=utf-8",  # Mac/emulation parity with the eval harness.
        *recovery_env,
        *_forwarded_env_flags(cfg),
        container,
        cfg.codex_container_bin,
        *_codex_config_kv(cfg, approval_policy=approval_policy),
        "app-server", "--listen", "stdio://",
    ])


# =====================================================================================
# Core: run Codex on one instance
# =====================================================================================
def _run_codex_on_instance(
    cfg: RunnerConfig,
    inst: dict[str, Any],
    container: str,
    *,
    recovery_ipc_root: Path | None = None,
) -> dict[str, Any]:
    """Drive Codex against /testbed in `container`. Returns a result dict for serialization."""
    from openai_codex import Codex, CodexConfig, Sandbox  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    iid = str(inst["instance_id"])
    task = str(inst["problem_statement"])
    base = _base_commit(container)
    if base is None:
        raise RuntimeError(
            "Unable to capture the immutable repository HEAD before Codex started; "
            "refusing to run because the resulting patch could be silently empty."
        )
    baseline_untracked = _list_untracked_files(container)
    active = cfg.recovery_mode == "active"
    if active and recovery_ipc_root is None:
        raise RuntimeError("Active Codex recovery requires a mounted hook IPC directory")
    # The approval policy is identical in both experimental conditions. Native hooks provide
    # the treatment interposition and do not depend on approval callbacks.
    approval_policy = "never"

    codex_cfg = CodexConfig(
        launch_args_override=_launch_args_override(cfg, container, approval_policy=approval_policy),
        # cwd/env here refer to the HOST `docker exec` process; the model env is forwarded via -e.
    )

    controller: Any = None
    hook_service: Any = None
    recovery_meta: dict[str, Any] = {}
    exit_status = "codex_completed"
    final_response = ""
    n_turns = 0
    last_usage: Any = None  # Codex reports cumulative token usage on each turn result.
    transcript: list[dict[str, Any]] = []  # full step-by-step trajectory (parity with mswea)
    error_str = ""
    delivery_retries = 0  # how many empty-diff delivery-retry turns we had to send
    salvage_kind = ""     # "" if no salvage; else "apply_patch"/"unified_diff" that rescued the diff
    salvage_debug: dict[str, Any] = {}

    # Map our Rust-style sandbox_mode to the SDK's Sandbox preset (names differ:
    # "danger-full-access" -> Sandbox.full_access).
    _sdk_sandbox = {
        "read-only": "read_only",
        "workspace-write": "workspace_write",
        "danger-full-access": "full_access",
        "full-access": "full_access",
    }.get(cfg.sandbox_mode, cfg.sandbox_mode.replace("-", "_"))
    sandbox = getattr(Sandbox, _sdk_sandbox, None)

    codex = Codex(config=codex_cfg)
    try:
        if active:
            from src.adapters.codex.controller import CodexHookRecoveryController
            from src.adapters.codex.controller import RecoveryHookService

            controller = CodexHookRecoveryController(
                task=task,
                reviewer_model=_reviewer_model_name(cfg),
                reviewer_timeout=cfg.recovery_model_timeout_seconds,
                reviewer_kwargs=cfg.recovery_model_kwargs,
                context_token_limit=cfg.recovery_context_token_limit,
                max_submission_blocks=cfg.recovery_max_submission_blocks,
                max_action_blocks=cfg.recovery_max_action_blocks,
                repeat_threshold=cfg.recovery_repeat_threshold,
                fail_open=cfg.recovery_fail_open,
                review_branches=cfg.recovery_review_branches,
                gate_mutations=cfg.recovery_gate_mutations,
                gate_submission=cfg.recovery_gate_submission,
                workspace_snapshot=lambda: _capture_diff(
                    container,
                    base,
                    baseline_untracked,
                ),
            )
            hook_service = RecoveryHookService(
                recovery_ipc_root,
                controller,
                fail_open=cfg.recovery_fail_open,
            )
            hook_service.start()

        thread = codex.thread_start(
            model=cfg.model_name,
            config=_thread_start_config(cfg),
        )
        def _one_turn(prompt: str):
            nonlocal n_turns, last_usage
            kwargs: dict[str, Any] = {"cwd": "/testbed"}
            if sandbox is not None:
                kwargs["sandbox"] = sandbox
            transcript.append({"role": "user", "type": "turn_prompt", "content": prompt})
            turn_result = thread.turn(prompt, **kwargs).run()
            n_turns += 1
            if getattr(turn_result, "usage", None) is not None:
                last_usage = turn_result.usage  # cumulative for the thread; keep the latest
            for item in (getattr(turn_result, "items", None) or []):
                transcript.append(_serialize_thread_item(item))
            return turn_result

        result = _one_turn(_build_task_prompt(task))
        # Codex dispatches SessionStart at the beginning of the first turn (not at
        # thread_start). Never accept an active result whose native hook handshake was absent.
        if active and controller is not None and not controller.wait_for_handshake(timeout=1.0):
            raise RuntimeError(
                "Codex recovery hooks did not complete the SessionStart handshake; refusing "
                "to accept an unmonitored repair result. Check hook trust and hook discovery."
            )
        final_response = getattr(result, "final_response", "") or ""

        if active and controller is not None:
            recovery_meta = controller.metadata()
            if hook_service is not None and hook_service.errors:
                recovery_meta["hook_service_errors"] = list(hook_service.errors)

        # Delivery floor (BOTH arms): ensure an evaluable, non-empty patch. If the turn(s) ended
        # with an empty working-tree diff, push Codex to actually land a change (re-apply via
        # apply_patch, verify non-empty). Deterministic, wrapper-controlled; delivery-only.
        for _ in range(cfg.delivery_max_retries):
            if _capture_diff(container, base, baseline_untracked).strip():
                break
            result = _one_turn(_DELIVERY_RETRY_FEEDBACK)
            final_response = getattr(result, "final_response", "") or final_response
            delivery_retries += 1

        # Last resort: if the diff is still empty, the model emitted a patch but applied it with
        # the wrong tool (or a no-op). Extract its intended patch from the transcript and apply
        # it ourselves so we always have an evaluable, non-empty artifact.
        if cfg.enable_patch_salvage and not _capture_diff(
            container,
            base,
            baseline_untracked,
        ).strip():
            salvage_kind, salvage_debug = _salvage_empty_patch(
                container,
                base,
                transcript,
                baseline_untracked,
            )
    except Exception as exc:
        exit_status = type(exc).__name__
        error_str = str(exc)
    finally:
        if hook_service is not None:
            hook_service.stop()
            if active and controller is not None:
                recovery_meta = controller.metadata()
                if hook_service.errors:
                    recovery_meta["hook_service_errors"] = list(hook_service.errors)
        try:
            codex.close()
        except Exception:
            pass

    model_patch = _capture_diff(container, base, baseline_untracked)
    tokens = _usage_breakdown(last_usage)
    solver_litellm = f"{cfg.model_provider or 'openai'}/{cfg.model_name}"
    instance_cost = _estimate_cost(
        solver_litellm, tokens.get("input_tokens", 0), tokens.get("output_tokens", 0)
    )
    # Mirror mini-swe-agent's info.model_stats shape (api_calls + instance_cost) plus the
    # full Codex token breakdown (input/output/cached/reasoning/total).
    # `api_calls` counts Codex ACTION STEPS (commandExecution + agentMessage items) so it is
    # comparable to mini-swe-agent's per-step api_calls. Codex runs many internal model calls
    # inside a single thread.turn(), so the raw turn count is kept separately as `turns`.
    step_count = sum(1 for it in transcript if it.get("type") in {"commandExecution", "agentMessage"})
    model_stats = {
        "api_calls": step_count,
        "turns": n_turns,
        "delivery_retries": delivery_retries,
        "salvaged": bool(salvage_kind),
        "salvage_kind": salvage_kind,
        "instance_cost": instance_cost,
        **tokens,
    }
    info: dict[str, Any] = {
        "model_stats": model_stats,
        "exit_status": exit_status,
        "submission": model_patch,
        "final_response": final_response,
        "config": {
            "model_name": cfg.model_name,
            "model_provider": cfg.model_provider,
            "reasoning_effort": cfg.model_reasoning_effort,
            "sandbox_mode": cfg.sandbox_mode,
            "recovery_mode": cfg.recovery_mode,
            "approval_policy": approval_policy,
            "hook_adapter": "codex-native-hooks-v1" if active else None,
            "strip_git_history": cfg.strip_git_history,
            "delivery_max_retries": cfg.delivery_max_retries,
            "patch_salvage_enabled": cfg.enable_patch_salvage,
            "recovery_review_branches": cfg.recovery_review_branches if active else None,
            "recovery_gate_mutations": cfg.recovery_gate_mutations if active else None,
            "recovery_gate_submission": cfg.recovery_gate_submission if active else None,
        },
    }
    if error_str:
        info["error"] = error_str
    if salvage_debug:
        info["salvage_debug"] = salvage_debug
    # Only recovery runs carry recovery metadata. Keep reviewer_stats in info for aggregate
    # readers, but append the full trajectory_recovery block at the top level below, matching
    # mini-swe-agent's trajectory layout without duplicating the potentially large evidence log.
    if active:
        if controller is not None:
            info["reviewer_stats"] = {
                "reviewer_calls": getattr(controller.recovery_policy, "n_calls", None),
                "reviewer_cost": getattr(controller.recovery_policy, "cost", None),
            }
    # Mirror mini-swe-agent traj layout: info + full messages transcript + instance_id.
    # model_patch is also kept at top level for eval/_process_instance convenience.
    trajectory = {
        "instance_id": iid,
        "info": info,
        "messages": transcript,
        "trajectory_format": "codex-app-server-native-hooks-v2",
        "base_commit": base,
        "model_patch": model_patch,
    }
    # Append this last so a raw trajectory ends with the same directly inspectable recovery
    # summary as mini-swe-agent. Off-mode trajectories intentionally omit the field.
    if active:
        trajectory["trajectory_recovery"] = recovery_meta or {"mode": "active"}
    return trajectory


def _update_preds_file(preds_path: Path, iid: str, model_name: str, patch: str) -> None:
    preds: dict[str, Any] = {}
    if preds_path.exists():
        try:
            preds = json.loads(preds_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            preds = {}
    preds[iid] = {
        "instance_id": iid,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }
    ensure_dir(preds_path.parent)
    preds_path.write_text(json.dumps(preds, indent=2), encoding="utf-8")


def _process_instance(cfg: RunnerConfig, inst: dict[str, Any], rep_dir: Path) -> dict[str, Any]:
    iid = str(inst["instance_id"])
    container = f"codex-sweb-{iid}-{os.getpid()}"
    inst_dir = rep_dir / iid
    ensure_dir(inst_dir)
    recovery_ipc_root: Path | None = None
    if cfg.recovery_mode == "active":
        recovery_ipc_root = inst_dir / ".codex_recovery_ipc"
        shutil.rmtree(recovery_ipc_root, ignore_errors=True)
        ensure_dir(recovery_ipc_root)
    result: dict[str, Any] = {"instance_id": iid, "exit_status": "unknown"}
    try:
        _start_container(
            cfg,
            inst,
            container,
            recovery_ipc_root=recovery_ipc_root,
        )
        result = _run_codex_on_instance(
            cfg,
            inst,
            container,
            recovery_ipc_root=recovery_ipc_root,
        )
    except Exception as exc:
        result = {"instance_id": iid, "exit_status": type(exc).__name__, "model_patch": "", "error": str(exc)}
        if cfg.raise_exceptions:
            _stop_container(container)
            raise
    finally:
        # Save trajectory/metadata + prediction before tearing down the container.
        (inst_dir / f"{iid}.traj.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (inst_dir / f"{iid}.patch").write_text(str(result.get("model_patch") or ""), encoding="utf-8")
        _update_preds_file(rep_dir / "preds.json", iid, cfg.model_name, str(result.get("model_patch") or ""))
        _stop_container(container)
        if recovery_ipc_root is not None:
            shutil.rmtree(recovery_ipc_root, ignore_errors=True)
    return result


# =====================================================================================
# Orchestration (per-instance subprocess + watchdog, mirrors the mini-swe-agent pipeline)
# =====================================================================================
def _extract_instance_summary(rep_dir: Path, iid: str) -> dict[str, Any] | None:
    traj = rep_dir / iid / f"{iid}.traj.json"
    if not traj.exists():
        return None
    try:
        data = json.loads(traj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    info = data.get("info") or {}
    ms = info.get("model_stats") or data.get("model_stats") or {}
    rs = info.get("reviewer_stats") or data.get("reviewer_stats") or {}
    return {
        "exit_status": info.get("exit_status") or data.get("exit_status"),
        "patch_len": len(str(data.get("model_patch") or "")),
        "api_calls": ms.get("api_calls"),
        "instance_cost": ms.get("instance_cost"),
        "input_tokens": ms.get("input_tokens"),
        "output_tokens": ms.get("output_tokens"),
        "cached_input_tokens": ms.get("cached_input_tokens"),
        "total_tokens": ms.get("total_tokens"),
        "reviewer_calls": rs.get("reviewer_calls"),
        "reviewer_cost": rs.get("reviewer_cost"),
    }


def _run_one_instance(instance_path: Path, rep_dir: Path) -> None:
    inst = json.loads(instance_path.read_text(encoding="utf-8"))
    if not isinstance(inst, dict):
        raise ValueError(f"One-instance file must contain a JSON object: {instance_path}")
    cfg = _build_runner_config()
    _process_instance(cfg, inst, rep_dir)


def _instances_filter_is_nonempty() -> bool:
    fv = CONFIG.get("INSTANCES_FILTER")
    if isinstance(fv, (list, tuple, set)):
        return bool(fv)
    if isinstance(fv, str):
        return bool(fv.strip()) and fv.strip() != ".*"
    return False


def run() -> None:
    # Set the LiteLLM cost-tracking mode before any reviewer/pricing call (parity with mswea).
    cost_tracking = str(CONFIG.get("MSWEA_COST_TRACKING") or "").strip()
    if cost_tracking:
        os.environ["MSWEA_COST_TRACKING"] = cost_tracking

    if len(sys.argv) >= 4 and sys.argv[1] == "--one-instance":
        _run_one_instance(Path(sys.argv[2]), Path(sys.argv[3]))
        return

    # A non-empty INSTANCES_FILTER means "I explicitly want these ids" -> force a redo so an
    # earlier (possibly failed) run doesn't get skipped. Mirrors src/runners/minisweagent_swebench_lite.py.
    if _instances_filter_is_nonempty() and not CONFIG.get("REDO_EXISTING", False):
        log_message("INSTANCES_FILTER is non-empty → forcing REDO_EXISTING=True")
        CONFIG["REDO_EXISTING"] = True

    cfg = _build_runner_config()
    if cfg.recovery_mode not in {"off", "active"}:
        raise ValueError(f"CODEX_RECOVERY_MODE must be 'off' or 'active', got {cfg.recovery_mode!r}")

    vroot = get_variant_root(cfg)
    rep_dir = vroot / "repair"
    ensure_dir(vroot / "localization")
    ensure_dir(rep_dir)

    instances = _load_variant_instances(cfg)
    if not instances:
        raise ValueError("No instances produced. Check dataset/filters.")
    (vroot / "instances.json").write_text(json.dumps(instances, indent=2), encoding="utf-8")

    selected = _apply_instances_filter(instances, cfg)
    if not selected:
        raise ValueError("No instances selected after INSTANCES_FILTER/SLICE")
    (vroot / "instances_selected.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    (vroot / "run_manifest.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": cfg.variant,
        "codex_recovery_mode": cfg.recovery_mode,
        "dataset_json": str(cfg.dataset_json),
        "model": {"model_name": cfg.model_name, "model_provider": cfg.model_provider,
                  "reasoning_effort": cfg.model_reasoning_effort},
        "sandbox_mode": cfg.sandbox_mode,
        "strip_git_history": cfg.strip_git_history,
        "run_mode": cfg.run_mode,
        "recovery": ({
            "reviewer_model": _reviewer_model_name(cfg),
            "max_action_blocks": cfg.recovery_max_action_blocks,
            "max_submission_blocks": cfg.recovery_max_submission_blocks,
            "gate_mutations": cfg.recovery_gate_mutations,
            "gate_submission": cfg.recovery_gate_submission,
        } if cfg.recovery_mode == "active" else {}),
    }, indent=2), encoding="utf-8")

    if bool(CONFIG.get("DRY_RUN", False)):
        log_message(f"DRY_RUN=True; would run {len(selected)} instances -> {rep_dir}")
        return

    if cfg.pull_docker_images_before_run:
        _pull_docker_images(selected)

    run_mode = cfg.run_mode.strip().lower()
    if run_mode not in {"per_instance", "batch"}:
        raise ValueError(f"RUN_MODE must be 'per_instance' or 'batch', got {cfg.run_mode!r}")

    if run_mode == "batch":
        for inst in selected:
            iid = str(inst.get("instance_id", "")).strip()
            traj = rep_dir / iid / f"{iid}.traj.json"
            if (not cfg.redo_existing) and traj.exists():
                log_message(f"Instance={iid} skipped (existing)")
                continue
            log_message(f"Instance={iid} starting")
            _process_instance(cfg, inst, rep_dir)
        log_message(f"Done. Outputs under: {rep_dir}")
        return

    # per_instance: subprocess + watchdog
    max_minutes = int(cfg.max_minutes_per_instance or 15)
    timeout_seconds = max(60, max_minutes * 60)
    tick = max(1, cfg.watchdog_tick_seconds)
    status_path = vroot / "per_instance_status.jsonl"
    skipped_path = vroot / "skipped_instances.jsonl"
    one_dir = vroot / "instances_one"
    ensure_dir(one_dir)
    exit_status_path = rep_dir / "run_batch_exit_statuses.yaml"
    agg, agg_cost = _read_exit_statuses_yaml(exit_status_path)
    agg_cost = float(agg_cost or 0.0)
    recorded = _recorded_instance_ids(agg)

    # Backfill exit statuses for instances a prior run recorded in status jsonl but not the
    # aggregate yaml (e.g. subprocess died before writing a trajectory). Mirrors mswea.
    for prior in read_jsonl(status_path):
        piid = str(prior.get("instance_id") or "").strip()
        if not piid or piid in recorded:
            continue
        es = str(prior.get("exit_status") or "").strip()
        if not es:
            reason = str(prior.get("reason") or "").strip()
            es = ("TimeoutExpired" if reason == "watchdog_timeout"
                  else "subprocess_error" if prior.get("status") == "error" else "unknown")
        _record_exit_status(agg, es, piid)
        recorded.add(piid)
    _write_exit_statuses_yaml(exit_status_path, agg, total_cost=agg_cost)

    log_message(f"Running (per_instance): {len(selected)} instances; watchdog={max_minutes} min")
    for inst in selected:
        iid = str(inst.get("instance_id", "")).strip()
        if not iid:
            continue
        traj = rep_dir / iid / f"{iid}.traj.json"
        if (not cfg.redo_existing) and traj.exists():
            summary = _extract_instance_summary(rep_dir, iid) or {}
            log_message(f"Instance={iid} skipped ({fmt_exit_status_hint(summary.get('exit_status')) or 'existing'})")
            _record_exit_status(agg, str(summary.get("exit_status") or "skipped_existing"), iid)
            try:
                agg_cost += float(summary.get("instance_cost") or 0.0)
            except (TypeError, ValueError):
                pass
            _write_exit_statuses_yaml(exit_status_path, agg, total_cost=agg_cost)
            append_jsonl(skipped_path, {"instance_id": iid, "reason": "redo_existing_false", **summary})
            continue

        log_message(f"Instance={iid} starting")
        one_path = one_dir / f"{iid}.json"
        one_path.write_text(json.dumps(inst, indent=2), encoding="utf-8")
        started = datetime.now(timezone.utc)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--one-instance", str(one_path), str(rep_dir)]
        proc: subprocess.Popen[Any] | None = None
        row: dict[str, Any] = {"instance_id": iid, "started_at_utc": started.isoformat(), "timeout_seconds": timeout_seconds}
        try:
            proc = subprocess.Popen(cmd)
            start_monotonic = time.monotonic()
            next_tick = start_monotonic
            while True:
                rc = proc.poll()
                now = time.monotonic()
                if rc is not None:
                    row["returncode"] = int(rc)
                    row["status"] = "ok" if rc == 0 else "error"
                    break
                if now - start_monotonic >= timeout_seconds:
                    raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds)
                if now >= next_tick:
                    elapsed = int(now - start_monotonic)
                    log_message(
                        f"Instance={iid} running; elapsed={_format_duration(elapsed)} "
                        f"remaining={_format_duration(max(0, timeout_seconds - elapsed))}"
                    )
                    next_tick = now + tick
                time.sleep(0.5)
        except subprocess.TimeoutExpired:
            row.update({"status": "skipped", "reason": "watchdog_timeout", "returncode": None})
            if proc is not None:
                try:
                    proc.kill()
                finally:
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        pass
            _stop_container(f"codex-sweb-{iid}-{proc.pid if proc else 'na'}")
            append_jsonl(skipped_path, {"instance_id": iid, "reason": "watchdog_timeout",
                                         "max_minutes_per_instance": max_minutes})
        ended = datetime.now(timezone.utc)
        row["ended_at_utc"] = ended.isoformat()
        row["duration_seconds"] = int((ended - started).total_seconds())
        summary = _extract_instance_summary(rep_dir, iid) or {}
        exit_status = str(summary.get("exit_status") or "").strip()
        if not exit_status:
            reason = str(row.get("reason") or "")
            exit_status = "TimeoutExpired" if reason == "watchdog_timeout" else (
                "subprocess_error" if row.get("status") == "error" else "unknown")
        _record_exit_status(agg, exit_status, iid)
        try:
            agg_cost += float(summary.get("instance_cost") or 0.0)
        except (TypeError, ValueError):
            pass
        _write_exit_statuses_yaml(exit_status_path, agg, total_cost=agg_cost)
        row.update(summary)
        append_jsonl(status_path, row)
        log_message(
            f"Instance={iid} done: status={row.get('status')} exit_status={exit_status} "
            f"duration={_format_duration(row.get('duration_seconds') or 0)} "
            f"calls={summary.get('api_calls')} "
            f"cost={summary.get('instance_cost')} in_tok={summary.get('input_tokens')} "
            f"out_tok={summary.get('output_tokens')} cached_tok={summary.get('cached_input_tokens')} "
            f"reviewer_cost={summary.get('reviewer_cost')} patch_len={summary.get('patch_len')}"
        )
        if cfg.delete_docker_image_after_each_instance:
            _cleanup_images_for_instances([inst], force=cfg.force_delete_docker_images)

    for inst in selected:
        iid = str(inst.get("instance_id", "")).strip()
        if iid and iid not in _recorded_instance_ids(agg):
            summary = _extract_instance_summary(rep_dir, iid) or {}
            _record_exit_status(agg, str(summary.get("exit_status") or "missing"), iid)
    _write_exit_statuses_yaml(exit_status_path, agg, total_cost=agg_cost)

    if cfg.delete_docker_images_after_run:
        _cleanup_images_for_instances(selected, force=cfg.force_delete_docker_images)
    log_message(f"Done. Outputs under: {rep_dir}  total_cost=${agg_cost:.4f}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
