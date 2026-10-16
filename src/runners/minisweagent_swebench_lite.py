"""src/runners/minisweagent_swebench_lite.py

One-click runner: run mini-swe-agent on a custom JSON dataset (SWE-bench-like list of dicts)
while storing outputs under a shared run directory that can later hold multiple variants:

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

This uses a shared per-variant directory layout and executes mini-swe-agent's
internal SWE-bench runner (`minisweagent.run.benchmarks.swebench.process_instance`).

Notes
-----
- mini-swe-agent's official `mini-extra swebench` CLI expects HF datasets. This wrapper loads a
  plain JSON list (your report-variant files) and runs instances directly.
- SWE-bench runs require docker (unless you change the environment class).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Executed by file path (e.g. the per-instance watchdog subprocess): make `src` importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name, process_instance
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.serialize import UNSET, recursive_merge

from src.adapters.minisweagent.docker import use_as_docker_environment
from src.runners.common import apply_instances_filter, build_recovery_config, cleanup_images_for_instances, extract_instance_summary, filter_records, fmt_exit_status_hint, normalize_problem_statement, read_exit_statuses_yaml, record_exit_status, recorded_instance_ids, setup_output_dirs, write_exit_statuses_yaml
from src.utils.docker import remove_minisweagent_containers, stop_and_remove_containers_for_image
from src.utils.io import append_jsonl, ensure_dir, load_json_list, read_jsonl, resolve_path
from src.utils.log import log_message

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Route environment_class="docker" to the environment that supports STRIP_GIT_HISTORY.
use_as_docker_environment()


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


# Substrings that, when found in MODEL_NAME, force the anti-leakage sandbox on (network cut +
# git history stripped). These are models observed to exploit the leaks; everything else runs
# unsandboxed by default. See DISABLE_NETWORK / STRIP_GIT_HISTORY below.
STRICT_SANDBOX_MODEL_MARKERS: tuple[str, ...] = ("deepseek",)


def _needs_strict_sandbox(model_name: str | None) -> bool:
    name = (model_name or "").lower()
    return any(marker in name for marker in STRICT_SANDBOX_MODEL_MARKERS)

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

    # Input dataset JSON for this variant (SWE-bench-like list of dicts)
    # Can be an absolute path or relative to the repository root.
    # "DATASET_JSON": "data/swe_bench_lite/astropy__astropy.json",
    "DATASET_JSON": "data/swe_bench_lite/pallets__flask.json",

    # Output layout
    "RUN_NAME": "pallets__flask",
    "RESULTS_ROOT": "results_trajrec/deepseek_flash",
    # "RESULTS_ROOT": "results_trajrec/gpt_5_mini",

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
    "MODEL_NAME": "deepseek/deepseek-flash",   # "openai/gpt-5-mini-2025-08-07" or "deepseek/deepseek-flash" or "minimax/minimax-m2.5" or "openai/gpt-5.4-mini-2026-03-17" or "openai/gpt-5.4-2026-03-05"
    "MODEL_CLASS": None, # None | "openrouter" [None - for openai gpt/deepseek models]
    # Cost tracking mode for LiteLLM model price lookup.
    # - "default": fail instance if model pricing metadata is missing
    # - "ignore_errors": continue even if cost can't be computed
    "MSWEA_COST_TRACKING": "ignore_errors",
    "ENVIRONMENT_CLASS": "docker",
    # --- Anti-leakage sandbox (see _build_minisweagent_config) ---
    # DISABLE_NETWORK: cut container networking (--network none) so the agent cannot fetch the
    #   ground-truth fix from GitHub (curl/wget/pip).
    # STRIP_GIT_HISTORY: rewrite the repo to a single base commit so the agent cannot recover the
    #   fix commit shipped in the image (`git log --all` / `git show <sha>`). git diff/apply still work.
    # Both default to None = AUTO: derived from MODEL_NAME. Models whose name matches
    #   STRICT_SANDBOX_MODEL_MARKERS (e.g. deepseek) run fully sandboxed (both True); all other
    #   models run unsandboxed (both False). Set a bool here to override the automatic choice.
    "DISABLE_NETWORK": None,
    "STRIP_GIT_HISTORY": None,
    "MODEL_REQUEST_TIMEOUT_SECONDS": 180,

    # Real-time trajectory recovery (strictly opt-in).
    # - "off": stock mini-swe-agent process_instance() path, unchanged.
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
    "PER_INSTANCE_CALL_LIMIT": 500,
    "PER_INSTANCE_COST_LIMIT": 0.50,
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
    "PULL_DOCKER_IMAGES_BEFORE_RUN": False,
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
}


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
        subprocess.run(["docker", "pull", img], check=True)


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
        # Both default to AUTO (None): sandbox strictly iff the model matches
        # STRICT_SANDBOX_MODEL_MARKERS (e.g. deepseek). An explicit bool overrides.
        disable_network=(
            _needs_strict_sandbox(CONFIG.get("MODEL_NAME"))
            if CONFIG.get("DISABLE_NETWORK") is None
            else bool(CONFIG.get("DISABLE_NETWORK"))
        ),
        strip_git_history=(
            _needs_strict_sandbox(CONFIG.get("MODEL_NAME"))
            if CONFIG.get("STRIP_GIT_HISTORY") is None
            else bool(CONFIG.get("STRIP_GIT_HISTORY"))
        ),
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

    configs = [get_config_from_spec(spec) for spec in cfg.config_spec]
    configs.append(
        {
            "agent": {
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
                # mini-swe-agent uses DockerEnvironmentConfig.pull_timeout for the *docker run* startup.
                # Default is 120s, which is often too low when images need pulling.
                "pull_timeout": max(120, int(cfg.max_minutes_per_instance or 15) * 60),
                # Cut container networking (DISABLE_NETWORK) so the agent cannot fetch the
                # ground-truth fix from GitHub (curl/wget/pip). Replaces the default ["--rm"]
                # (recursive_merge overrides lists, so it is not concatenated). Set
                # DISABLE_NETWORK=False for trusted models that need in-loop `pip install`.
                "run_args": ["--rm", "--network", "none"] if cfg.disable_network else UNSET,
                # Strip the repo's git history to a single base commit at startup, so the
                # agent cannot recover the ground-truth fix commit shipped in the image via
                # `git log --all` / `git show <sha>`. `git diff`/`git apply` still work.
                "strip_git_history": cfg.strip_git_history or UNSET,
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
        base["problem_statement"] = normalize_problem_statement(
            r.get(cfg.problem_field, r.get("problem_statement", ""))
        )
        records.append(base)

    records = filter_records(records, prefix=cfg.project_prefix, limit=cfg.limit_instances)
    log_message(f"Variant={cfg.variant} records ready: {len(records)}")

    instances: list[dict[str, Any]] = []
    for r in records:
        iid = str(r.get("instance_id", "")).strip()
        if not iid:
            continue
        image_name = get_swebench_docker_image_name({"instance_id": iid, "image_name": r.get("image_name")})
        instances.append(
            {
                "instance_id": iid,
                "problem_statement": str(r.get("problem_statement", "")),
                "image_name": image_name,
                **{k: v for k, v in r.items() if k not in {"problem_statement"}},
            }
        )

    return instances

def _process_instance_for_config(
    cfg: RunnerConfig,
    inst: dict[str, Any],
    rep_dir: Path,
    miniconfig: dict[str, Any],
    progress_manager: RunBatchProgressManager,
) -> None:
    """Dispatch without importing recovery code on normal runs."""
    if cfg.trajectory_recovery_mode == "off":
        process_instance(inst, rep_dir, miniconfig, progress_manager)
        return

    from src.adapters.minisweagent.swebench import process_instance_with_recovery

    process_instance_with_recovery(
        inst,
        rep_dir,
        miniconfig,
        progress_manager,
        recovery_config=build_recovery_config(cfg),
    )


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
    if live_cm:
        live_cm.__enter__()
    try:
        _process_instance_for_config(cfg, inst, rep_dir, miniconfig, progress_manager)
    finally:
        if live_cm:
            live_cm.__exit__(None, None, None)

    preds_path = rep_dir / "preds.json"
    if preds_path.exists() and iid:
        preds = json.loads(preds_path.read_text(encoding="utf-8"))
        patch = str((preds.get(iid) or {}).get("model_patch") or "")
        (rep_dir / iid).mkdir(parents=True, exist_ok=True)
        (rep_dir / iid / f"{iid}.patch").write_text(patch, encoding="utf-8")


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
    (vroot / "instances_selected.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    miniconfig = _build_minisweagent_config(cfg)

    if cfg.global_call_limit is not None:
        GLOBAL_MODEL_STATS.call_limit = int(cfg.global_call_limit)
    if cfg.global_cost_limit is not None:
        GLOBAL_MODEL_STATS.cost_limit = float(cfg.global_cost_limit)

    (vroot / "run_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "variant": cfg.variant,
                "dataset_json": str(cfg.dataset_json),
                "reference_json": str(cfg.reference_json) if cfg.reference_json else None,
                "problem_field": cfg.problem_field,
                "workers": cfg.workers,
                "model": {"model_name": cfg.model_name, "model_class": cfg.model_class},
                "environment": {"environment_class": cfg.environment_class},
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
                    try:
                        proc.kill()
                    finally:
                        try:
                            proc.wait(timeout=10)
                        except Exception:
                            pass
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

    log_message(f"Done. Outputs written under: {rep_dir}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
