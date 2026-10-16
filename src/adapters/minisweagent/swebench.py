"""Opt-in SWE-bench runner for real-time trajectory recovery.

This module is deliberately separate from ``swebench.py``. Existing callers
continue to use the stock ``process_instance()`` unless a wrapper explicitly
selects this implementation.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    get_sb_environment,
    remove_from_preds_file,
    update_preds_file,
)
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import logger

from src.recovery import LiteLLMRecoveryPolicy, RecoveryProgressTrackingAgent


def process_instance_with_recovery(
    instance: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
    progress_manager: RunBatchProgressManager,
    *,
    recovery_config: dict[str, Any],
) -> None:
    """Process one instance with online review between solver actions."""

    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    traj_path = instance_dir / f"{instance_id}.traj.json"
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    traj_path.unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]
    reviewer_model = recovery_config.get("model_name") or model.config.model_name
    policy = LiteLLMRecoveryPolicy(
        reviewer_model,
        timeout=int(recovery_config.get("timeout", 180)),
        model_kwargs=recovery_config.get("model_kwargs") or {},
        context_token_limit=int(recovery_config.get("context_token_limit", 250_000)),
    )

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent = None
    exit_status = None
    result = None
    extra_info: dict[str, Any] = {}

    try:
        env = get_sb_environment(config, instance)
        agent_config = dict(config.get("agent", {}))
        # Recovery runs are checkpointed after every step by DefaultAgent.run().
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
        exit_status = info.get("exit_status")
        result = info.get("submission")
    except Exception as exc:
        logger.error(f"Error processing instance {instance_id}: {exc}", exc_info=True)
        exit_status, result = type(exc).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(exc)}
    finally:
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
            logger.info(f"Saved recovery trajectory to '{traj_path}'")
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)
