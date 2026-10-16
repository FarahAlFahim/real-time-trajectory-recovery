"""Native-hook adapter from Codex CLI events to the shared TrajRec controller."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from src.recovery import LiteLLMRecoveryPolicy
from src.recovery import RecoveryAgent
from src.recovery import RecoveryDecision


class CodexHookRecoveryController(RecoveryAgent):
    """Adapt native Codex lifecycle events to the tested RecoveryAgent policy.

    RecoveryAgent's solver-specific query/execute loop is intentionally not
    used. This class initializes only its recovery state and reuses its review,
    evidence, semantic-constraint, repetition, and validation logic.
    """

    def __init__(
        self,
        *,
        task: str,
        reviewer_model: str,
        reviewer_timeout: int,
        reviewer_kwargs: dict[str, Any],
        context_token_limit: int,
        max_submission_blocks: int,
        max_action_blocks: int,
        repeat_threshold: int,
        fail_open: bool,
        review_branches: bool,
        gate_mutations: bool,
        gate_submission: bool,
        workspace_snapshot: Callable[[], str],
    ) -> None:
        # Do not call RecoveryAgent.__init__: it requires a mini-swe-agent
        # model/environment. The fields below are the agent-independent state
        # consumed by RecoveryAgent's recovery methods.
        self.recovery_policy = LiteLLMRecoveryPolicy(
            reviewer_model,
            timeout=reviewer_timeout,
            model_kwargs=reviewer_kwargs,
            context_token_limit=context_token_limit,
        )
        self.recovery_mode = "active"
        self.recovery_max_submission_blocks = max(0, max_submission_blocks)
        self.recovery_max_action_blocks = max(0, max_action_blocks)
        self.recovery_repeat_threshold = max(1, repeat_threshold)
        self.recovery_fail_open = fail_open
        self.recovery_review_branches = review_branches
        self.recovery_gate_mutations = gate_mutations
        self.recovery_gate_submission = gate_submission
        self.recovery_events: list[dict[str, Any]] = []
        self.recovery_state: dict[str, Any] = {
            "retained_findings": [],
            "unresolved_obligations": [],
            "behavioral_specification": [],
            "next_focus": "",
            "investigation_complete": False,
        }
        self.recovery_counts = {
            "branch_prunes": 0,
            "branch_stays": 0,
            "repeat_blocks": 0,
            "action_blocks": 0,
            "submission_blocks": 0,
        }
        self._task = task
        self._base_commit = None
        self._last_injection_signature = ""
        self._active_scope: str | None = None
        self._scope_revision = 0
        self._reviewed_scope_revision = -1
        self._branch_reviewed_keys: set[tuple[int, str]] = set()
        self._scope_history: list[dict[str, Any]] = []
        self._pruned_scopes: list[str] = []
        self._evidence_ledger: list[dict[str, Any]] = []
        self._semantic_constraints: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._workspace_revision = 0
        self._exploration_counts: defaultdict[str, int] = defaultdict(int)
        self._scope_counts: defaultdict[str, int] = defaultdict(int)
        self._revision_exploration_counts: defaultdict[str, int] = defaultdict(int)
        self._revision_scope_counts: defaultdict[str, int] = defaultdict(int)
        self._post_mutation_validation_scopes: set[str] = set()
        self._mutation_signature_counts: defaultdict[str, int] = defaultdict(int)
        self._mutation_scope_counts: defaultdict[str, int] = defaultdict(int)
        self._repeat_reviewed_keys: set[str] = set()
        self._submission_history: list[dict[str, Any]] = []
        self._last_submission_block_message_index: int | None = None
        self._review_message_start = 0
        self._semantic_guarded_scopes: set[str] = set()
        self._workspace_snapshot_callback = workspace_snapshot
        self._pending: dict[str, dict[str, Any]] = {}
        self._seen_stop_messages: set[tuple[str, str]] = set()
        self._seen_thoughts: set[str] = set()
        self._hook_event_counts: defaultdict[str, int] = defaultdict(int)
        self._session_start_seen = threading.Event()
        self.messages: list[dict[str, Any]] = []
        self.n_calls = 0
        self._lock = threading.RLock()

    def _workspace_snapshot(self) -> str:
        return self._workspace_snapshot_callback()

    @staticmethod
    def _proposed_message(payload: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(payload.get("tool_name") or "tool")
        tool_input = payload.get("tool_input")
        command = ""
        if isinstance(tool_input, dict):
            command = str(tool_input.get("command") or "")
        if not command:
            command = json.dumps(tool_input, ensure_ascii=False, default=str)
        return {
            "role": "assistant",
            "content": f"Proposed Codex tool: {tool_name}",
            "extra": {"actions": [{"command": command}]},
        }

    @staticmethod
    def _tool_response_text(payload: dict[str, Any]) -> str:
        response = payload.get("tool_response")
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            for key in ("output", "content", "stdout", "message"):
                value = response.get(key)
                if isinstance(value, str):
                    return value
        return json.dumps(response, ensure_ascii=False, default=str)

    @staticmethod
    def _feedback(decision: RecoveryDecision) -> str:
        parts = [f"[REAL-TIME TRAJECTORY RECOVERY: {decision.phase}]"]
        if decision.phase == "branch_review" and decision.decision == "PRUNE":
            parts.append("Branch label: IRRELEVANT (trajectory history retained; nothing was deleted).")
        if decision.reason:
            parts.append(f"Failure if omitted: {decision.reason}")
        if decision.evidence:
            parts.append(f"Blocking evidence: {decision.evidence[0]}")
        if decision.unresolved_obligations:
            parts.append(f"Immediate obligation: {decision.unresolved_obligations[0]}")
        if decision.next_focus:
            parts.append(f"Required next action: {decision.next_focus}")
        return "\n".join(parts)

    @classmethod
    def _pretool_deny(cls, decision: RecoveryDecision) -> dict[str, Any]:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": cls._feedback(decision),
            }
        }

    @classmethod
    def _stop_block(cls, decision: RecoveryDecision) -> dict[str, Any]:
        feedback = cls._feedback(decision)
        feedback += (
            "\nBefore stopping again, address this blocker or explain with repository/test evidence "
            "why it is resolved or irrelevant."
        )
        return {"decision": "block", "reason": feedback}

    def _record_event(self, decision: RecoveryDecision, *, policy_note: str = "") -> None:
        self._update_recovery_state(decision)
        self.recovery_events.append(
            {
                "step": self.n_calls,
                "phase": decision.phase,
                "mode": "active",
                "active_scope": self._active_scope,
                "recommended_decision": decision.decision,
                "effective_decision": decision.decision,
                "relevance_label": (
                    "IRRELEVANT"
                    if decision.phase == "branch_review" and decision.decision == "PRUNE"
                    else "RELEVANT" if decision.phase == "branch_review" else ""
                ),
                "policy_note": policy_note,
                **asdict(decision),
            }
        )

    def _git_commit_decision(self) -> RecoveryDecision:
        return RecoveryDecision(
            phase="before_action",
            decision="BLOCK_ACTION",
            reason="A git commit changes HEAD and can hide the repair from the required patch artifact.",
            evidence=["The proposed tool action contains git commit."],
            next_focus="Keep the repair uncommitted and continue testing or submit the working-tree diff.",
            unresolved_obligations=["Preserve the repair as an uncommitted patch."],
            owning_handling_points=["git commit action"],
            behavior_decision="UNRESOLVED",
            blocker_id="solver-commit-prohibited",
        )

    @staticmethod
    def _empty_patch_decision() -> RecoveryDecision:
        return RecoveryDecision(
            phase="before_submission",
            decision="BLOCK_SUBMISSION",
            reason="The repository diff is empty, so stopping now would return no repair.",
            evidence=["The diff from the immutable initial revision contains no source change."],
            next_focus="Implement the intended source change, inspect git diff, and stop only after the diff is non-empty.",
            unresolved_obligations=["Produce a non-empty patch implementing the reported repair."],
            demonstrated_paths=["workspace source change -> initial-revision diff -> submitted patch"],
            owning_handling_points=["working-tree patch artifact"],
            behavior_decision="UNRESOLVED",
            blocker_id="empty-submission-artifact",
        )

    def handle(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            for thought in payload.get("trajectory_thoughts") or []:
                text = str(thought).strip()
                if text and text not in self._seen_thoughts:
                    self.messages.append({"role": "assistant", "content": text})
                    self._seen_thoughts.add(text)
            event = str(payload.get("hook_event_name") or "")
            self._hook_event_counts[event] += 1
            if event == "SessionStart":
                self._session_start_seen.set()
                return None
            if event == "PreToolUse":
                return self._handle_pre_tool(payload)
            if event == "PostToolUse":
                self._handle_post_tool(payload)
                return None
            if event == "Stop":
                return self._handle_stop(payload)
            return None

    def wait_for_handshake(self, timeout: float) -> bool:
        """Return only after Codex proves that the installed native hooks are executing."""
        return self._session_start_seen.wait(timeout=max(0.0, timeout))

    def _handle_pre_tool(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        self.n_calls += 1
        proposed = self._proposed_message(payload)
        next_scope = self._proposed_scope(proposed)
        is_mutation = self._is_mutation(proposed)

        if self._contains_git_commit(proposed):
            decision = self._git_commit_decision()
            self._record_event(decision, policy_note="solver_commit_blocked")
            return self._pretool_deny(decision)

        repeat_key, _ = self._repeat_candidate(proposed, next_scope)
        if repeat_key and repeat_key not in self._repeat_reviewed_keys:
            decision = self._review("repeat_exploration", proposed_message=proposed, outputs=None)
            self._repeat_reviewed_keys.add(repeat_key)
            if decision.decision == "BLOCK_REPEAT":
                self.recovery_counts["repeat_blocks"] += 1
                return self._pretool_deny(decision)

        leaving_scope = bool(next_scope and self._active_scope and next_scope != self._active_scope)
        if self._needs_branch_review(leaving_scope=leaving_scope):
            decision = self._review("branch_review", proposed_message=proposed, outputs=None)
            self._reviewed_scope_revision = self._scope_revision
            self._branch_reviewed_keys.add((self._workspace_revision, self._active_scope or ""))
            self._record_scope_review(decision, proposed_scope=next_scope)
            if decision.decision == "PRUNE":
                self.recovery_counts["branch_prunes"] += 1
                if self._active_scope:
                    self._pruned_scopes.append(self._active_scope)
                return self._pretool_deny(decision)
            self.recovery_counts["branch_stays"] += 1

        if is_mutation and self.recovery_gate_mutations:
            if self.recovery_counts["action_blocks"] < self.recovery_max_action_blocks:
                decision = self._review("before_action", proposed_message=proposed, outputs=None)
                if decision.decision == "BLOCK_ACTION":
                    self.recovery_counts["action_blocks"] += 1
                    return self._pretool_deny(decision)
            else:
                self._record_synthetic_allow(
                    "Action block limit reached; allowing the proposed mutation without another semantic block."
                )

        tool_use_id = str(payload.get("tool_use_id") or "")
        self._pending[tool_use_id] = {
            "proposed": proposed,
            "next_scope": next_scope,
            "is_mutation": is_mutation,
            "workspace_before": self._workspace_snapshot() if is_mutation else "",
        }
        return None

    def _handle_post_tool(self, payload: dict[str, Any]) -> None:
        tool_use_id = str(payload.get("tool_use_id") or "")
        pending = self._pending.pop(tool_use_id, None)
        proposed = (pending or {}).get("proposed") or self._proposed_message(payload)
        next_scope = (pending or {}).get("next_scope") or self._proposed_scope(proposed)
        is_mutation = bool((pending or {}).get("is_mutation", self._is_mutation(proposed)))
        self.messages.append(proposed)
        self.messages.append(
            {
                "role": "tool",
                "type": str(payload.get("tool_name") or "tool"),
                "tool_call_id": tool_use_id,
                "output": self._tool_response_text(payload),
            }
        )
        if is_mutation:
            before = str((pending or {}).get("workspace_before") or "")
            if self._workspace_snapshot() != before:
                self._record_mutation(proposed)
            else:
                self._record_synthetic_allow(
                    "Proposed mutation produced no workspace change; recovery revision unchanged."
                )
        self._record_exploration(proposed, next_scope)

    def _handle_stop(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        self.n_calls += 1
        last_message = str(payload.get("last_assistant_message") or "").strip()
        stop_key = (str(payload.get("turn_id") or ""), last_message)
        if last_message and stop_key not in self._seen_stop_messages:
            self.messages.append({"role": "assistant", "content": last_message})
            self._seen_stop_messages.add(stop_key)
        if not self.recovery_gate_submission:
            return None
        if self.recovery_counts["submission_blocks"] >= self.recovery_max_submission_blocks:
            self._record_synthetic_allow("Submission block limit reached; allowing Codex to stop.")
            return None

        workspace = self._workspace_snapshot()
        if not workspace.strip():
            decision = self._empty_patch_decision()
            self.recovery_counts["submission_blocks"] += 1
            self._record_event(decision, policy_note="empty_submission_artifact")
            self._record_submission_block(decision)
            return self._stop_block(decision)

        decision = self._review("before_submission", proposed_message=None, outputs=None)
        if decision.decision == "BLOCK_SUBMISSION":
            self.recovery_counts["submission_blocks"] += 1
            self._record_submission_block(decision)
            return self._stop_block(decision)
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "active",
            "adapter": "codex-native-hooks-v1",
            "events": list(self.recovery_events),
            "state": self._review_state(),
            "counts": dict(self.recovery_counts),
            "hook_event_counts": dict(self._hook_event_counts),
            "handshake_complete": self._session_start_seen.is_set(),
            "reviewer_model": getattr(self.recovery_policy, "model_name", None),
            "reviewer_calls": getattr(self.recovery_policy, "n_calls", None),
            "reviewer_cost": getattr(self.recovery_policy, "cost", None),
        }


class RecoveryHookService:
    """Serve synchronous hook requests over a bind-mounted filesystem."""

    def __init__(
        self,
        ipc_root: Path,
        controller: CodexHookRecoveryController,
        *,
        fail_open: bool,
    ) -> None:
        self.ipc_root = ipc_root
        self.controller = controller
        self.fail_open = fail_open
        self.requests = ipc_root / "requests"
        self.responses = ipc_root / "responses"
        self.requests.mkdir(parents=True, exist_ok=True)
        self.responses.mkdir(parents=True, exist_ok=True)
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="codex-recovery-hooks", daemon=True)

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set() or any(self.requests.glob("*.json")):
            found = False
            for request_path in sorted(self.requests.glob("*.json")):
                found = True
                response_path = self.responses / request_path.name
                try:
                    payload = json.loads(request_path.read_text(encoding="utf-8"))
                    hook_output = self.controller.handle(payload)
                    response = {"hook_output": hook_output}
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    self.errors.append(message)
                    if self.fail_open:
                        response = {"hook_output": None, "failed_open": message}
                    else:
                        event = ""
                        try:
                            event = str(payload.get("hook_event_name") or "")
                        except Exception:
                            pass
                        if event == "PreToolUse":
                            response = {
                                "hook_output": {
                                    "hookSpecificOutput": {
                                        "hookEventName": "PreToolUse",
                                        "permissionDecision": "deny",
                                        "permissionDecisionReason": f"Recovery controller failed closed: {message}",
                                    }
                                }
                            }
                        elif event == "Stop":
                            response = {
                                "hook_output": {
                                    "decision": "block",
                                    "reason": f"Recovery controller failed closed: {message}",
                                }
                            }
                        else:
                            response = {"hook_output": None, "error": message}
                self._atomic_json(response_path, response)
                request_path.unlink(missing_ok=True)
            if not found:
                time.sleep(0.01)
