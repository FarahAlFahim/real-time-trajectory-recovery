"""Opt-in, real-time trajectory recovery for mini-swe-agent.

The solver's native message history is never rewritten. Recovery reviews a
completed exploration scope when the solver is about to leave it, edit, or
submit. It can also interrupt demonstrably repetitive exploration and can gate
submission a small, configurable number of times.
"""

from __future__ import annotations

import copy
import hashlib
import re
import shlex
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Literal

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments import Environment
from minisweagent.models import Model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager

from src.recovery.decision import RecoveryDecision, RecoveryPhase, RecoveryPolicy
from src.recovery.policy import LiteLLMRecoveryPolicy


class RecoveryAgent(DefaultAgent):
    """DefaultAgent with scope-aware, real-time trajectory recovery."""

    _SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    _DIRECT_MUTATION_PATTERNS = (
        re.compile(r"\bapply_patch\b"),
        re.compile(r"(?:^|[;&|]\s*)(?:git\s+apply|patch)\s"),
        re.compile(r"\b(?:sed|perl|ruby)\s+[^\n]*(?:-i|--in-place)\b"),
        re.compile(r"\bgit\s+(?:checkout|restore|reset|clean)\b"),
    )
    _SOURCE_SUFFIXES = {
        ".py", ".pyx", ".pxd", ".c", ".cc", ".cpp", ".h", ".hpp", ".java", ".js", ".jsx",
        ".ts", ".tsx", ".rs", ".go", ".rb", ".php", ".sql", ".html", ".css", ".xml", ".rst",
        ".toml", ".ini", ".cfg", ".yaml", ".yml",
    }
    _EXPLORATION_COMMAND = re.compile(
        r"(?:^|[;&|]\s*)(?:cat|sed|head|tail|nl|less|more|rg|grep|find|ls|git\s+(?:show|diff|grep))\b"
    )
    _PATH_TOKEN = re.compile(
        r"(?<![\w])((?:/testbed/|\./|\.\./)?[A-Za-z0-9_.@+\-]+(?:/[A-Za-z0-9_.@+\-]+)+"
        r"|[A-Za-z0-9_.@+\-]+\.(?:py|pyx|pxd|c|cc|cpp|h|hpp|java|js|jsx|ts|tsx|rs|go|rb|php|sql|"
        r"html|css|xml|json|ya?ml|toml|ini|cfg|rst|md|txt))"
    )

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        recovery_policy: RecoveryPolicy,
        recovery_mode: Literal["active", "shadow"] = "active",
        recovery_max_submission_blocks: int = 2,
        recovery_max_action_blocks: int = 3,
        recovery_repeat_threshold: int = 2,
        recovery_fail_open: bool = True,
        recovery_review_branches: bool = True,
        recovery_gate_mutations: bool = True,
        recovery_gate_submission: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, env, **kwargs)
        if recovery_mode not in {"active", "shadow"}:
            raise ValueError("recovery_mode must be 'active' or 'shadow'")
        self.recovery_policy = recovery_policy
        self.recovery_mode = recovery_mode
        self.recovery_max_submission_blocks = max(0, recovery_max_submission_blocks)
        self.recovery_max_action_blocks = max(0, recovery_max_action_blocks)
        self.recovery_repeat_threshold = max(1, recovery_repeat_threshold)
        self.recovery_fail_open = recovery_fail_open
        self.recovery_review_branches = recovery_review_branches
        self.recovery_gate_mutations = recovery_gate_mutations
        self.recovery_gate_submission = recovery_gate_submission

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
        self._task = ""
        self._base_commit: str | None = None
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

    def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
        self._task = task
        self._base_commit = self._resolve_base_commit()
        if self.recovery_mode == "active":
            task += (
                "\n\n[Real-time trajectory recovery is active. Explore a file or method before judging it. "
                "Continue within the same scope when useful. When recovery reports that evidence was already checked, "
                "either explain the materially new question or move to the nearest unresolved repair obligation. "
                "If a submission concern is irrelevant after investigation, state why with repository/test evidence "
                "before resubmitting.]"
            )
        return super().run(task, **kwargs)

    def step(self) -> list[dict[str, Any]]:
        proposed = self.query()
        execution_message = proposed
        next_scope = self._proposed_scope(proposed)
        is_submission = self._is_submission(proposed)
        is_mutation = self._is_mutation(proposed)

        if self.recovery_mode == "active" and self._contains_git_commit(proposed):
            decision = RecoveryDecision(
                phase="before_action",
                decision="BLOCK_ACTION",
                reason=(
                    "The solver attempted to create a git commit. Commits change HEAD and can hide the repair from "
                    "the required patch artifact; the task requires leaving source changes uncommitted."
                ),
                evidence=["Proposed action contains git commit."],
                next_focus=(
                    "Keep the existing source changes uncommitted, inspect the diff against the initial repository "
                    "revision, and continue testing or submit that diff."
                ),
                unresolved_obligations=["Preserve the repair as an uncommitted patch."],
                owning_handling_points=["git commit action"],
                behavior_decision="UNRESOLVED",
                blocker_id="solver-commit-prohibited",
            )
            self._record_deterministic_decision(
                decision,
                policy_note="solver_commit_blocked",
            )
            return self._block(proposed, decision)

        repeat_key, repeat_count = self._repeat_candidate(proposed, next_scope)
        if repeat_key and repeat_key not in self._repeat_reviewed_keys:
            decision = self._review("repeat_exploration", proposed_message=proposed, outputs=None)
            self._repeat_reviewed_keys.add(repeat_key)
            if decision.decision == "BLOCK_REPEAT":
                self.recovery_counts["repeat_blocks"] += 1
                return self._block(proposed, decision)

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
                return self._block(proposed, decision)
            self.recovery_counts["branch_stays"] += 1

        if is_submission and self.recovery_gate_submission:
            # Artifact integrity is deterministic bookkeeping, not a semantic
            # rejection. Always refresh it, even after the semantic block
            # budget is exhausted, so a final staged repair cannot disappear.
            # Keep the native solver message intact for trajectory/reviewer
            # provenance while normalizing only the action that will execute.
            execution_message = copy.deepcopy(proposed)
            self._prepare_safe_submission_artifacts(execution_message)
            artifact_decision = self._refresh_submission_patch_artifact(proposed)
            if self.recovery_counts["submission_blocks"] < self.recovery_max_submission_blocks:
                if artifact_decision is not None:
                    self.recovery_counts["submission_blocks"] += 1
                    self._record_deterministic_decision(
                        artifact_decision,
                        policy_note="empty_submission_artifact",
                    )
                    self._record_submission_block(artifact_decision)
                    return self._block(proposed, artifact_decision, submission_feedback=True)
                decision = self._review("before_submission", proposed_message=proposed, outputs=None)
                if decision.decision == "BLOCK_SUBMISSION":
                    self.recovery_counts["submission_blocks"] += 1
                    self._record_submission_block(decision)
                    return self._block(proposed, decision, submission_feedback=True)
            else:
                self._record_synthetic_allow("Submission block limit reached; allowing solver submission.")
        elif is_mutation and self.recovery_gate_mutations:
            if self.recovery_counts["action_blocks"] < self.recovery_max_action_blocks:
                decision = self._review("before_action", proposed_message=proposed, outputs=None)
                if decision.decision == "BLOCK_ACTION":
                    self.recovery_counts["action_blocks"] += 1
                    return self._block(proposed, decision)
            else:
                self._record_synthetic_allow("Action block limit reached; allowing proposed mutation without review.")

        workspace_before = self._workspace_snapshot() if is_mutation else ""
        observations = self.execute_actions(execution_message)
        if is_mutation:
            workspace_after = self._workspace_snapshot()
            if workspace_after != workspace_before or (
                self._base_commit is None
                and not workspace_before.strip()
                and not workspace_after.strip()
            ):
                self._record_mutation(proposed)
            else:
                self._record_synthetic_allow(
                    "Proposed mutation produced no workspace change; recovery revision unchanged."
                )
        self._record_exploration(proposed, next_scope)
        return observations

    def _review(
        self,
        phase: RecoveryPhase,
        *,
        proposed_message: dict[str, Any] | None,
        outputs: list[dict[str, Any]] | None,
    ) -> RecoveryDecision:
        workspace = self._workspace_snapshot()
        review_messages = self.messages[self._review_message_start :]
        review_state = self._review_state(
            for_reviewer=True,
            proposed_message=proposed_message,
            workspace=workspace,
        )
        semantic_risk_context = LiteLLMRecoveryPolicy._semantic_risk_context(
            review_messages,
            proposed_message,
            workspace,
        )
        # Submission reviews reconcile constraints discovered during
        # exploration/mutation review. They must not manufacture new hard
        # constraints from incidental comparisons in the final diff context.
        if phase != "before_submission":
            self._remember_semantic_constraints(semantic_risk_context)
        semantic_risk_context = self._semantic_context_for_review(
            semantic_risk_context,
            proposed_message=proposed_message,
            workspace=workspace,
            include_local_constraints=phase != "before_submission",
        )
        review_state["current_semantic_risk_context"] = semantic_risk_context
        review_state["canonical_semantic_constraints"] = self._canonical_semantic_constraints(
            proposed_message=proposed_message,
            workspace=workspace,
        )
        try:
            decision = self.recovery_policy.review(
                phase=phase,
                task=self._task,
                messages=review_messages,
                proposed_message=proposed_message,
                outputs=outputs,
                workspace=workspace,
                state=review_state,
            )
        except Exception as exc:
            if not self.recovery_fail_open:
                raise
            decision = RecoveryDecision(
                phase=phase,
                decision=self._default_decision(phase),
                reason=f"Recovery reviewer failed open: {type(exc).__name__}: {exc}",
                failed_open=True,
            )
        self._review_message_start = len(self.messages)

        if self.recovery_mode == "active" and phase in {"before_action", "before_submission"}:
            decision = self._require_semantic_risk_acknowledgement(
                decision,
                phase=phase,
                proposed_message=proposed_message,
                workspace=workspace,
                risk=semantic_risk_context,
            )

        recommended = decision.decision
        effective = recommended
        policy_note = ""
        if self.recovery_mode == "shadow":
            effective = self._default_decision(phase)
            policy_note = "shadow_mode"
        elif phase == "before_action" and recommended == "BLOCK_ACTION":
            if not self._valid_action_block(decision):
                effective = "CONTINUE"
                policy_note = "insufficient_evidence_for_action_block"
        elif phase == "before_submission" and recommended == "BLOCK_SUBMISSION":
            if not self._valid_hard_submission_block(decision):
                effective = "ALLOW_SUBMISSION"
                policy_note = "insufficient_evidence_for_hard_block"

        if effective != recommended:
            decision = RecoveryDecision(**{**asdict(decision), "decision": effective})
        self._update_recovery_state(decision)
        self.recovery_events.append(
            {
                "step": self.n_calls,
                "phase": phase,
                "mode": self.recovery_mode,
                "active_scope": self._active_scope,
                "recommended_decision": recommended,
                "effective_decision": effective,
                "relevance_label": (
                    "IRRELEVANT"
                    if phase == "branch_review" and recommended == "PRUNE"
                    else "RELEVANT" if phase == "branch_review" else ""
                ),
                "review_context_digest": dict(
                    review_state.get("evidence_ledger_digest", {})
                ),
                "policy_note": policy_note,
                **asdict(decision),
            }
        )
        return decision

    def _require_semantic_risk_acknowledgement(
        self,
        decision: RecoveryDecision,
        *,
        phase: RecoveryPhase,
        proposed_message: dict[str, Any] | None,
        workspace: str,
        risk: dict[str, Any],
    ) -> RecoveryDecision:
        allow_decision = "CONTINUE" if phase == "before_action" else "ALLOW_SUBMISSION"
        if decision.decision != allow_decision:
            return decision
        consumers = risk["observed_exact_value_consumers"]
        audit_text = "\n".join(
            [
                decision.reason,
                *decision.evidence,
                *decision.demonstrated_paths,
                *decision.owning_handling_points,
                *decision.rejected_alternatives,
            ]
        )

        def concretely_validated(snippet: str) -> bool:
            tokens = re.findall(
                r"[A-Za-z_][A-Za-z0-9_]*|-?\d+|['\"][^'\"]+['\"]",
                snippet,
            )
            significant = [
                token.strip("'\"").casefold()
                for token in tokens
                if token.casefold() not in {"if", "is", "none", "true", "false"}
            ]
            audit_folded = audit_text.casefold()
            mentions_constraint = bool(significant) and all(
                token in audit_folded for token in significant
            )
            passing_validation = bool(
                re.search(
                    r"(?:pytest|test|repro(?:duction)?|command|check).{0,160}"
                    r"(?:pass(?:ed|es)?|succeed(?:ed|s)?|returncode\s*[:=]?\s*0|exit(?: code)?\s*[:=]?\s*0)"
                    r"|(?:pass(?:ed|es)?|succeed(?:ed|s)?|returncode\s*[:=]?\s*0|exit(?: code)?\s*[:=]?\s*0)"
                    r".{0,160}(?:pytest|test|repro(?:duction)?|command|check)",
                    audit_text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            )
            return mentions_constraint and passing_validation

        def guard_key(kind: str, snippet: str) -> str:
            normalized = re.sub(r"\s+", " ", snippet).strip().casefold()
            digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            return f"{self._workspace_revision}:{kind}:{digest}"

        def addressed_by_proposed_action(snippet: str) -> bool:
            if not proposed_message:
                return False
            code = snippet.strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\.", code):
                code = code.split(".", 1)[1]
            normalized = re.sub(r"\s+", "", code).casefold()
            for command in self._commands(proposed_message):
                removed = False
                retained = False
                for line in command.splitlines():
                    if line.startswith(("+++", "---", "@@")) or not line.startswith(("+", "-")):
                        continue
                    line_text = re.sub(r"\s+", "", line[1:]).casefold()
                    if normalized not in line_text:
                        continue
                    if line.startswith("-"):
                        removed = True
                    else:
                        retained = True
                if removed and not retained:
                    return True
                compact_command = re.sub(r"\s+", "", command).casefold()
                if normalized in compact_command and re.search(
                    r"\.(?:replace|sub)\s*\(|\bre\.sub\s*\(", command
                ):
                    return True
            return False

        matching_key = guard_key("matching", consumers[0]) if consumers else ""
        guard_once = phase == "before_action"
        missing_kind = ""
        missing_evidence = ""
        if (
            risk["broad_matching_or_normalization_change"]
            and consumers
            and not concretely_validated(consumers[0])
            and not addressed_by_proposed_action(consumers[0])
            and (not guard_once or matching_key not in self._semantic_guarded_scopes)
        ):
            missing_kind = "exact-value consumer"
            missing_evidence = consumers[0]
        if not missing_evidence:
            if guard_once and risk["broad_matching_or_normalization_change"] and consumers:
                self._semantic_guarded_scopes.add(matching_key)
            return decision
        # Give one deterministic correction for this unchanged workspace and
        # scope. Further blocking requires a fresh, concrete reviewer finding,
        # preventing the guard itself from creating a loop.
        if guard_once:
            self._semantic_guarded_scopes.add(matching_key)
        blocking_decision = "BLOCK_ACTION" if phase == "before_action" else "BLOCK_SUBMISSION"
        return RecoveryDecision(
            phase=phase,
            decision=blocking_decision,
            reason=(
                f"The proposed change has one nearest unresolved {missing_kind}: {missing_evidence}"
            ),
            evidence=[missing_evidence],
            next_focus=(
                f"Inspect this exact handling point: {missing_evidence}. Update it for the proposed behavior, or "
                "explain with repository/test evidence why it is unaffected; do not survey unrelated variants."
            ),
            retained_findings=decision.retained_findings,
            unresolved_obligations=[
                f"Resolve or refute the concrete {missing_kind}: {missing_evidence}"
            ],
            behavioral_specification=decision.behavioral_specification,
            demonstrated_paths=[
                "Proposed semantic change -> observed dependent constraint -> downstream or paired operation"
            ],
            owning_handling_points=[missing_evidence],
            rejected_alternatives=decision.rejected_alternatives,
            branch_finding=decision.branch_finding,
            behavior_decision="UNRESOLVED",
            blocker_id="unreviewed-semantic-dependencies",
            investigation_complete=False,
            raw={
                **decision.raw,
                "recovery_guard": "semantic_risk_acknowledgement_required",
                "reviewer_allow_decision": allow_decision,
            },
        )

    @staticmethod
    def _default_decision(phase: RecoveryPhase) -> str:
        return {
            "branch_review": "KEEP",
            "repeat_exploration": "ALLOW_REPEAT",
            "before_action": "CONTINUE",
            "before_submission": "ALLOW_SUBMISSION",
        }[phase]

    @staticmethod
    def _valid_action_block(decision: RecoveryDecision) -> bool:
        handling_evidence = decision.demonstrated_paths or decision.owning_handling_points
        block_text = "\n".join(
            [decision.reason, decision.next_focus, *decision.unresolved_obligations]
        )
        requests_validation = bool(
            re.search(
                r"\b(?:runtime )?(?:validation|verification)\b|\b(?:validate|verify|run)\b.{0,80}\b(?:test|repro)",
                block_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        demonstrated_failure = bool(
            re.search(
                r"\b(?:failed|failure|traceback|regression|incorrect output|output mismatch)\b|"
                r"returncode\s*[:=]?\s*[1-9]\d*|exit(?: code)?\s*[:=]?\s*[1-9]\d*",
                "\n".join([decision.reason, *decision.evidence]),
                flags=re.IGNORECASE,
            )
        )
        return bool(
            decision.reason.strip()
            and decision.evidence
            and decision.next_focus.strip()
            and (decision.unresolved_obligations or handling_evidence)
            and not RecoveryAgent._requests_exhaustive_audit(decision)
            and not (requests_validation and not demonstrated_failure)
        )

    @staticmethod
    def _valid_hard_submission_block(decision: RecoveryDecision) -> bool:
        handling_evidence = decision.demonstrated_paths or decision.owning_handling_points
        return bool(
            decision.blocker_id.strip()
            and decision.reason.strip()
            and decision.evidence
            and handling_evidence
            and decision.unresolved_obligations
            and decision.next_focus.strip()
            and not RecoveryAgent._requests_exhaustive_audit(decision)
        )

    @staticmethod
    def _requests_exhaustive_audit(decision: RecoveryDecision) -> bool:
        text = "\n".join(
            [
                decision.reason,
                decision.next_focus,
                *decision.unresolved_obligations,
            ]
        )
        return bool(
            re.search(
                r"\b(?:every code path|all (?:code )?paths?|all .* variants?|"
                r"enumerate (?:every|all)|reconcile every|exhaustive (?:audit|survey))\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _update_recovery_state(self, decision: RecoveryDecision) -> None:
        if decision.evidence or decision.branch_finding:
            self._evidence_ledger.append(
                {
                    "step": self.n_calls,
                    "phase": decision.phase,
                    "scope": self._active_scope,
                    "scope_revision": self._scope_revision,
                    "decision": decision.decision,
                    "evidence": list(decision.evidence),
                    "branch_finding": decision.branch_finding,
                }
            )
        self.recovery_state = {
            "retained_findings": decision.retained_findings,
            "unresolved_obligations": decision.unresolved_obligations,
            "behavioral_specification": decision.behavioral_specification,
            "demonstrated_paths": decision.demonstrated_paths,
            "owning_handling_points": decision.owning_handling_points,
            "rejected_alternatives": decision.rejected_alternatives,
            "branch_finding": decision.branch_finding,
            "behavior_decision": decision.behavior_decision,
            "next_focus": decision.next_focus,
            "investigation_complete": decision.investigation_complete,
        }

    def _evidence_for_review(
        self,
        proposed_message: dict[str, Any] | None,
        workspace: str,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        touched_scopes = set(self._source_paths(workspace))
        if proposed_message:
            for command in self._commands(proposed_message):
                touched_scopes.update(self._source_paths(command))
        active_scopes = set((self._active_scope or "").split(" | ")) - {""}
        relevant_scopes = touched_scopes | active_scopes
        irrelevant_scopes = set(self._pruned_scopes) - relevant_scopes

        non_irrelevant = [
            dict(entry)
            for entry in self._evidence_ledger
            if entry.get("scope") not in irrelevant_scopes
        ]
        eligible = []
        for entry in non_irrelevant:
            scope = str(entry.get("scope") or "")
            text = "\n".join(
                [
                    scope,
                    str(entry.get("branch_finding") or ""),
                    *(str(item) for item in entry.get("evidence", [])),
                ]
            )
            if (
                not relevant_scopes
                or scope in relevant_scopes
                or any(path and path in text for path in relevant_scopes)
            ):
                eligible.append(entry)

        # The full append-only ledger remains in the serialized trajectory for
        # audit. Online reviews receive only the latest canonical state for a
        # scope, preventing repeated reviewer paraphrases from becoming input
        # evidence on every subsequent call.
        latest_by_scope: dict[str, dict[str, Any]] = {}
        for entry in eligible:
            scope = str(entry.get("scope") or "<unknown-scope>")
            previous = latest_by_scope.get(scope)
            if previous is None or int(entry.get("step") or 0) >= int(previous.get("step") or 0):
                latest_by_scope[scope] = entry
        candidates = list(latest_by_scope.values())

        def relevance(entry: dict[str, Any]) -> tuple[bool, bool, bool, int]:
            scope = str(entry.get("scope") or "")
            text = "\n".join(
                [
                    scope,
                    str(entry.get("branch_finding") or ""),
                    *(str(item) for item in entry.get("evidence", [])),
                ]
            )
            return (
                scope in touched_scopes,
                scope in active_scopes,
                any(path and path in text for path in relevant_scopes),
                int(entry.get("step") or 0),
            )

        candidates.sort(key=relevance, reverse=True)
        return candidates, {
            "total_entries": len(self._evidence_ledger),
            "relevant_entries": len(eligible),
            "excluded_irrelevant_entries": len(self._evidence_ledger) - len(non_irrelevant),
            "excluded_nonlocal_entries": len(non_irrelevant) - len(eligible),
            "superseded_relevant_entries": len(eligible) - len(candidates),
            "included_entries": len(candidates),
        }

    def _review_state(
        self,
        *,
        for_reviewer: bool = False,
        proposed_message: dict[str, Any] | None = None,
        workspace: str = "",
    ) -> dict[str, Any]:
        since_block: list[dict[str, Any]] = []
        if self._last_submission_block_message_index is not None:
            since_block = self._summarize_solver_response(
                self.messages[self._last_submission_block_message_index :]
            )
        recent_commands = sorted(
            self._exploration_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:20]
        if for_reviewer:
            evidence_ledger, evidence_digest = self._evidence_for_review(
                proposed_message,
                workspace,
            )
        else:
            evidence_ledger = [dict(item) for item in self._evidence_ledger]
            evidence_digest = {
                "total_entries": len(evidence_ledger),
                "relevant_entries": len(evidence_ledger),
                "excluded_irrelevant_entries": 0,
                "included_entries": len(evidence_ledger),
            }
        return {
            **self.recovery_state,
            "active_scope": self._active_scope,
            "active_scope_revision": self._scope_revision,
            "workspace_revision": self._workspace_revision,
            "scope_history": [dict(item) for item in self._scope_history[-5:]],
            "irrelevant_scopes": list(self._pruned_scopes[-20:]),
            # Backward-compatible alias for earlier trajectory consumers.
            "pruned_scopes": list(self._pruned_scopes[-20:]),
            "evidence_ledger": evidence_ledger,
            "evidence_ledger_digest": evidence_digest,
            "canonical_semantic_constraints": (
                self._canonical_semantic_constraints(
                    proposed_message=proposed_message,
                    workspace=workspace,
                )
                if for_reviewer
                else sorted(
                    (dict(item) for item in self._semantic_constraints.values()),
                    key=lambda item: (
                        str(item.get("scope")),
                        str(item.get("kind")),
                        int(item.get("priority", 0)),
                    ),
                )
            ),
            "exploration_ledger": {
                "scope_counts": dict(self._scope_counts),
                "most_repeated_commands": recent_commands,
                "mutation_signature_counts": dict(self._mutation_signature_counts),
                "mutation_scope_counts": dict(self._mutation_scope_counts),
                "pending_post_mutation_validation": sorted(
                    self._post_mutation_validation_scopes
                ),
                "since_last_mutation": {
                    "scope_counts": dict(self._revision_scope_counts),
                    "command_counts": dict(self._revision_exploration_counts),
                },
            },
            "submission_history": [dict(item) for item in self._submission_history],
            "solver_response_since_last_block": since_block,
            "control_counts": dict(self.recovery_counts),
            "submission_block_limit": self.recovery_max_submission_blocks,
        }

    def _review_scopes(
        self,
        proposed_message: dict[str, Any] | None,
        workspace: str,
    ) -> tuple[set[str], set[str]]:
        touched_scopes = set(self._source_paths(workspace))
        if proposed_message:
            for command in self._commands(proposed_message):
                touched_scopes.update(self._source_paths(command))
        active_scopes = set((self._active_scope or "").split(" | ")) - {""}
        return touched_scopes, active_scopes

    def _remember_semantic_constraints(self, risk: dict[str, Any]) -> None:
        """Retain exact code constraints without retaining reviewer paraphrases."""
        fields = {
            "exact_value_consumer": "observed_exact_value_consumers",
            "dependent_offset": "observed_dependent_offsets",
            "dependent_index": "observed_dependent_indices",
        }
        scopes = (self._active_scope or "<unknown-scope>").split(" | ")
        for kind, field_name in fields.items():
            for priority, raw_snippet in enumerate(risk.get(field_name, [])):
                snippet = str(raw_snippet).strip()
                if not snippet:
                    continue
                normalized = re.sub(r"\s+", " ", snippet).strip().casefold()
                for scope in scopes:
                    key = (kind, scope, normalized)
                    previous = self._semantic_constraints.get(key, {})
                    self._semantic_constraints[key] = {
                        "kind": kind,
                        "scope": scope,
                        "snippet": snippet,
                        "priority": min(priority, int(previous.get("priority", priority))),
                        "first_step": int(previous.get("first_step", self.n_calls)),
                        "last_step": self.n_calls,
                    }

    def _canonical_semantic_constraints(
        self,
        *,
        proposed_message: dict[str, Any] | None,
        workspace: str,
    ) -> list[dict[str, Any]]:
        touched_scopes, active_scopes = self._review_scopes(proposed_message, workspace)
        relevant_scopes = touched_scopes | active_scopes
        constraints = [
            dict(item)
            for item in self._semantic_constraints.values()
            if not relevant_scopes or item.get("scope") in relevant_scopes
            if not self._constraint_removed_by_workspace_diff(
                str(item.get("snippet", "")), workspace
            )
        ]
        constraints.sort(
            key=lambda item: (
                0 if item.get("scope") in touched_scopes else 1,
                0 if item.get("scope") in active_scopes else 1,
                int(item.get("priority", 0)),
                -int(item.get("last_step", 0)),
            )
        )
        return constraints

    @staticmethod
    def _constraint_removed_by_workspace_diff(snippet: str, workspace: str) -> bool:
        """Return whether an exact code constraint was removed and not retained."""
        code = snippet.strip()
        # Evidence is commonly qualified with its nearest class owner, e.g.
        # ``SimpleData.start_line = 3`` while the diff contains only the code.
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\.", code):
            code = code.split(".", 1)[1]
        normalized = re.sub(r"\s+", "", code).casefold()
        if not normalized:
            return False
        removed = False
        retained = False
        for line in workspace.splitlines():
            if line.startswith(("+++", "---", "diff ", "@@")):
                continue
            if not line.startswith(("+", "-", " ")):
                continue
            line_text = re.sub(r"\s+", "", line[1:]).casefold()
            if normalized not in line_text:
                continue
            if line.startswith("-"):
                removed = True
            else:
                retained = True
        return removed and not retained

    def _semantic_context_for_review(
        self,
        local_risk: dict[str, Any],
        *,
        proposed_message: dict[str, Any] | None,
        workspace: str,
        include_local_constraints: bool = True,
    ) -> dict[str, Any]:
        """Merge the current delta with exact, scope-relevant code constraints."""
        merged = copy.deepcopy(local_risk)
        fields = {
            "exact_value_consumer": "observed_exact_value_consumers",
            "dependent_offset": "observed_dependent_offsets",
            "dependent_index": "observed_dependent_indices",
        }
        constraints = self._canonical_semantic_constraints(
            proposed_message=proposed_message,
            workspace=workspace,
        )
        for kind, field_name in fields.items():
            snippets = [
                str(item["snippet"])
                for item in constraints
                if item.get("kind") == kind
                and not self._constraint_removed_by_workspace_diff(
                    str(item["snippet"]), workspace
                )
            ]
            if include_local_constraints:
                for snippet in local_risk.get(field_name, []):
                    if (
                        snippet not in snippets
                        and not self._constraint_removed_by_workspace_diff(snippet, workspace)
                    ):
                        snippets.append(snippet)
            merged[field_name] = snippets
        proposed_text = workspace
        if proposed_message:
            proposed_text += "\n" + str(proposed_message.get("content") or "")
            proposed_text += "\n" + "\n".join(self._commands(proposed_message))
        mentions_variable_count = bool(
            re.search(
                r"(?<![A-Za-z0-9])(?:len|count|rows?|items?|elements?|arity|size)(?![A-Za-z0-9])",
                proposed_text,
                flags=re.IGNORECASE,
            )
        )
        if mentions_variable_count and (
            merged["observed_dependent_offsets"] or merged["observed_dependent_indices"]
        ):
            merged["variable_cardinality_or_layout_change"] = True
            merged["required_cardinality_check"] = (
                "Check the nearest observed dependent index, offset, boundary, or paired read/write operation that "
                "may be affected by the proposed change. Trace whether it lies on the original report's "
                "producer-to-output path. If traced, reconcile or validate it before submission; otherwise keep it "
                "advisory and record the concrete refutation."
            )
        return merged

    @staticmethod
    def _semantic_messages_for_review(
        recent_messages: list[dict[str, Any]],
        evidence_ledger: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return only the current delta; exact cumulative constraints are stored separately."""
        del evidence_ledger
        return list(recent_messages)

    @classmethod
    def _summarize_solver_response(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for message in messages[-20:]:
            if message.get("role") != "assistant":
                continue
            item: dict[str, Any] = {"role": "assistant"}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                item["thought"] = content
            commands = cls._commands(message)
            if commands:
                item["actions"] = [
                    command
                    if len(command) <= 2000
                    else f"[large command omitted; targets: {', '.join(cls._source_paths(command)) or 'none'}]"
                    for command in commands
                ]
            if len(item) > 1:
                summary.append(item)
        return summary

    def _needs_branch_review(self, *, leaving_scope: bool) -> bool:
        review_key = (self._workspace_revision, self._active_scope or "")
        return bool(
            self.recovery_review_branches
            and self._active_scope
            and self._scope_revision > self._reviewed_scope_revision
            and review_key not in self._branch_reviewed_keys
            and leaving_scope
        )

    def _record_scope_review(self, decision: RecoveryDecision, *, proposed_scope: str | None) -> None:
        self._scope_history.append(
            {
                "scope": self._active_scope,
                "revision": self._scope_revision,
                "decision": decision.decision,
                "relevance_label": "IRRELEVANT" if decision.decision == "PRUNE" else "RELEVANT",
                "branch_finding": decision.branch_finding,
                "reason": decision.reason,
                "next_focus": decision.next_focus,
                "proposed_next_scope": proposed_scope,
            }
        )

    def _record_submission_block(self, decision: RecoveryDecision) -> None:
        self._submission_history.append(
            {
                "attempt": len(self._submission_history) + 1,
                "step": self.n_calls,
                "blocker_id": decision.blocker_id,
                "reason": decision.reason,
                "evidence": decision.evidence,
                "unresolved_obligations": decision.unresolved_obligations,
                "next_focus": decision.next_focus,
            }
        )
        self._last_submission_block_message_index = len(self.messages)

    def _record_deterministic_decision(
        self,
        decision: RecoveryDecision,
        *,
        policy_note: str,
    ) -> None:
        """Record a non-LLM recovery invariant using the normal event schema."""
        self._update_recovery_state(decision)
        self.recovery_events.append(
            {
                "step": self.n_calls,
                "phase": decision.phase,
                "mode": self.recovery_mode,
                "active_scope": self._active_scope,
                "recommended_decision": decision.decision,
                "effective_decision": decision.decision,
                "relevance_label": "",
                "review_context_digest": {},
                "policy_note": policy_note,
                **asdict(decision),
            }
        )

    @classmethod
    def _uses_patch_artifact(cls, message: dict[str, Any]) -> bool:
        return any(
            re.search(r"\bcat\s+(?:--\s+)?['\"]?(?:\./)?patch\.txt(?:['\"]|\b)", command)
            for command in cls._commands(message)
        )

    def _resolve_base_commit(self) -> str | None:
        """Capture the immutable repository baseline before solver actions."""
        try:
            result = self.env.execute(
                {"command": "git rev-parse --verify HEAD 2>/dev/null"}
            )
        except Exception:
            return None
        if result.get("returncode") != 0:
            return None
        revision = str(result.get("output") or "").strip().splitlines()[-1:]
        if not revision or not re.fullmatch(r"[0-9a-fA-F]{7,64}", revision[0]):
            return None
        return revision[0]

    def _diff_baseline(self) -> str:
        return self._base_commit or "HEAD"

    def _refresh_submission_patch_artifact(
        self,
        message: dict[str, Any],
    ) -> RecoveryDecision | None:
        """Make patch.txt reflect staged and unstaged source changes before submission."""
        if not self._uses_patch_artifact(message):
            return None
        try:
            repository = self.env.execute(
                {"command": "git rev-parse --is-inside-work-tree >/dev/null 2>&1"}
            )
            if repository.get("returncode") != 0:
                return None
            baseline = self._diff_baseline()
            refreshed = self.env.execute(
                {
                    "command": (
                        f"git diff --no-ext-diff {baseline} -- . ':(exclude)patch.txt' > patch.txt"
                    )
                }
            )
            populated = self.env.execute({"command": "test -s patch.txt"})
        except Exception:
            # Preserve fail-open behavior for environment/tool failures. The
            # semantic reviewer still receives the workspace snapshot.
            return None
        if refreshed.get("returncode") == 0 and populated.get("returncode") == 0:
            self._record_synthetic_allow(
                "Refreshed patch.txt from the initial repository revision so committed, staged, and unstaged "
                "solver changes are submitted."
            )
            return None
        return RecoveryDecision(
            phase="before_submission",
            decision="BLOCK_SUBMISSION",
            reason=(
                "The submission command reads patch.txt, but rebuilding it from the initial repository revision "
                "produced an empty artifact. Submitting now would return no repair."
            ),
            evidence=[
                "patch.txt is empty after including committed, staged, and unstaged changes relative to the initial "
                "repository revision."
            ],
            next_focus=(
                "Restore or implement the intended source change, inspect the diff from the initial repository "
                "revision, and resubmit only after patch.txt contains that diff."
            ),
            unresolved_obligations=["Produce a non-empty patch containing the intended repository change."],
            demonstrated_paths=[
                "workspace source change -> initial-revision diff -> patch.txt -> submission output"
            ],
            owning_handling_points=["patch.txt submission artifact"],
            behavior_decision="UNRESOLVED",
            blocker_id="empty-submission-artifact",
            investigation_complete=False,
            raw={"recovery_guard": "submission_artifact_integrity"},
        )

    def _prepare_safe_submission_artifacts(self, message: dict[str, Any]) -> None:
        marker = re.compile(
            rf"(?:^|[;&|]\s*)echo\s+{re.escape(self._SUBMISSION_MARKER)}\b"
        )
        for action in message.get("extra", {}).get("actions", []):
            if not isinstance(action, dict):
                continue
            command = str(action.get("command", ""))
            match = marker.search(command)
            if not match:
                continue
            prefix = command[: match.start()].rstrip(" ;&|")
            if not prefix:
                continue
            prefix_message = {"extra": {"actions": [{"command": prefix}]}}
            if self._is_mutation(prefix_message):
                continue
            try:
                self.env.execute({"command": prefix})
            except Exception:
                # Artifact preparation is best-effort; the normal solver turn
                # receives the submission block and can regenerate explicitly.
                pass
            if self._uses_patch_artifact(message) and re.search(
                r">\s*['\"]?(?:\./)?patch\.txt(?:['\"]|\b)",
                prefix,
            ):
                # The prefix has already run and patch.txt will be rebuilt from
                # HEAD below. Don't let an unstaged-only git diff overwrite the
                # canonical artifact immediately before cat prints it.
                action["command"] = command[match.start() :].lstrip(";&| ")

    def _record_synthetic_allow(self, reason: str) -> None:
        self.recovery_events.append(
            {
                "step": self.n_calls,
                "phase": "control_policy",
                "mode": self.recovery_mode,
                "recommended_decision": "ALLOW",
                "effective_decision": "ALLOW",
                "reason": reason,
            }
        )

    def _workspace_snapshot(self) -> str:
        baseline = self._diff_baseline()
        command = (
            "git status --short --untracked-files=all 2>/dev/null; "
            f"git diff --no-ext-diff {baseline} -- . ':(exclude)patch.txt' 2>/dev/null"
        )
        try:
            result = self.env.execute({"command": command})
        except Exception as exc:
            return f"Workspace snapshot unavailable: {type(exc).__name__}: {exc}"
        return str(result.get("output", ""))

    @classmethod
    def _commands(cls, message: dict[str, Any]) -> list[str]:
        return [
            str(action["command"])
            for action in message.get("extra", {}).get("actions", [])
            if isinstance(action, dict) and action.get("command")
        ]

    @classmethod
    def _is_submission(cls, message: dict[str, Any]) -> bool:
        return any(cls._SUBMISSION_MARKER in command for command in cls._commands(message))

    @classmethod
    def _contains_git_commit(cls, message: dict[str, Any]) -> bool:
        return any(
            re.search(r"(?:^|[;&|]\s*)git\s+commit\b", command)
            for command in cls._commands(message)
        )

    @classmethod
    def _is_mutation(cls, message: dict[str, Any]) -> bool:
        for command in cls._commands(message):
            if any(pattern.search(command) for pattern in cls._DIRECT_MUTATION_PATTERNS):
                return True
            source_paths = cls._source_paths(command)
            if source_paths and re.search(r"\b(?:write_text|write_bytes|unlink|rename)\s*\(", command):
                return True
            if source_paths and re.search(
                r"\bopen\s*\([^\n]*(?:['\"](?:w|a|x)[+b]?['\"]|mode\s*=)",
                command,
            ):
                return True
            if source_paths and re.search(r"\b(?:tee|truncate|touch|rm|mv|cp)\b", command):
                return True
            if cls._redirects_to_source(command):
                return True
        return False

    @classmethod
    def _source_paths(cls, command: str) -> list[str]:
        paths: list[str] = []
        for match in cls._PATH_TOKEN.finditer(command):
            path = cls._normalize_target(match.group(1))
            if path and any(path.lower().endswith(suffix) for suffix in cls._SOURCE_SUFFIXES):
                paths.append(path)
        return paths

    @classmethod
    def _redirects_to_source(cls, command: str) -> bool:
        scrubbed = re.sub(
            r"<<\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?[^\n]*\n.*?\n\1(?:\n|$)",
            "",
            command,
            flags=re.DOTALL,
        )
        scrubbed = re.sub(r"'(?:[^']|'\"'\"')*'|\"(?:\\.|[^\"])*\"", "", scrubbed, flags=re.DOTALL)
        for match in re.finditer(r"(?<![-<>=])>>?\s*([^\s;&|]+)", scrubbed):
            target = cls._normalize_target(match.group(1))
            if target and any(target.lower().endswith(suffix) for suffix in cls._SOURCE_SUFFIXES):
                return True
        return False

    @staticmethod
    def _has_shell_redirection(command: str) -> bool:
        # Remove heredoc bodies first. Characters such as the closing `>` in a
        # Python named regex group are data, not shell redirection.
        scrubbed = re.sub(
            r"<<\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?[^\n]*\n.*?\n\1(?:\n|$)",
            "",
            command,
            flags=re.DOTALL,
        )
        # Redirection operators inside quoted arguments are data as well.
        scrubbed = re.sub(r"'(?:[^']|'\"'\"')*'|\"(?:\\.|[^\"])*\"", "", scrubbed, flags=re.DOTALL)
        return bool(re.search(r"(?<![-<>=])>>?(?![=>])", scrubbed))

    @classmethod
    def _exploration_targets(cls, message: dict[str, Any]) -> list[str]:
        targets: list[str] = []
        for command in cls._commands(message):
            is_python_inspection = bool(re.search(r"(?:^|[;&|]\s*)python(?:3)?\b", command)) and not cls._is_mutation(message)
            is_patch_packaging = bool(
                re.search(r"\bgit\s+(?:add|status)\b", command)
                or (re.search(r"\bgit\s+diff\b", command) and re.search(r">\s*patch\.txt\b", command))
            )
            if (
                not cls._EXPLORATION_COMMAND.search(command)
                and not is_python_inspection
            ) or cls._SUBMISSION_MARKER in command or is_patch_packaging:
                continue
            if is_python_inspection:
                candidates = [
                    match.group(1)
                    for match in re.finditer(r"\b(?:Path|open)\s*\(\s*['\"]([^'\"]+)['\"]", command)
                ]
            else:
                try:
                    shell_tokens = shlex.split(command.splitlines()[0], comments=False, posix=True)
                except ValueError:
                    shell_tokens = command.splitlines()[0].split()
                candidates = []
                for token in shell_tokens:
                    match = cls._PATH_TOKEN.fullmatch(token.strip("'\"`()[]{}:,;"))
                    if match:
                        candidates.append(match.group(1))
            for candidate in candidates:
                target = cls._normalize_target(candidate)
                if (
                    target
                    and cls._is_reviewable_target(target)
                    and not target.endswith((".bak", ".orig", ".rej"))
                    and target not in targets
                ):
                    targets.append(target)
        return targets

    @classmethod
    def _is_reviewable_target(cls, target: str) -> bool:
        """Return whether a target is a concrete file rather than a directory scope."""
        lowered = target.lower()
        return (
            any(lowered.endswith(suffix) for suffix in cls._SOURCE_SUFFIXES)
            or target.rsplit("/", 1)[-1] in {"Dockerfile", "Makefile"}
        )

    @staticmethod
    def _normalize_target(target: str) -> str:
        target = target.strip("'\"`()[]{}:,;")
        target = re.sub(r"^/testbed/", "", target)
        target = re.sub(r"^(?:\./)+", "", target)
        if target in {"", ".", "..", "/dev/null", "patch.txt"}:
            return ""
        return target

    @classmethod
    def _command_key(cls, message: dict[str, Any]) -> str:
        commands = [re.sub(r"\s+", " ", command).strip() for command in cls._commands(message)]
        return " && ".join(commands)

    @classmethod
    def _proposed_scope(cls, message: dict[str, Any]) -> str | None:
        targets = cls._exploration_targets(message)
        return " | ".join(sorted(targets)) if targets else None

    def _repeat_candidate(self, message: dict[str, Any], next_scope: str | None) -> tuple[str | None, int]:
        if self._is_submission(message):
            return None, 0
        if self._is_mutation(message):
            signature = self._mutation_signature(message)
            count = self._mutation_signature_counts[signature]
            if count:
                return f"revision:{self._workspace_revision}:mutation:{signature}", count
            touched_scopes = {
                path
                for command in self._commands(message)
                for path in self._source_paths(command)
            }
            repeated_scope_counts = {
                scope: self._mutation_scope_counts[scope]
                for scope in touched_scopes
                if self._mutation_scope_counts[scope] >= self.recovery_repeat_threshold
            }
            if repeated_scope_counts:
                scope, scope_count = max(
                    repeated_scope_counts.items(),
                    key=lambda item: item[1],
                )
                return (
                    f"revision:{self._workspace_revision}:mutation-scope:{scope}",
                    scope_count,
                )
            return None, 0
        if not next_scope:
            return None, 0
        next_scopes = set(next_scope.split(" | "))
        if next_scopes & self._post_mutation_validation_scopes:
            return None, 0
        command_key = self._command_key(message)
        command_count = self._exploration_counts[command_key]
        if command_count >= self.recovery_repeat_threshold:
            return f"revision:{self._workspace_revision}:command:{command_key}", command_count
        scope_count = self._scope_counts[next_scope]
        if next_scope == self._active_scope and scope_count >= self.recovery_repeat_threshold:
            return f"revision:{self._workspace_revision}:scope:{next_scope}", scope_count
        return None, 0

    @classmethod
    def _mutation_signature(cls, message: dict[str, Any]) -> str:
        material = "\n".join(
            [
                cls._command_key(message),
                *sorted(
                    {
                        path
                        for command in cls._commands(message)
                        for path in cls._source_paths(command)
                    }
                ),
            ]
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def _record_mutation(self, message: dict[str, Any]) -> None:
        signature = self._mutation_signature(message)
        self._mutation_signature_counts[signature] += 1
        touched_scopes = {
            path
            for command in self._commands(message)
            for path in self._source_paths(command)
        }
        if not touched_scopes and self._active_scope:
            touched_scopes.update(self._active_scope.split(" | "))
        for scope in touched_scopes:
            self._mutation_scope_counts[scope] += 1
        self._post_mutation_validation_scopes.update(touched_scopes)
        self._workspace_revision += 1
        self._revision_exploration_counts.clear()
        self._revision_scope_counts.clear()

    def _record_exploration(self, message: dict[str, Any], next_scope: str | None) -> None:
        if not next_scope:
            return
        command_key = self._command_key(message)
        self._exploration_counts[command_key] += 1
        self._scope_counts[next_scope] += 1
        self._revision_exploration_counts[command_key] += 1
        self._revision_scope_counts[next_scope] += 1
        self._post_mutation_validation_scopes.difference_update(next_scope.split(" | "))
        if next_scope != self._active_scope:
            self._active_scope = next_scope
            self._scope_revision = 1
            self._reviewed_scope_revision = 0
        else:
            self._scope_revision += 1

    def _block(
        self,
        message: dict[str, Any],
        decision: RecoveryDecision,
        *,
        submission_feedback: bool = False,
    ) -> list[dict[str, Any]]:
        branch_label = (
            "Branch label: IRRELEVANT; trajectory history is retained. "
            if decision.phase == "branch_review" and decision.decision == "PRUNE"
            else ""
        )
        reason = branch_label + (
            decision.reason or "The recovery reviewer requested a focused continuation."
        )
        focus = f" Next focus: {decision.next_focus}" if decision.next_focus else ""
        outputs = [
            {
                "output": f"Action not executed by trajectory recovery. {reason}{focus}",
                "returncode": 126,
                "exception_info": "",
            }
            for _ in message.get("extra", {}).get("actions", [])
        ]
        observations = self.model.format_observation_messages(message, outputs, self.get_template_vars())
        self.add_messages(*observations)
        self._inject(decision, submission_feedback=submission_feedback)
        return observations

    def _inject(self, decision: RecoveryDecision, *, submission_feedback: bool = False) -> None:
        signature = "\n".join(
            [decision.phase, decision.reason, decision.next_focus, *decision.unresolved_obligations]
        )
        if not signature.strip() or signature == self._last_injection_signature:
            return
        self._last_injection_signature = signature
        parts = [f"[REAL-TIME TRAJECTORY RECOVERY: {decision.phase}]"]
        if decision.phase == "branch_review" and decision.decision == "PRUNE":
            parts.append("Branch label: IRRELEVANT (trajectory history retained; nothing was deleted).")
        if decision.reason:
            parts.append(f"Failure if omitted: {decision.reason}")
        if decision.evidence:
            parts.append("Blocking evidence: " + decision.evidence[0])
        if decision.unresolved_obligations:
            parts.append("Immediate obligation: " + decision.unresolved_obligations[0])
        if decision.next_focus:
            parts.append(f"Required next action: {decision.next_focus}")
        if submission_feedback:
            parts.append(
                "Before resubmitting, address this blocker or explicitly explain—with repository or test evidence—"
                "why it is resolved or irrelevant. Your intervening actions and explanation will be shown to the "
                "semantic reviewer."
            )
        else:
            parts.append(
                "Use retained evidence and avoid re-reading an already exhausted scope unless you have a materially "
                "new question."
            )
        self.add_messages(self.model.format_message(role="user", content="\n".join(parts)))

    def serialize(self, *extra_dicts: dict[str, Any]) -> dict[str, Any]:
        data = super().serialize(*extra_dicts)
        data["trajectory_recovery"] = {
            "mode": self.recovery_mode,
            "events": self.recovery_events,
            "state": self._review_state(),
            "counts": self.recovery_counts,
            "reviewer_model": getattr(self.recovery_policy, "model_name", None),
            "reviewer_calls": getattr(self.recovery_policy, "n_calls", None),
            "reviewer_cost": getattr(self.recovery_policy, "cost", None),
        }
        return data



class RecoveryProgressTrackingAgent(RecoveryAgent):
    """RecoveryAgent that reports solver usage to the SWE-bench progress UI."""

    def __init__(
        self,
        *args: Any,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> list[dict[str, Any]]:
        self.progress_manager.update_instance_status(
            self.instance_id,
            f"Step {self.n_calls + 1:3d} (${self.cost:.2f})",
        )
        return super().step()
