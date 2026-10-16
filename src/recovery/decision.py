"""Reviewer decision type and the policy protocol shared by every adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


RecoveryPhase = Literal["branch_review", "repeat_exploration", "before_action", "before_submission"]



@dataclass
class RecoveryDecision:
    """Structured result returned by the online reviewer."""

    phase: RecoveryPhase
    decision: str
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    next_focus: str = ""
    retained_findings: list[str] = field(default_factory=list)
    unresolved_obligations: list[str] = field(default_factory=list)
    behavioral_specification: list[str] = field(default_factory=list)
    demonstrated_paths: list[str] = field(default_factory=list)
    owning_handling_points: list[str] = field(default_factory=list)
    rejected_alternatives: list[str] = field(default_factory=list)
    branch_finding: str = ""
    behavior_decision: str = ""
    blocker_id: str = ""
    investigation_complete: bool = False
    failed_open: bool = False
    raw: dict[str, Any] = field(default_factory=dict)



class RecoveryPolicy(Protocol):
    def review(
        self,
        *,
        phase: RecoveryPhase,
        task: str,
        messages: list[dict[str, Any]],
        proposed_message: dict[str, Any] | None,
        outputs: list[dict[str, Any]] | None,
        workspace: str,
        state: dict[str, Any],
    ) -> RecoveryDecision: ...
