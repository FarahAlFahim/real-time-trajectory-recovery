from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import src.runners.codex_swebench_lite as codex_pipeline
from src.adapters.codex import hook_bridge
from src.adapters.codex.controller import CodexHookRecoveryController
from src.adapters.codex.controller import RecoveryHookService
from src.adapters.codex.hook_bridge import _recent_solver_thoughts
from src.recovery import RecoveryDecision
from src.runners.codex_swebench_lite import _launch_args_override
from src.runners.codex_swebench_lite import _thread_start_config


def test_diff_capture_excludes_baseline_untracked_but_keeps_new_files(monkeypatch) -> None:
    calls: list[str] = []

    def fake_exec(_container: str, command: str, **_: object) -> tuple[int, str]:
        calls.append(command)
        if "git ls-files --others" in command:
            return 0, "build/lib/generated.py\0src/new_module.py\0"
        if "git add -N" in command:
            return 0, ""
        if "git diff --no-color" in command:
            return 0, (
                "diff --git a/build/lib/generated.py b/build/lib/generated.py\n"
                "new file mode 100644\n--- /dev/null\n+++ b/build/lib/generated.py\n"
                "@@ -0,0 +1 @@\n+generated\n"
                "diff --git a/src/new_module.py b/src/new_module.py\n"
                "new file mode 100644\n--- /dev/null\n+++ b/src/new_module.py\n"
                "@@ -0,0 +1 @@\n+source\n"
            )
        raise AssertionError(command)

    monkeypatch.setattr(codex_pipeline, "_docker_exec_capture", fake_exec)
    patch = codex_pipeline._capture_diff(
        "container",
        "base",
        frozenset({"build/lib/generated.py"}),
    )

    add_command = next(command for command in calls if "git add -N" in command)
    assert "src/new_module.py" in add_command
    assert "build/lib/generated.py" not in add_command
    assert "src/new_module.py" in patch
    assert "build/lib/generated.py" not in patch


class FakePolicy:
    model_name = "fake/reviewer"
    n_calls = 0
    cost = 0.0

    def __init__(self, decisions: dict[str, str] | None = None) -> None:
        self.decisions = decisions or {}
        self.phases: list[str] = []

    def review(self, *, phase: str, **_: object) -> RecoveryDecision:
        self.n_calls += 1
        self.phases.append(phase)
        decision = self.decisions.get(
            phase,
            {
                "branch_review": "KEEP",
                "repeat_exploration": "ALLOW_REPEAT",
                "before_action": "CONTINUE",
                "before_submission": "ALLOW_SUBMISSION",
            }[phase],
        )
        return RecoveryDecision(
            phase=phase,  # type: ignore[arg-type]
            decision=decision,
            reason="Concrete report-grounded reason.",
            evidence=["repository evidence"],
            next_focus="Inspect the named handling point.",
            unresolved_obligations=["Resolve the report-path behavior."],
            demonstrated_paths=["producer -> consumer"],
            owning_handling_points=["src/owner.py"],
            blocker_id="stable-blocker",
            branch_finding="The completed scope is unrelated." if decision == "PRUNE" else "Relevant finding.",
        )


def make_controller(workspace: list[str], decisions: dict[str, str] | None = None):
    controller = CodexHookRecoveryController(
        task="Fix the reported behavior.",
        reviewer_model="openai/fake",
        reviewer_timeout=1,
        reviewer_kwargs={},
        context_token_limit=10_000,
        max_submission_blocks=2,
        max_action_blocks=3,
        repeat_threshold=2,
        fail_open=True,
        review_branches=True,
        gate_mutations=True,
        gate_submission=True,
        workspace_snapshot=lambda: workspace[0],
    )
    policy = FakePolicy(decisions)
    controller.recovery_policy = policy
    return controller, policy


def pre(command: str, tool_use_id: str = "tool-1", tool_name: str = "Bash") -> dict:
    return {
        "hook_event_name": "PreToolUse",
        "turn_id": "turn-1",
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "tool_input": {"command": command},
    }


def post(command: str, output: str, tool_use_id: str = "tool-1") -> dict:
    return {
        "hook_event_name": "PostToolUse",
        "turn_id": "turn-1",
        "tool_name": "Bash",
        "tool_use_id": tool_use_id,
        "tool_input": {"command": command},
        "tool_response": {"output": output},
    }


def test_scope_departure_is_blocked_and_explicitly_labelled_irrelevant() -> None:
    workspace = [""]
    controller, policy = make_controller(workspace, {"branch_review": "PRUNE"})
    first = "sed -n '1,120p' src/a.py"
    assert controller.handle(pre(first)) is None
    controller.handle(post(first, "def a(): pass"))

    response = controller.handle(pre("sed -n '1,120p' src/b.py", "tool-2"))
    assert response is not None
    reason = response["hookSpecificOutput"]["permissionDecisionReason"]
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "IRRELEVANT" in reason
    assert policy.phases == ["branch_review"]
    assert controller.recovery_counts["branch_prunes"] == 1


def test_session_start_completes_native_hook_handshake() -> None:
    workspace = [""]
    controller, policy = make_controller(workspace)
    assert not controller.wait_for_handshake(timeout=0)
    assert controller.handle({"hook_event_name": "SessionStart", "session_id": "session-1"}) is None
    assert controller.wait_for_handshake(timeout=0)
    metadata = controller.metadata()
    assert metadata["handshake_complete"] is True
    assert metadata["hook_event_counts"] == {"SessionStart": 1}
    assert policy.n_calls == 0


def test_hook_trust_bypass_is_thread_scoped_and_active_only() -> None:
    common = {
        "recovery_hook_timeout_seconds": 210,
        "recovery_model_timeout_seconds": 180,
        "forward_env_keys": [],
        "model_providers": {},
        "model_provider": None,
        "codex_container_bin": "/usr/local/bin/codex",
        "model_name": "gpt-5-mini-2025-08-07",
        "model_reasoning_effort": None,
        "sandbox_mode": "danger-full-access",
    }
    active = _launch_args_override(
        SimpleNamespace(**common, recovery_mode="active"),
        "container",
        approval_policy="never",
    )
    baseline = _launch_args_override(
        SimpleNamespace(**common, recovery_mode="off"),
        "container",
        approval_policy="never",
    )
    assert "--dangerously-bypass-hook-trust" not in active
    assert "--dangerously-bypass-hook-trust" not in baseline
    assert _thread_start_config(SimpleNamespace(**common, recovery_mode="active")) == {
        "bypass_hook_trust": True
    }
    assert _thread_start_config(SimpleNamespace(**common, recovery_mode="off")) is None


def test_empty_diff_stop_continues_same_session_with_bounded_feedback() -> None:
    workspace = [""]
    controller, _ = make_controller(workspace)
    payload = {
        "hook_event_name": "Stop",
        "turn_id": "turn-1",
        "last_assistant_message": "I am done.",
    }
    response = controller.handle(payload)
    assert response is not None
    assert response["decision"] == "block"
    assert "diff is empty" in response["reason"]
    assert controller.recovery_counts["submission_blocks"] == 1

    controller.handle(payload)
    allowed = controller.handle(payload)
    assert allowed is None
    assert controller.recovery_counts["submission_blocks"] == 2


def test_filesystem_service_returns_hook_output(tmp_path: Path) -> None:
    workspace = [""]
    controller, _ = make_controller(workspace)
    service = RecoveryHookService(tmp_path, controller, fail_open=True)
    service.start()
    request = service.requests / "one.json"
    request.write_text(
        json.dumps(
            {
                "hook_event_name": "Stop",
                "turn_id": "turn-1",
                "last_assistant_message": "done",
            }
        ),
        encoding="utf-8",
    )
    response = service.responses / "one.json"
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not response.exists():
        time.sleep(0.01)
    service.stop()
    assert response.exists()
    value = json.loads(response.read_text(encoding="utf-8"))
    assert value["hook_output"]["decision"] == "block"


def test_container_bridge_round_trip(tmp_path: Path) -> None:
    workspace = [""]
    controller, _ = make_controller(workspace)
    service = RecoveryHookService(tmp_path, controller, fail_open=True)
    service.start()
    environment = dict(os.environ)
    environment["TRAJREC_IPC_ROOT"] = str(tmp_path)
    environment["TRAJREC_HOOK_TIMEOUT_SECONDS"] = "2"
    completed = subprocess.run(
        [sys.executable, hook_bridge.__file__],
        input=json.dumps(
            {
                "hook_event_name": "Stop",
                "turn_id": "turn-1",
                "last_assistant_message": "done",
            }
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
        timeout=5,
    )
    service.stop()
    assert completed.returncode == 0
    output = json.loads(completed.stdout)
    assert output["decision"] == "block"


def test_transcript_extraction_keeps_assistant_text_not_tool_output(tmp_path: Path) -> None:
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Inspect the downstream consumer."}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {"type": "function_call_output", "output": "very large code"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    assert _recent_solver_thoughts(str(transcript)) == ["Inspect the downstream consumer."]
