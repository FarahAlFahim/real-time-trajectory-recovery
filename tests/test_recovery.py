from __future__ import annotations

import subprocess
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import yaml

from minisweagent.config import builtin_config_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output

from src.recovery import LiteLLMRecoveryPolicy, RecoveryAgent, RecoveryDecision
from src.recovery import policy as recovery_module


class ScriptedRecoveryPolicy:
    def __init__(self, decisions: dict[str, list[RecoveryDecision]]):
        self.decisions = {phase: deque(items) for phase, items in decisions.items()}
        self.calls = defaultdict(int)
        self.contexts = defaultdict(list)
        self.model_name = "scripted-reviewer"
        self.n_calls = 0
        self.cost = 0.0

    def review(self, *, phase, **kwargs):
        self.calls[phase] += 1
        self.contexts[phase].append(kwargs)
        self.n_calls += 1
        if self.decisions.get(phase):
            return self.decisions[phase].popleft()
        defaults = {
            "branch_review": "KEEP",
            "repeat_exploration": "ALLOW_REPEAT",
            "before_action": "CONTINUE",
            "before_submission": "ALLOW_SUBMISSION",
        }
        return RecoveryDecision(phase=phase, decision=defaults[phase])


def make_model(commands: list[str]) -> DeterministicToolcallModel:
    outputs = []
    for index, command in enumerate(commands):
        call_id = f"call_{index}"
        outputs.append(
            make_toolcall_output(
                f"step {index}",
                [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
                [{"command": command, "tool_call_id": call_id}],
            )
        )
    return DeterministicToolcallModel(outputs=outputs)


def agent_config() -> dict:
    with (builtin_config_dir / "mini.yaml").open() as stream:
        config = yaml.safe_load(stream)["agent"]
    return {**config, "cost_limit": 100.0}


def make_agent(
    tmp_path: Path,
    commands: list[str],
    policy: ScriptedRecoveryPolicy,
    **kwargs,
) -> RecoveryAgent:
    return RecoveryAgent(
        model=make_model(commands),
        env=LocalEnvironment(cwd=str(tmp_path)),
        recovery_policy=policy,
        **kwargs,
        **agent_config(),
    )


def hard_submission_block(blocker_id: str) -> RecoveryDecision:
    return RecoveryDecision(
        phase="before_submission",
        decision="BLOCK_SUBMISSION",
        blocker_id=blocker_id,
        reason="A report-grounded consumer remains unresolved.",
        evidence=["The consumer still compares the captured sentinel exactly."],
        demonstrated_paths=["parser -> captured value -> exact sentinel consumer"],
        unresolved_obligations=["Validate or normalize the sentinel consumer."],
        next_focus="Inspect the consumer and test mixed-case input.",
        behavior_decision="UNRESOLVED",
    )


def test_reviews_scope_only_when_leaving_it(tmp_path):
    (tmp_path / "a.py").write_text("one\ntwo\n")
    (tmp_path / "b.py").write_text("three\n")
    policy = ScriptedRecoveryPolicy({})
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1p' a.py",
            "sed -n '2p' a.py",
            "sed -n '1p' b.py",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    info = agent.run("Trace the parser")

    assert info["exit_status"] == "Submitted"
    assert policy.calls["branch_review"] == 1
    assert agent.recovery_counts["branch_stays"] == 1
    assert agent.recovery_counts["branch_prunes"] == 0
    first_state = policy.contexts["branch_review"][0]["state"]
    assert first_state["active_scope"] == "a.py"
    assert first_state["active_scope_revision"] == 2


def test_revisited_scope_is_reviewed_once_per_workspace_revision(tmp_path):
    (tmp_path / "a.py").write_text("one\ntwo\n")
    (tmp_path / "b.py").write_text("three\n")
    policy = ScriptedRecoveryPolicy({})
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1p' a.py",
            "sed -n '1p' b.py",
            "sed -n '2p' a.py",
            "sed -n '1p' b.py",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    agent.run("Trace the parser without re-reviewing the same file")

    assert policy.calls["branch_review"] == 2


def test_directory_navigation_is_not_treated_as_a_reviewable_scope(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "sub").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("value = 1\n")
    policy = ScriptedRecoveryPolicy({})
    agent = make_agent(
        tmp_path,
        [
            "find pkg/sub -maxdepth 1 -type f",
            "sed -n '1p' pkg/a.py",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    agent.run("Inspect a concrete source file")

    assert policy.calls["branch_review"] == 0
    assert agent._active_scope == "pkg/a.py"


def test_branch_pruning_has_no_global_intervention_cap(tmp_path):
    scope_count = 11
    for index in range(scope_count):
        (tmp_path / f"scope_{index}.py").write_text(f"scope {index}\n")

    branch_decisions = [
        RecoveryDecision(
            phase="branch_review",
            decision="PRUNE",
            reason=f"scope_{index}.py is an exhausted branch.",
            branch_finding="No report-grounded relationship was found.",
            next_focus=f"Move to scope_{index + 1}.py with the unresolved requirement.",
        )
        for index in range(scope_count - 1)
    ]
    policy = ScriptedRecoveryPolicy({"branch_review": branch_decisions})

    commands = ["sed -n '1p' scope_0.py"]
    for index in range(1, scope_count):
        command = f"sed -n '1p' scope_{index}.py"
        commands.extend([command, command])  # First is pruned; second proceeds without a same-evidence re-review.
    commands.append("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
    agent = make_agent(tmp_path, commands, policy)

    info = agent.run("Find the owning implementation")

    assert info["exit_status"] == "Submitted"
    assert agent.recovery_counts["branch_prunes"] == 10
    assert policy.calls["branch_review"] == 10
    state = agent.serialize()["trajectory_recovery"]["state"]
    assert len(state["pruned_scopes"]) == 10
    assert all(item["relevance_label"] == "IRRELEVANT" for item in state["scope_history"])
    assert any(
        "Branch label: IRRELEVANT" in str(message.get("content", ""))
        for message in agent.messages
    )


def test_repeated_exploration_is_detected_and_interrupted(tmp_path):
    (tmp_path / "repeat.py").write_text("value = 1\n")
    policy = ScriptedRecoveryPolicy(
        {
            "repeat_exploration": [
                RecoveryDecision(
                    phase="repeat_exploration",
                    decision="BLOCK_REPEAT",
                    reason="This exact command has already produced the same evidence twice.",
                    next_focus="Inspect the downstream consumer instead.",
                )
            ]
        }
    )
    command = "sed -n '1p' repeat.py"
    agent = make_agent(
        tmp_path,
        [command, command, command, command, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"],
        policy,
    )

    info = agent.run("Avoid cycling")

    assert info["exit_status"] == "Submitted"
    assert policy.calls["repeat_exploration"] == 1
    assert agent.recovery_counts["repeat_blocks"] == 1
    assert agent.serialize()["trajectory_recovery"]["state"]["exploration_ledger"]["scope_counts"]["repeat.py"] == 3


def test_allowed_same_scope_repeat_is_reviewed_only_once(tmp_path):
    (tmp_path / "repeat.py").write_text("one\ntwo\nthree\nfour\n")
    policy = ScriptedRecoveryPolicy(
        {
            "repeat_exploration": [
                RecoveryDecision(phase="repeat_exploration", decision="ALLOW_REPEAT")
            ]
        }
    )
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1p' repeat.py",
            "sed -n '2p' repeat.py",
            "sed -n '3p' repeat.py",
            "sed -n '4p' repeat.py",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    agent.run("Inspect distinct ranges without reviewer churn")

    assert policy.calls["repeat_exploration"] == 1


def test_reviews_use_retained_summary_plus_new_scope_delta(tmp_path):
    for name, value in (("a.py", "A_ONLY"), ("b.py", "B_ONLY"), ("c.py", "C_ONLY")):
        (tmp_path / name).write_text(value + "\n")
    policy = ScriptedRecoveryPolicy(
        {
            "branch_review": [
                RecoveryDecision(
                    phase="branch_review",
                    decision="KEEP",
                    retained_findings=["summary of a.py"],
                    evidence=["a.py: EXACT_OFFSET = 3"],
                ),
                RecoveryDecision(phase="branch_review", decision="KEEP"),
            ]
        }
    )
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1p' a.py",
            "sed -n '1p' b.py",
            "sed -n '1p' c.py",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    agent.run("Trace three scopes")

    second = policy.contexts["branch_review"][1]
    second_messages = str(second["messages"])
    assert "B_ONLY" in second_messages
    assert "A_ONLY" not in second_messages
    assert second["state"]["retained_findings"] == ["summary of a.py"]
    assert second["state"]["evidence_ledger"] == []
    assert second["state"]["evidence_ledger_digest"]["excluded_nonlocal_entries"] == 1


def test_post_mutation_validation_starts_a_new_repeat_revision(tmp_path):
    source = tmp_path / "repeat.py"
    source.write_text("one\ntwo\n")
    policy = ScriptedRecoveryPolicy({})
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1p' repeat.py",
            "sed -n '2p' repeat.py",
            "printf 'changed\\n' > repeat.py",
            "sed -n '1p' repeat.py",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    agent.run("Edit and then validate the same scope")

    assert policy.calls["repeat_exploration"] == 0
    state = agent.serialize()["trajectory_recovery"]["state"]
    assert state["workspace_revision"] == 1
    assert state["exploration_ledger"]["since_last_mutation"]["scope_counts"] == {
        "repeat.py": 1
    }


def test_post_mutation_validation_allowance_is_single_use(tmp_path):
    source = tmp_path / "repeat.py"
    source.write_text("one\ntwo\n")
    policy = ScriptedRecoveryPolicy(
        {
            "repeat_exploration": [
                RecoveryDecision(
                    phase="repeat_exploration",
                    decision="BLOCK_REPEAT",
                    reason="The post-edit validation allowance has already been used.",
                )
            ]
        }
    )
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1p' repeat.py",
            "sed -n '2p' repeat.py",
            "printf 'changed\\n' > repeat.py",
            "sed -n '1p' repeat.py",
            "sed -n '1p' repeat.py",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    agent.run("Allow one post-edit validation, not an exploration loop")

    assert policy.calls["repeat_exploration"] == 1
    assert agent.recovery_counts["repeat_blocks"] == 1


def test_oscillating_mutation_signature_is_reviewed(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("initial\n")
    first_mutation = "printf 'first\\n' > source.py"
    second_mutation = "printf 'second\\n' > source.py"
    policy = ScriptedRecoveryPolicy(
        {
            "repeat_exploration": [
                RecoveryDecision(
                    phase="repeat_exploration",
                    decision="BLOCK_REPEAT",
                    reason="This mutation signature already executed in an earlier revision.",
                    next_focus="Validate the current second revision instead of restoring the first.",
                )
            ]
        }
    )
    agent = make_agent(
        tmp_path,
        [
            first_mutation,
            second_mutation,
            first_mutation,
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    agent.run("Do not oscillate between the same edits")

    assert source.read_text() == "second\n"
    assert policy.calls["repeat_exploration"] == 1
    assert agent.recovery_counts["repeat_blocks"] == 1


def test_distinct_mutations_to_same_scope_use_configured_repeat_threshold(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("initial\n")
    policy = ScriptedRecoveryPolicy(
        {
            "repeat_exploration": [
                RecoveryDecision(
                    phase="repeat_exploration",
                    decision="BLOCK_REPEAT",
                    reason="This scope has already received the configured number of edits.",
                )
            ]
        }
    )
    agent = make_agent(
        tmp_path,
        [
            "printf 'first\\n' > source.py",
            "printf 'second\\n' > source.py",
            "printf 'third\\n' > source.py",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
        recovery_repeat_threshold=2,
    )

    agent.run("Review sustained editing even when commands differ")

    assert source.read_text() == "second\n"
    assert policy.calls["repeat_exploration"] == 1
    assert agent.recovery_counts["repeat_blocks"] == 1


def test_action_block_limit_skips_further_mutation_reviews(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 0\n")
    policy = ScriptedRecoveryPolicy(
        {
            "before_action": [
                RecoveryDecision(
                    phase="before_action",
                    decision="BLOCK_ACTION",
                    reason="The observed consumer conflicts with this edit.",
                    evidence=["consumer.py: value == sentinel"],
                    next_focus="Inspect and reconcile the exact consumer.",
                    unresolved_obligations=["Preserve the observed consumer behavior."],
                ),
                RecoveryDecision(
                    phase="before_action",
                    decision="BLOCK_ACTION",
                    reason="should not be requested",
                    evidence=["consumer.py: value == sentinel"],
                    next_focus="should not be requested",
                    unresolved_obligations=["should not be requested"],
                ),
            ]
        }
    )
    mutation = "printf 'value = 1\\n' > source.py"
    agent = make_agent(
        tmp_path,
        [mutation, mutation, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"],
        policy,
        recovery_max_action_blocks=1,
    )

    agent.run("Make one focused edit")

    assert policy.calls["before_action"] == 1
    assert source.read_text() == "value = 1\n"


def test_astropy_style_scope_review_blocks_premature_edit(tmp_path):
    qdp = tmp_path / "qdp.py"
    qdp.write_text('flags = 0\nif v == "NO":\n    pass\n')
    policy = ScriptedRecoveryPolicy(
        {
            "before_action": [
                RecoveryDecision(
                    phase="before_action",
                    decision="BLOCK_ACTION",
                    reason="The proposed global matching change affects an exact sentinel consumer in this scope.",
                    evidence=['if v == "NO"'],
                    next_focus="Trace captured casing and validate the sentinel branch before editing.",
                    unresolved_obligations=["Preserve NO sentinel behavior for mixed-case input."],
                )
            ]
        }
    )
    mutation = "python -c \"from pathlib import Path; Path('qdp.py').write_text('changed')\""
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1,20p' qdp.py",
            mutation,
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    info = agent.run("Make QDP commands case-insensitive")

    assert info["exit_status"] == "Submitted"
    assert qdp.read_text() == 'flags = 0\nif v == "NO":\n    pass\n'
    assert policy.calls["before_action"] == 1
    assert agent.recovery_counts["action_blocks"] == 1


def test_submission_blocks_are_limited_and_include_solver_feedback(tmp_path):
    policy = ScriptedRecoveryPolicy(
        {
            "before_submission": [
                hard_submission_block("sentinel-consumer"),
                hard_submission_block("sentinel-consumer"),
                hard_submission_block("sentinel-consumer"),
            ]
        }
    )
    agent = make_agent(
        tmp_path,
        [
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            "echo investigated-consumer",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            "echo explained-why-consumer-is-irrelevant",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
        recovery_max_submission_blocks=2,
    )

    info = agent.run("Validate before submission")

    assert info["exit_status"] == "Submitted"
    assert agent.n_calls == 5
    assert policy.calls["before_submission"] == 2
    assert agent.recovery_counts["submission_blocks"] == 2
    second_state = policy.contexts["before_submission"][1]["state"]
    feedback = str(second_state["solver_response_since_last_block"])
    assert "investigated-consumer" in feedback
    assert len(second_state["submission_history"]) == 1


def test_submission_block_preserves_safe_patch_packaging(tmp_path):
    (tmp_path / "source.py").write_text("value = 1\n")
    policy = ScriptedRecoveryPolicy(
        {"before_submission": [hard_submission_block("consumer-check")]}
    )
    combined = (
        "git diff -- source.py > patch.txt; "
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
    )
    agent = make_agent(
        tmp_path,
        [combined, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"],
        policy,
        recovery_max_submission_blocks=1,
    )

    agent.run("Preserve patch packaging across one submission block")

    assert (tmp_path / "patch.txt").exists()


def test_unsupported_semantic_block_is_advisory_not_hard(tmp_path):
    policy = ScriptedRecoveryPolicy(
        {
            "before_submission": [
                RecoveryDecision(
                    phase="before_submission",
                    decision="BLOCK_SUBMISSION",
                    reason="More exploration might be useful.",
                )
            ]
        }
    )
    agent = make_agent(tmp_path, ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"], policy)

    info = agent.run("Do not hard-block vague concerns")

    assert info["exit_status"] == "Submitted"
    assert agent.recovery_counts["submission_blocks"] == 0
    event = agent.recovery_events[0]
    assert event["recommended_decision"] == "BLOCK_SUBMISSION"
    assert event["effective_decision"] == "ALLOW_SUBMISSION"
    assert event["policy_note"] == "insufficient_evidence_for_hard_block"


def test_shadow_mode_records_but_does_not_block_mutation(tmp_path):
    policy = ScriptedRecoveryPolicy(
        {
            "before_action": [
                RecoveryDecision(
                    phase="before_action",
                    decision="BLOCK_ACTION",
                    reason="Shadow recommendation.",
                )
            ]
        }
    )
    agent = make_agent(
        tmp_path,
        ["printf 'value = 1\n' > shadow.py", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"],
        policy,
        recovery_mode="shadow",
    )

    info = agent.run("Observe without intervention")

    assert info["exit_status"] == "Submitted"
    assert (tmp_path / "shadow.py").read_text() == "value = 1\n"
    event = agent.recovery_events[0]
    assert event["recommended_decision"] == "BLOCK_ACTION"
    assert event["effective_decision"] == "CONTINUE"


def test_python_regex_data_is_not_mistaken_for_shell_redirection():
    regex_test = make_model(
        [
            "python - << 'PY'\n"
            "import re\n"
            "pattern = re.compile(r'(?P<command>READ)')\n"
            "print(bool(pattern.match('READ')))\n"
            "PY"
        ]
    ).config.outputs[0]
    patch_packaging = make_model(["git diff -- source.py > patch.txt"]).config.outputs[0]
    real_redirect = make_model(["printf changed > source.py"]).config.outputs[0]

    assert RecoveryAgent._is_mutation(regex_test) is False
    assert RecoveryAgent._is_mutation(patch_packaging) is False
    assert RecoveryAgent._is_mutation(real_redirect) is True


def test_shell_code_fragments_are_not_treated_as_new_scopes():
    message = make_model(
        [
            "grep -n 's/if/re.c/raw/f/' astropy/io/ascii/qdp.py; "
            "cp astropy/io/ascii/qdp.py astropy/io/ascii/qdp.py.bak"
        ]
    ).config.outputs[0]

    assert RecoveryAgent._proposed_scope(message) == "astropy/io/ascii/qdp.py"


def test_wrapper_off_mode_uses_stock_process_instance():
    import dataclasses

    import src.runners.minisweagent_swebench_lite as wrapper

    calls = []
    original = wrapper.process_instance
    wrapper.process_instance = lambda *args: calls.append(args)
    try:
        cfg = dataclasses.replace(
            wrapper._build_runner_config(),
            trajectory_recovery_mode="off",
        )
        marker_args = ({"instance_id": "example"}, Path("out"), {"agent": {}}, object())
        wrapper._process_instance_for_config(cfg, *marker_args)
    finally:
        wrapper.process_instance = original

    assert calls == [marker_args]
    off_config = wrapper._build_minisweagent_config(cfg)
    shadow_config = wrapper._build_minisweagent_config(
        dataclasses.replace(cfg, trajectory_recovery_mode="shadow")
    )
    active_config = wrapper._build_minisweagent_config(
        dataclasses.replace(cfg, trajectory_recovery_mode="active")
    )
    assert off_config["model"]["model_kwargs"]["parallel_tool_calls"] is True
    assert shadow_config["model"]["model_kwargs"]["parallel_tool_calls"] is True
    assert active_config["model"]["model_kwargs"]["parallel_tool_calls"] is False


def test_litellm_policy_parses_robust_branch_decision():
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(
            content=(
                '{"decision":"KEEP","reason":"the related consumer remains relevant",'
                '"evidence":["if v == NO"],"next_focus":"inspect the consumer",'
                '"retained_findings":["matching became case-insensitive"],'
                '"unresolved_obligations":["normalize the sentinel"],'
                '"behavioral_specification":["mixed-case commands preserve behavior"],'
                '"demonstrated_paths":["parser -> sentinel"],'
                '"owning_handling_points":["qdp.py sentinel branch"],'
                '"rejected_alternatives":[],"branch_finding":"scope remains relevant",'
                '"behavior_decision":"UNRESOLVED","blocker_id":"sentinel",'
                '"investigation_complete":false}'
            )
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], _hidden_params={"response_cost": 0.01})

    original = recovery_module.litellm.completion
    recovery_module.litellm.completion = fake_completion
    try:
        policy = LiteLLMRecoveryPolicy("test/reviewer")
        decision = policy.review(
            phase="branch_review",
            task="Make commands case-insensitive",
            messages=[{"role": "tool", "content": 'if v == "NO"'}],
            proposed_message=None,
            outputs=None,
            workspace="",
            state={"active_scope": "qdp.py"},
        )
    finally:
        recovery_module.litellm.completion = original

    assert decision.decision == "KEEP"
    assert decision.behavioral_specification == ["mixed-case commands preserve behavior"]
    assert policy.n_calls == 1
    assert policy.cost == 0.01
    assert captured["messages"][1]["content"].endswith("Make commands case-insensitive")
    prompt = captured["messages"][0]["content"] + captured["messages"][2]["content"]
    assert "already explored active_scope" in prompt
    assert "Do not judge an unexplored next file from its name" in prompt


def test_unsupported_action_block_is_advisory_not_hard(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 0\n")
    policy = ScriptedRecoveryPolicy(
        {
            "before_action": [
                RecoveryDecision(
                    phase="before_action",
                    decision="BLOCK_ACTION",
                    reason="",
                )
            ]
        }
    )
    agent = make_agent(
        tmp_path,
        [
            "printf 'value = 1\\n' > source.py",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    agent.run("Apply a focused repair")

    assert source.read_text() == "value = 1\n"
    assert agent.recovery_counts["action_blocks"] == 0
    event = next(event for event in agent.recovery_events if event["phase"] == "before_action")
    assert event["recommended_decision"] == "BLOCK_ACTION"
    assert event["effective_decision"] == "CONTINUE"
    assert event["policy_note"] == "insufficient_evidence_for_action_block"


def test_exhaustive_action_block_is_advisory_not_hard(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 0\n")
    policy = ScriptedRecoveryPolicy(
        {
            "before_action": [
                RecoveryDecision(
                    phase="before_action",
                    decision="BLOCK_ACTION",
                    reason="The layout might affect other variants.",
                    evidence=["Data.start_line = 3"],
                    next_focus="Enumerate every code path and all FixedWidth variants.",
                    unresolved_obligations=["Reconcile every dependent offset."],
                )
            ]
        }
    )
    agent = make_agent(
        tmp_path,
        ["printf 'value = 1\\n' > source.py", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"],
        policy,
    )

    agent.run("Make the focused source change")

    assert source.read_text() == "value = 1\n"
    assert agent.recovery_counts["action_blocks"] == 0
    event = next(event for event in agent.recovery_events if event["phase"] == "before_action")
    assert event["effective_decision"] == "CONTINUE"
    assert event["policy_note"] == "insufficient_evidence_for_action_block"


def test_litellm_policy_keeps_stable_prefix_before_variable_review_data():
    policy = LiteLLMRecoveryPolicy("test/reviewer")
    task = "Repair the parser without breaking exact sentinel handling."
    prefix = [
        {"role": "system", "content": policy._SYSTEM_PROMPT},
        {"role": "user", "content": "Original software-repair problem statement:\n\n" + task},
    ]
    branch_prompt = policy._build_prompt(
        phase="branch_review", task=task, messages=[], proposed_message=None,
        outputs=None, workspace="", state={"active_scope": "parser.py"},
    )
    action_prompt = policy._build_prompt(
        phase="before_action", task=task, messages=[{"role": "tool", "content": "new evidence"}],
        proposed_message={"role": "assistant", "content": "edit parser.py"}, outputs=None,
        workspace="diff", state={"active_scope": "parser.py"},
    )
    assert prefix[0]["content"] == policy._SYSTEM_PROMPT
    assert prefix[1]["content"].endswith(task)
    assert branch_prompt != action_prompt
    assert "original_problem_statement" not in branch_prompt


def test_compaction_preserves_astropy_sentinel_consumer():
    prefix = "x" * 2289
    evidence = '                if v == "NO":\n                    values.append(masked)\n'
    output = prefix + evidence + "y" * 6200
    compact = LiteLLMRecoveryPolicy._compact_messages([{"role": "tool", "content": output}])
    assert 'if v == "NO"' in compact[0]["content"]


def test_context_overflow_keeps_thoughts_not_large_observations():
    policy = LiteLLMRecoveryPolicy("test/reviewer", context_token_limit=2_000)
    prompt = policy._build_prompt(
        phase="branch_review",
        task="Repair the parser.",
        messages=[
            {"role": "assistant", "content": "THOUGHT: trace the sentinel consumer before editing."},
            {"role": "tool", "content": "LARGE_CODE_BLOCK " + "x" * 50_000},
        ],
        proposed_message=None,
        outputs=None,
        workspace="",
        state={"active_scope": "parser.py"},
    )
    assert "trace the sentinel consumer before editing" in prompt
    assert "LARGE_CODE_BLOCK" not in prompt


def test_reviewer_evidence_pack_excludes_irrelevant_scopes_but_audit_keeps_them(tmp_path):
    policy = ScriptedRecoveryPolicy({})
    agent = make_agent(
        tmp_path,
        ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"],
        policy,
    )
    agent._active_scope = "active.py"
    agent._pruned_scopes = ["irrelevant.py"]
    agent._evidence_ledger = [
        {"step": 1, "scope": "irrelevant.py", "evidence": ["irrelevant evidence"]},
        {"step": 2, "scope": "active.py", "evidence": ["active evidence"]},
    ]

    review_state = agent._review_state(for_reviewer=True)
    audit_state = agent._review_state()

    assert [item["scope"] for item in review_state["evidence_ledger"]] == ["active.py"]
    assert review_state["evidence_ledger_digest"]["excluded_irrelevant_entries"] == 1
    assert {item["scope"] for item in audit_state["evidence_ledger"]} == {
        "active.py",
        "irrelevant.py",
    }


def test_online_evidence_pack_keeps_latest_state_per_scope(tmp_path):
    policy = ScriptedRecoveryPolicy({})
    agent = make_agent(tmp_path, ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"], policy)
    agent._active_scope = "active.py"
    agent._evidence_ledger = [
        {"step": 1, "scope": "active.py", "evidence": ["old paraphrase"]},
        {"step": 2, "scope": "active.py", "evidence": ["canonical current finding"]},
    ]

    review_state = agent._review_state(for_reviewer=True)
    audit_state = agent._review_state()

    assert [item["evidence"] for item in review_state["evidence_ledger"]] == [
        ["canonical current finding"]
    ]
    assert review_state["evidence_ledger_digest"]["superseded_relevant_entries"] == 1
    assert len(audit_state["evidence_ledger"]) == 2


def test_evidence_pack_size_is_governed_by_context_budget_not_item_count():
    policy = LiteLLMRecoveryPolicy("test/reviewer", context_token_limit=10_000)
    ledger = [
        {
            "step": index,
            "scope": "active.py",
            "evidence": [f"evidence-{index}-" + "x" * 4_000],
        }
        for index in range(20)
    ]
    original_count = len(ledger)
    state = {
        "active_scope": "active.py",
        "evidence_ledger": ledger,
        "evidence_ledger_digest": {
            "total_entries": len(ledger),
            "relevant_entries": len(ledger),
            "excluded_irrelevant_entries": 0,
            "included_entries": len(ledger),
        },
    }
    task = "Repair active.py."

    prompt = policy._build_prompt(
        phase="before_submission",
        task=task,
        messages=[],
        proposed_message=None,
        outputs=None,
        workspace="",
        state=state,
    )
    total_tokens = policy._token_count(
        policy._SYSTEM_PROMPT
        + "\nOriginal software-repair problem statement:\n\n"
        + task
        + prompt
    )

    assert total_tokens <= policy.context_token_limit
    assert 0 < state["evidence_ledger_digest"]["included_entries"] < original_count


def test_broad_matching_change_surfaces_exact_value_consumers():
    proposed = {
        "role": "assistant",
        "extra": {
            "actions": [
                {"command": "replace re.compile(pattern) with re.compile(pattern, re.IGNORECASE)"}
            ]
        },
    }
    context = LiteLLMRecoveryPolicy._semantic_risk_context(
        [{"role": "tool", "content": 'for v in values:\n    if v == "NO":\n        mask(v)'}],
        proposed,
    )
    assert context["broad_matching_or_normalization_change"] is True
    assert any('v == "NO"' in item for item in context["observed_exact_value_consumers"])


def test_variable_cardinality_change_surfaces_dependent_offsets_and_indices():
    proposed = {
        "role": "assistant",
        "extra": {
            "actions": [
                {
                    "command": (
                        "set header_rows from kwargs; "
                        "set pos_index = len(header_rows)"
                    )
                }
            ]
        },
    }
    context = LiteLLMRecoveryPolicy._semantic_risk_context(
        [
            {
                "role": "tool",
                "content": (
                    "class Data:\n"
                    "    start_line = 3\n"
                    "if self.data.start_line is None:\n"
                    "    self.data.start_line = len(header_rows)\n"
                    "border = lines[1]\n"
                ),
            }
        ],
        proposed,
    )
    assert context["variable_cardinality_or_layout_change"] is True
    assert any("start_line = 3" in item for item in context["observed_dependent_offsets"])
    assert any("lines[1]" in item for item in context["observed_dependent_indices"])


def test_submission_policy_requires_complete_report_path_without_promoting_hints():
    prompt = LiteLLMRecoveryPolicy._SYSTEM_PROMPT

    assert "every explicit reproduction outcome" in prompt
    assert "Removing an initial exception is not sufficient" in prompt
    assert "unrelated offsets or speculative edge cases" in prompt
    assert "producer-to-output path" in prompt
    assert "mere proximity is not" in prompt
    assert "exactly one immediate action" in prompt


def test_validation_only_concern_does_not_block_provisional_action():
    decision = RecoveryDecision(
        phase="before_action",
        decision="BLOCK_ACTION",
        reason="The report-path dependency still requires runtime validation.",
        evidence=["The changed value reaches the output formatter."],
        next_focus="Run the report reproduction and verify its output.",
        unresolved_obligations=["Validate the report-path output."],
        demonstrated_paths=["input -> formatter -> output"],
    )

    assert RecoveryAgent._valid_action_block(decision) is False


def test_demonstrated_failure_can_still_block_action():
    decision = RecoveryDecision(
        phase="before_action",
        decision="BLOCK_ACTION",
        reason="The proposed mutation has a demonstrated output mismatch.",
        evidence=["The report reproduction failed with an output mismatch."],
        next_focus="Correct the formatter before repeating the mutation.",
        unresolved_obligations=["Preserve the required output."],
        demonstrated_paths=["input -> formatter -> incorrect output"],
    )

    assert RecoveryAgent._valid_action_block(decision) is True


def test_semantic_risk_ignores_prose_mentions_and_keeps_code_constraints():
    proposed = {
        "role": "assistant",
        "extra": {"actions": [{"command": "make record_count configurable"}]},
    }
    context = LiteLLMRecoveryPolicy._semantic_risk_context(
        [
            {
                "role": "tool",
                "content": (
                    "# start_line and position_line are described here\n"
                    "The start_line documentation mentions a boundary.\n"
                    "class Data:\n"
                    "    start_line = 3\n"
                    "value = lines[1]\n"
                ),
            }
        ],
        proposed,
    )

    assert context["observed_dependent_offsets"] == ["Data.start_line = 3"]
    assert context["observed_dependent_indices"] == ["value = lines[1]"]


def test_snake_case_cardinality_change_uses_dependencies_from_observed_evidence():
    proposed = {
        "role": "assistant",
        "extra": {
            "actions": [
                {
                    "command": (
                        "change RST.__init__ to accept kwargs containing header_rows "
                        "and forward them to FixedWidth"
                    )
                }
            ]
        },
    }
    context = LiteLLMRecoveryPolicy._semantic_risk_context(
        [
            {
                "role": "tool",
                "content": (
                    "class SimpleRSTData:\n"
                    "    start_line = 3\n"
                    "if self.data.start_line is None:\n"
                    "    self.data.start_line = len(header_rows)\n"
                    "lines = [lines[1]] + lines + [lines[1]]\n"
                ),
            }
        ],
        proposed,
    )

    assert context["variable_cardinality_or_layout_change"] is True
    assert any("start_line = 3" in item for item in context["observed_dependent_offsets"])
    assert any("lines[1]" in item for item in context["observed_dependent_indices"])


def test_cardinality_hint_recalls_exact_constraint_without_blocking(tmp_path):
    layout = tmp_path / "layout.py"
    layout.write_text(
        "class SimpleRSTData:\n"
        "    start_line = 3\n"
        "class RST:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "border = lines[1]\n"
    )
    (tmp_path / "helper.py").write_text("value = 1\n")
    policy = ScriptedRecoveryPolicy(
        {
            "before_action": [
                RecoveryDecision(
                    phase="before_action",
                    decision="CONTINUE",
                    reason="The constructor now forwards header_rows.",
                )
            ]
        }
    )
    mutation = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "p = Path('layout.py')\n"
        "s = p.read_text()\n"
        "p.write_text(s.replace('def __init__(self):', "
        "'def __init__(self, header_rows=None):'))\n"
        "PY"
    )
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1,30p' layout.py",
            "sed -n '1,10p' helper.py",
            mutation,
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    agent.run("Support a variable number of header_rows")

    assert agent.recovery_counts["action_blocks"] == 0
    event = next(event for event in agent.recovery_events if event["phase"] == "before_action")
    assert event["effective_decision"] == "CONTINUE"
    risk = policy.contexts["before_action"][0]["state"]["current_semantic_risk_context"]
    assert risk["observed_dependent_offsets"] == ["SimpleRSTData.start_line = 3"]
    assert "producer-to-output path" in risk["required_cardinality_check"]


def test_cardinality_hint_reports_constraints_without_forcing_a_block(tmp_path):
    source = tmp_path / "layout.py"
    source.write_text(
        "class Data:\n"
        "    start_line = 3\n"
        "other_start_line = 9\n"
        "border = lines[1]\n"
    )
    policy = ScriptedRecoveryPolicy(
        {
            "before_action": [
                RecoveryDecision(
                    phase="before_action",
                    decision="CONTINUE",
                    reason="Forward header_rows to the base writer.",
                )
            ]
        }
    )
    mutation = "printf '# header_rows count changes layout\\n' >> layout.py"
    agent = make_agent(
        tmp_path,
        ["sed -n '1,20p' layout.py", mutation, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"],
        policy,
    )

    agent.run("Support variable header_rows")

    event = next(event for event in agent.recovery_events if event["phase"] == "before_action")
    assert event["effective_decision"] == "CONTINUE"
    risk = policy.contexts["before_action"][0]["state"]["current_semantic_risk_context"]
    assert risk["observed_dependent_offsets"][0] == "Data.start_line = 3"


def test_cardinality_evidence_remains_advisory_without_demonstrated_conflict(tmp_path):
    source = tmp_path / "layout.py"
    source.write_text(
        "class SimpleRSTData:\n"
        "    start_line = 3\n"
        "if self.data.start_line is None:\n"
        "    self.data.start_line = len(header_rows)\n"
        "border = lines[1]\n"
        "class RST:\n"
        "    def __init__(self):\n"
        "        pass\n"
    )
    policy = ScriptedRecoveryPolicy(
        {
            "before_action": [
                RecoveryDecision(
                    phase="before_action",
                    decision="CONTINUE",
                    reason="Forwarding header_rows is sufficient because the base class sets layout metadata.",
                    evidence=[
                        "start_line = 3",
                        "if self.data.start_line is None:",
                        "border = lines[1]",
                    ],
                )
            ]
        }
    )
    mutation = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "p = Path('layout.py')\n"
        "s = p.read_text()\n"
        "p.write_text(s.replace('def __init__(self):', "
        "'def __init__(self, **kwargs):  # forwards header_rows'))\n"
        "PY"
    )
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1,40p' layout.py",
            mutation,
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    agent.run("Support header_rows without breaking the variable layout")

    assert agent.recovery_counts["action_blocks"] == 0
    event = next(event for event in agent.recovery_events if event["phase"] == "before_action")
    assert event["effective_decision"] == "CONTINUE"


def test_staged_change_refreshes_empty_patch_artifact_from_head(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    source.write_text("value = 2\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    (tmp_path / "patch.txt").write_text("")

    policy = ScriptedRecoveryPolicy({})
    agent = make_agent(
        tmp_path,
        ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"],
        policy,
    )

    info = agent.run("Submit the staged repair")

    patch = (tmp_path / "patch.txt").read_text()
    assert info["exit_status"] == "Submitted"
    assert "diff --git a/source.py b/source.py" in patch
    assert "+value = 2" in patch
    assert "diff --git a/source.py b/source.py" in info["submission"]
    assert "diff --git a/source.py b/source.py" in policy.contexts["before_submission"][0]["workspace"]
    assert any(
        event.get("phase") == "control_policy"
        and "Refreshed patch.txt" in event.get("reason", "")
        for event in agent.recovery_events
    )


def test_active_recovery_blocks_solver_commit_and_preserves_patch(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()

    policy = ScriptedRecoveryPolicy({})
    agent = make_agent(
        tmp_path,
        [
            "printf 'value = 2\\n' > source.py",
            "git add -A && git commit -m 'solver commit'",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
        ],
        policy,
    )

    info = agent.run("Change the value without committing")

    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip() == base_commit
    assert "+value = 2" in info["submission"]
    assert any(
        event.get("policy_note") == "solver_commit_blocked"
        for event in agent.recovery_events
    )


def test_initial_revision_diff_recovers_already_committed_solver_change(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    source.write_text("value = 2\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "solver commit"], cwd=tmp_path, check=True)

    policy = ScriptedRecoveryPolicy({})
    agent = make_agent(tmp_path, ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"], policy)
    agent._base_commit = base_commit
    message = {
        "extra": {
            "actions": [
                {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"}
            ]
        }
    }

    decision = agent._refresh_submission_patch_artifact(message)

    assert decision is None
    assert "+value = 2" in (tmp_path / "patch.txt").read_text()


def test_patch_artifact_refresh_is_independent_of_semantic_block_budget(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    source.write_text("value = 2\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    (tmp_path / "patch.txt").write_text("")

    policy = ScriptedRecoveryPolicy({})
    agent = make_agent(
        tmp_path,
        [
            "git diff -- source.py > patch.txt; "
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
        ],
        policy,
        recovery_max_submission_blocks=0,
    )

    info = agent.run("Submit a staged repair after the semantic budget is exhausted")

    assert info["exit_status"] == "Submitted"
    assert "+value = 2" in info["submission"]
    assert policy.calls["before_submission"] == 0


def test_allow_decision_keeps_cardinality_dependency_advisory(tmp_path):
    source = tmp_path / "layout.py"
    source.write_text(
        "class Data:\n"
        "    start_line = 3\n"
        "if self.data.start_line is None:\n"
        "    self.data.start_line = len(header_rows)\n"
        "border = lines[1]\n"
    )
    policy = ScriptedRecoveryPolicy(
        {
            "before_action": [
                RecoveryDecision(
                    phase="before_action",
                    decision="CONTINUE",
                    reason="The output border now follows the number of header rows.",
                    evidence=["The hard-coded border index is replaced."],
                )
            ]
        }
    )
    mutation = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "p = Path('layout.py')\n"
        "text = p.read_text()\n"
        "p.write_text(text.replace('border = lines[1]', "
        "'pos_index = len(header_rows)\\nborder = lines[pos_index]'))\n"
        "PY"
    )
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1,20p' layout.py",
            mutation,
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    agent.run("Support a variable number of header rows")

    assert agent.recovery_counts["action_blocks"] == 0
    event = next(event for event in agent.recovery_events if event["phase"] == "before_action")
    assert event["effective_decision"] == "CONTINUE"


def test_allow_decision_cannot_ignore_broad_matching_consumer(tmp_path):
    policy = ScriptedRecoveryPolicy(
        {
            "before_action": [
                RecoveryDecision(
                    phase="before_action",
                    decision="CONTINUE",
                    reason="The regex now accepts the reported lowercase command.",
                )
            ]
        }
    )
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1,120p' parser.py",
            "python - <<'PY'\nfrom pathlib import Path\np=Path('parser.py')\ns=p.read_text()\np.write_text(s.replace('re.compile(p)', 're.compile(p, re.IGNORECASE)'))\nPY",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )
    (tmp_path / "parser.py").write_text('if v == "NO":\n    pass\npattern = re.compile(p)\n')
    agent.run("Make command keywords case-insensitive")
    assert agent.recovery_counts["action_blocks"] == 1
    event = next(event for event in agent.recovery_events if event["phase"] == "before_action")
    assert event["effective_decision"] == "BLOCK_ACTION"
    assert any('v == "NO"' in item for item in event["evidence"])


def test_semantic_risk_does_not_promote_solver_hypotheses_to_evidence():
    proposed = {
        "role": "assistant",
        "content": "I may make the record count variable.",
        "extra": {"actions": [{"command": "make record_count configurable"}]},
    }
    context = LiteLLMRecoveryPolicy._semantic_risk_context(
        [
            {
                "role": "assistant",
                "content": 'Maybe Ghost.start_line = 99 and value == "SPECULATION" matter.',
            },
            {
                "role": "tool",
                "content": "class Data:\n    start_line = 3\n",
            },
        ],
        proposed,
    )

    assert context["observed_dependent_offsets"] == ["Data.start_line = 3"]
    assert all("SPECULATION" not in item for item in context["observed_exact_value_consumers"])


def test_semantic_risk_ignores_recovery_feedback_as_repository_evidence():
    proposed = {
        "role": "assistant",
        "extra": {"actions": [{"command": "enable re.IGNORECASE"}]},
    }
    context = LiteLLMRecoveryPolicy._semantic_risk_context(
        [
            {"role": "tool", "content": 'if value == "NO":\n    mask(value)\n'},
            {
                "role": "tool",
                "content": (
                    "<returncode>126</returncode>\n"
                    "Action not executed by trajectory recovery. The proposed change "
                    "does not update if ghost == \"SPECULATION\"."
                ),
            },
        ],
        proposed,
    )

    assert context["observed_exact_value_consumers"] == ['if value == "NO":']


def test_workspace_semantic_evidence_excludes_changed_diff_lines():
    workspace = (
        " M parser.py\n"
        "diff --git a/parser.py b/parser.py\n"
        "index 123..456 100644\n"
        "--- a/parser.py\n"
        "+++ b/parser.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-if value == \"NO\":\n"
        "+if value.upper() == \"NO\":\n"
        " if type_ == \"data\":\n"
    )
    context = LiteLLMRecoveryPolicy._semantic_risk_context([], None, workspace)

    assert context["broad_matching_or_normalization_change"] is True
    assert all(not item.startswith("-") for item in context["observed_exact_value_consumers"])
    assert 'if value == "NO":' not in context["observed_exact_value_consumers"]
    assert 'if value.upper() == "NO":' not in context["observed_exact_value_consumers"]


def test_existing_call_result_comparisons_remain_visible_semantic_evidence():
    proposed = {
        "role": "assistant",
        "extra": {"actions": [{"command": "compile pattern with re.IGNORECASE"}]},
    }
    context = LiteLLMRecoveryPolicy._semantic_risk_context(
        [
            {
                "role": "tool",
                "content": (
                    '_token_re = r"VALUE|NO"\n'
                    'if first.upper() == "NO":\n'
                    '    mask(first)\n'
                    'if second.strip().casefold() == "no":\n'
                    '    mask(second)\n'
                    'if third.lower() != "value":\n'
                    '    reject(third)\n'
                ),
            }
        ],
        proposed,
    )

    assert context["observed_exact_value_consumers"] == [
        'if first.upper() == "NO":',
        'if second.strip().casefold() == "no":',
        'if third.lower() != "value":',
    ]


def test_new_patch_comparisons_do_not_manufacture_semantic_constraints():
    workspace = (
        "diff --git a/parser.py b/parser.py\n"
        "--- a/parser.py\n"
        "+++ b/parser.py\n"
        "@@ -1 +1,3 @@\n"
        " existing = True\n"
        "+compiled = re.compile(pattern, re.IGNORECASE)\n"
        '+if transformed == "sentinel":\n'
        "+    consume(transformed)\n"
    )

    context = LiteLLMRecoveryPolicy._semantic_risk_context(
        [
            {
                "role": "tool",
                "content": (
                    "compiled = re.compile(pattern, re.IGNORECASE)\n"
                    'if transformed == "sentinel":\n'
                ),
            }
        ],
        None,
        workspace,
    )

    assert context["observed_exact_value_consumers"] == []


def test_exact_consumer_normalizes_search_result_provenance():
    proposed = {
        "role": "assistant",
        "extra": {"actions": [{"command": "compile pattern with re.IGNORECASE"}]},
    }
    context = LiteLLMRecoveryPolicy._semantic_risk_context(
        [
            {
                "role": "tool",
                "content": (
                    '_token_re = r"VALUE|NO"\n'
                    'parser.py:42:    if value == "NO":\n'
                    'if value == "NO":\n'
                ),
            }
        ],
        proposed,
    )

    assert context["observed_exact_value_consumers"] == ['if value == "NO":']


def test_regex_semantic_risk_excludes_internal_group_label_comparisons():
    proposed = {
        "role": "assistant",
        "extra": {"actions": [{"command": "compile pattern with re.IGNORECASE"}]},
    }
    context = LiteLLMRecoveryPolicy._semantic_risk_context(
        [
            {
                "role": "tool",
                "content": (
                    '_data_re = r"VALUE|NO"\n'
                    '_type_re = rf"(?P<data>{{_data_re}})"\n'
                    'if type_ == "data":\n'
                    '    dispatch()\n'
                    'if value == "NO":\n'
                    '    mask(value)\n'
                ),
            }
        ],
        proposed,
    )

    assert context["observed_exact_value_consumers"] == ['if value == "NO":']


def test_submission_context_reconciles_only_previously_observed_constraints(tmp_path):
    agent = make_agent(tmp_path, [], ScriptedRecoveryPolicy({}))
    agent._active_scope = "parser.py"
    agent._semantic_constraints[("exact_value_consumer", "parser.py", 'if value == "no":')] = {
        "kind": "exact_value_consumer",
        "scope": "parser.py",
        "snippet": 'if value == "NO":',
        "priority": 0,
        "first_step": 1,
        "last_step": 1,
    }
    workspace = (
        "diff --git a/parser.py b/parser.py\n"
        "--- a/parser.py\n"
        "+++ b/parser.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-if value == \"NO\":\n"
        "+if value.upper() == \"NO\":\n"
        " if type_ == \"data\":\n"
    )
    local_risk = {
        "broad_matching_or_normalization_change": True,
        "variable_cardinality_or_layout_change": False,
        "observed_exact_value_consumers": [
            '-if value == "NO":',
            'if type_ == "data":',
        ],
        "observed_dependent_offsets": [],
        "observed_dependent_indices": [],
    }

    merged = agent._semantic_context_for_review(
        local_risk,
        proposed_message=None,
        workspace=workspace,
        include_local_constraints=False,
    )

    assert merged["observed_exact_value_consumers"] == []


def test_matching_recovery_blocks_once_then_allows_coordinated_edit(tmp_path):
    source = tmp_path / "parser.py"
    source.write_text(
        "import re\n"
        '_value_re = r"VALUE|NO"\n'
        '_type_re = rf"(?P<data>{{_value_re}})"\n'
        "compiled = re.compile(_type_re)\n"
        'if type_ == "data":\n'
        "    dispatch()\n"
        'if value == "NO":\n'
        "    mask(value)\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "parser.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    first_block = RecoveryDecision(
        phase="before_action",
        decision="BLOCK_ACTION",
        reason="Case-insensitive matching conflicts with the raw sentinel consumer.",
        evidence=['if value == "NO":'],
        next_focus="Update the exact sentinel comparison in parser.py.",
        unresolved_obligations=["Keep sentinel handling consistent with matching."],
        owning_handling_points=['if value == "NO":'],
    )
    policy = ScriptedRecoveryPolicy(
        {
            "before_action": [
                first_block,
                RecoveryDecision(phase="before_action", decision="CONTINUE"),
            ],
            "before_submission": [
                RecoveryDecision(phase="before_submission", decision="ALLOW_SUBMISSION")
            ],
        }
    )
    regex_only = (
        "python3 -c \"from pathlib import Path; p=Path('parser.py'); s=p.read_text(); "
        "p.write_text(s.replace('re.compile(_type_re)', "
        "'re.compile(_type_re, re.IGNORECASE)'))\""
    )
    coordinated = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "p = Path('parser.py')\n"
        "s = p.read_text()\n"
        "s = s.replace('re.compile(_type_re)', 're.compile(_type_re, re.IGNORECASE)')\n"
        "s = s.replace('if value == \"NO\":', 'if value.upper() == \"NO\":')\n"
        "p.write_text(s)\n"
        "PY\n"
    )
    agent = make_agent(
        tmp_path,
        [
            "sed -n '1,40p' parser.py",
            regex_only,
            coordinated,
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ],
        policy,
    )

    info = agent.run("Make the parser matching case-insensitive")

    assert info["exit_status"] == "Submitted"
    assert agent.recovery_counts["action_blocks"] == 1
    assert agent.recovery_counts["submission_blocks"] == 0
    assert 'if value.upper() == "NO":' in source.read_text()


def test_before_action_semantic_guard_does_not_promote_cardinality_hints(tmp_path):
    agent = make_agent(tmp_path, [], ScriptedRecoveryPolicy({}))
    risk = {
        "broad_matching_or_normalization_change": False,
        "variable_cardinality_or_layout_change": True,
        "observed_exact_value_consumers": [],
        "observed_dependent_offsets": ["Data.start_line = 3"],
        "observed_dependent_indices": [],
    }
    allow = RecoveryDecision(phase="before_action", decision="CONTINUE")

    first = agent._require_semantic_risk_acknowledgement(
        allow,
        phase="before_action",
        proposed_message={"extra": {"actions": [{"command": "edit first.py"}]}},
        workspace="",
        risk=risk,
    )
    second = agent._require_semantic_risk_acknowledgement(
        allow,
        phase="before_action",
        proposed_message={"extra": {"actions": [{"command": "edit second.py"}]}},
        workspace="",
        risk=risk,
    )

    assert first.decision == "CONTINUE"
    assert second.decision == "CONTINUE"


def test_semantic_guard_allows_action_that_updates_exact_constraint(tmp_path):
    agent = make_agent(tmp_path, [], ScriptedRecoveryPolicy({}))
    risk = {
        "broad_matching_or_normalization_change": False,
        "variable_cardinality_or_layout_change": True,
        "observed_exact_value_consumers": [],
        "observed_dependent_offsets": ["Data.start_line = 3"],
        "observed_dependent_indices": [],
    }
    proposed = {
        "extra": {
            "actions": [
                {
                    "command": (
                        "python -c \"from pathlib import Path; p=Path('layout.py'); "
                        "p.write_text(p.read_text().replace('start_line = 3', "
                        "'start_line = compute_start()'))\""
                    )
                }
            ]
        }
    }

    decision = agent._require_semantic_risk_acknowledgement(
        RecoveryDecision(phase="before_action", decision="CONTINUE"),
        phase="before_action",
        proposed_message=proposed,
        workspace="",
        risk=risk,
    )

    assert decision.decision == "CONTINUE"


def test_submission_guard_does_not_block_on_cardinality_hint_alone(tmp_path):
    agent = make_agent(tmp_path, [], ScriptedRecoveryPolicy({}))
    risk = {
        "broad_matching_or_normalization_change": False,
        "variable_cardinality_or_layout_change": True,
        "observed_exact_value_consumers": [],
        "observed_dependent_offsets": ["Data.start_line = 3"],
        "observed_dependent_indices": [],
    }
    narrative_allow = RecoveryDecision(
        phase="before_submission",
        decision="ALLOW_SUBMISSION",
        reason="Data.start_line = 3 is safe and can remain unchanged.",
    )

    decisions = [
        agent._require_semantic_risk_acknowledgement(
            narrative_allow,
            phase="before_submission",
            proposed_message=None,
            workspace="",
            risk=risk,
        )
        for _ in range(2)
    ]

    assert [decision.decision for decision in decisions] == [
        "ALLOW_SUBMISSION",
        "ALLOW_SUBMISSION",
    ]


def test_removed_constraint_is_not_carried_into_submission_context(tmp_path):
    agent = make_agent(tmp_path, [], ScriptedRecoveryPolicy({}))
    agent._semantic_constraints[("dependent_offset", "layout.py", "data.start_line = 3")] = {
        "kind": "dependent_offset",
        "scope": "layout.py",
        "snippet": "Data.start_line = 3",
        "priority": 0,
        "first_step": 1,
        "last_step": 1,
    }
    agent._active_scope = "layout.py"
    workspace = (
        "diff --git a/layout.py b/layout.py\n"
        "--- a/layout.py\n"
        "+++ b/layout.py\n"
        "@@ -1 +1 @@\n"
        "-    start_line = 3\n"
        "+    start_line = compute_start()\n"
    )

    constraints = agent._canonical_semantic_constraints(
        proposed_message=None,
        workspace=workspace,
    )

    assert constraints == []


def test_noop_mutation_does_not_advance_git_workspace_revision(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    policy = ScriptedRecoveryPolicy({})
    mutation = (
        "python -c \"from pathlib import Path; "
        "Path('source.py').write_text(Path('source.py').read_text())\""
    )
    agent = make_agent(
        tmp_path,
        [mutation, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"],
        policy,
        recovery_gate_submission=False,
    )

    agent.run("Leave an already-correct source unchanged")

    assert agent._workspace_revision == 0
    assert any(
        event.get("reason", "").startswith("Proposed mutation produced no workspace change")
        for event in agent.recovery_events
    )
