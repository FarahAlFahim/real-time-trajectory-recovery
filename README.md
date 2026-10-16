# TrajRec: Real-Time Recovery from Evidence-Commitment Divergence in LLM Coding Agents

This repository contains the replication package for *TrajRec*. LLM coding agents
can discover relevant repository evidence yet later make a decision that
disregards its implications. The paper calls this failure mode
*evidence-commitment divergence* (ECD). TrajRec is a runtime recovery mechanism
that maintains active evidence and unresolved obligations, and reviews the
agent's proposed **Direction**, **Edit**, and **Completion** commitments. When
deterministic runtime checks authorize an intervention, it withholds the proposed
action and returns focused feedback to the same running agent. It needs no gold
patch, reference trajectory, or hidden evaluation signal.

## Paper Results

All numbers are single-run Pass@1 under the official benchmark evaluation. The
aggregate results, including per-repository counts, are in [results/](results/)
(see [results/README.md](results/README.md)).

| Paper | Setting | BASE | TrajRec | Summary file |
|---|---|---|---|---|
| Table 1 | SWE-bench Lite (300), Mini-SWE-Agent V2, GPT-5-mini | 123 (41.0%) | 183 (61.0%) | `results/swe_bench_lite/mini_swe_agent_v2/gpt_5_mini/` |
| Table 3 | SWE-bench Pro V2 (100-task repository-stratified sample), Mini-SWE-Agent V2, GPT-5-mini | 53 (53.0%) | 66 (66.0%) | `results/swe_bench_pro/mini_swe_agent_v2/gpt_5_mini/` |
| Table 3 | SWE-bench Lite (300), Mini-SWE-Agent V2, DeepSeek-Flash | 217 (72.3%) | 225 (75.0%) | `results/swe_bench_lite/mini_swe_agent_v2/deepseek_flash/` |
| Table 3 | SWE-bench Lite (300), Codex, GPT-5-mini | 157 (52.3%) | 174 (58.0%) | `results/swe_bench_lite/codex/gpt_5_mini/` |

The progressive ablation (Table 2; SWE-bench Lite, Mini-SWE-Agent V2, GPT-5-mini)
is in the same summary as Table 1:

| Paper configuration | Solved | Pass@1 | `VARIANT` | `RECOVERY_REVIEW_BRANCHES` | `RECOVERY_GATE_MUTATIONS` | `RECOVERY_GATE_SUBMISSION` |
|---|---|---|---|---|---|---|
| BASE | 123 | 41.0% | `original` (`TRAJECTORY_RECOVERY_MODE="off"`) | – | – | – |
| + Direction | 147 | 49.0% | `ablation_direction` | `True` | `False` | `False` |
| + Edit | 168 | 56.0% | `ablation_direction_consistency` | `True` | `True` | `False` |
| + Completion (Full TrajRec) | 183 | 61.0% | `realtime_recovery` | `True` | `True` | `True` |

Recovery variants use `TRAJECTORY_RECOVERY_MODE="active"`. The Codex runs use
`VARIANT="codex_original"` with `CODEX_RECOVERY_MODE="off"` (BASE), and
`VARIANT="codex_realtime_recovery"` with `CODEX_RECOVERY_MODE="active"` (TrajRec).

The models were `openai/gpt-5-mini-2025-08-07` and `deepseek/deepseek-flash`
(runner `MODEL_NAME`). The reviewer reuses the solver model
(`RECOVERY_MODEL_NAME=None`). DeepSeek-Flash runs use the anti-leakage sandbox
(see [Runner configuration](#runner-configuration)). The SWE-bench Pro V2 sample is
listed in [data/swe_bench_pro_evaluation_subset_100.json](data/swe_bench_pro_evaluation_subset_100.json).

The summaries report overall and per-repository resolved counts. Per-instance
outcomes (the paired F→P / P→F counts and McNemar tests), per-instance patches and
trajectories, and the token and cost measurements behind Table 4 are not included
in this package.

## From the Paper to the Code

| Paper concept (Section 3) | Implementation |
|---|---|
| Evidence findings ei = ⟨claim, source, scope⟩ | Append-only evidence ledger `RecoveryAgent._evidence_ledger` (cited evidence with step, phase, scope, and scope revision) and the source-derived constraint index `_semantic_constraints` in [src/recovery/agent.py](src/recovery/agent.py) |
| Evidence state St = (Et, Ut) | `RecoveryAgent.recovery_state` (`retained_findings`, `unresolved_obligations`, `behavioral_specification`, ...), refreshed from each review by `_update_recovery_state` |
| Commitment recognizer φ | `RecoveryAgent.step`: repetition (`_repeat_candidate`), then departure from an explored scope (`_needs_branch_review`), then submission (`_is_submission`), which takes precedence over mutation (`_is_mutation`) |
| Direction / Edit / Completion checkpoints | Reviewer phases `branch_review` (plus auxiliary `repeat_exploration`), `before_action`, and `before_submission` ([src/recovery/decision.py](src/recovery/decision.py)) |
| Phase-conditioned LLM reviewer and review result Rt | `LiteLLMRecoveryPolicy` ([src/recovery/policy.py](src/recovery/policy.py)) returning a `RecoveryDecision` |
| Source-consistency signals and narrow consistency guard | `LiteLLMRecoveryPolicy._semantic_risk_context` and `RecoveryAgent._require_semantic_risk_acknowledgement` |
| Runtime validation of interventions | Edit and Completion blocks must pass `_valid_action_block` and `_valid_hard_submission_block` and are bounded by `RECOVERY_MAX_ACTION_BLOCKS` and `RECOVERY_MAX_SUBMISSION_BLOCKS`; Direction reviews are admitted by their trigger conditions in `_needs_branch_review`; malformed or failed reviews fall back to the permissive decision (`RECOVERY_FAIL_OPEN`) |
| Withhold the action and return recovery feedback to the same agent | `_block` and `_inject`: the native history is appended to, never rewritten |
| Mini-SWE-Agent V2 / Codex integration | [src/adapters/minisweagent/](src/adapters/minisweagent/) and [src/adapters/codex/](src/adapters/codex/) (native Codex lifecycle hooks) |

[docs/realtime_trajectory_recovery.md](docs/realtime_trajectory_recovery.md)
describes the checkpoint semantics and the saved audit trail in detail.

## Repository Layout

```text
.
├── run.py                         # CLI entry point: python run.py <phase>
├── requirements.txt
├── pytest.ini
├── data/
│   ├── swe_bench_lite/            # SWE-bench Lite: 300 instances in 12 per-repository JSON files
│   ├── swe_bench_pro/             # SWE-bench Pro: 642 instances in 11 per-repository JSON files
│   └── swe_bench_pro_evaluation_subset_100.json  # The 100-task SWE-bench Pro V2 sample
├── docs/
│   └── realtime_trajectory_recovery.md   # Checkpoints, semantics, outputs
├── results/                       # Aggregate Pass@1 summaries behind Tables 1-3
│   ├── swe_bench_lite/{mini_swe_agent_v2/{gpt_5_mini,deepseek_flash},codex/gpt_5_mini}/
│   └── swe_bench_pro/mini_swe_agent_v2/gpt_5_mini/
├── third_party_licenses/
│   ├── mini-swe-agent-LICENSE.md  # mini-swe-agent (MIT)
│   ├── SWE-bench-LICENSE.md       # SWE-bench, source of data/swe_bench_lite/ (MIT)
│   └── SWE-bench_Pro-LICENSE.md   # SWE-bench Pro, source of data/swe_bench_pro/ (MIT)
├── src/
│   ├── recovery/                  # TrajRec core, shared by both adapters
│   │   ├── decision.py            # RecoveryDecision, RecoveryPhase, RecoveryPolicy protocol
│   │   ├── policy.py              # LiteLLM reviewer: prompts, evidence packing, parsing
│   │   └── agent.py               # RecoveryAgent: state, boundaries, gates, pruning, serialization
│   ├── adapters/
│   │   ├── minisweagent/
│   │   │   ├── swebench.py        # process_instance_with_recovery() for mini-swe-agent SWE-bench
│   │   │   └── docker.py          # Docker environment with optional git-history stripping
│   │   └── codex/
│   │       ├── controller.py      # Host-side Codex lifecycle-event controller
│   │       ├── hook_bridge.py     # In-container synchronous hook/IPC bridge
│   │       └── apply_patch_salvage.py  # Optional empty-patch salvage helper
│   ├── runners/
│   │   ├── minisweagent_swebench_lite.py  # SWE-bench Lite: Original, ablations, Realtime Recovery
│   │   ├── minisweagent_swebench_pro.py   # SWE-bench Pro V2
│   │   ├── codex_swebench_lite.py         # Codex Original and Realtime Recovery
│   │   └── common.py              # Dataset loading, instance selection, results layout, exit statuses
│   ├── evaluation/
│   │   ├── swebench_lite.py       # SWE-bench Lite evaluator
│   │   └── swebench_pro.py        # SWE-bench Pro V2 evaluator (official verifiers)
│   └── utils/                     # Logging, JSON/JSONL I/O, Docker cleanup helpers
└── tests/
    ├── test_recovery.py           # mini-swe-agent recovery regression tests
    ├── test_codex_recovery_controller.py  # Codex adapter/controller tests
    └── test_docker_environment.py # Git-history stripping environment tests
```

Apart from the benchmark inputs in `data/` and the aggregate summaries in
`results/`, the package includes no generated repairs, trajectories, per-instance
evaluation outputs, Docker images, API credentials, or virtual environments.

### Data

`data/swe_bench_lite/` contains the official SWE-bench Lite test split:
all 300 instances, grouped into one JSON list per repository, with the standard
fields (`instance_id`, `repo`, `base_commit`, `problem_statement`, `patch`,
`test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, ...). `data/swe_bench_pro/`
contains the SWE-bench Pro instances (642 in total) in the same per-repository
form, including the fields the Pro runner uses (`docker_image`, `requirements`,
`interface`, `fail_to_pass`, `pass_to_pass`, ...). Problem statements are kept
verbatim, including any file paths quoted in the original upstream issues.

The instances are redistributed unchanged from
[SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) and
[SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro). See
[Third-party code and data](#third-party-code-and-data) for their licenses.

`data/swe_bench_pro_evaluation_subset_100.json` lists the 100 instance IDs of the
preselected, repository-stratified SWE-bench Pro V2 sample used in the paper,
grouped by repository. The sample was frozen before any run and is identical for
BASE and TrajRec. Its per-repository counts match
`results/swe_bench_pro/mini_swe_agent_v2/gpt_5_mini/combined_eval_summary.json`.
To run the sample for one repository, set the Pro runner's `DATASET_JSON` to that
repository's file and `INSTANCES_FILTER` to the IDs listed for it.

## Setup

Use Python 3.11+ (the experiments used Python 3.12.3).

TrajRec runs on top of mini-swe-agent 2.1.0 (Mini-SWE-Agent V2 in the paper).
Install the dependencies from this repository's root:

```bash
pip install -r requirements.txt
```

The recovery reviewer is called through LiteLLM. Model credentials are read from
the environment and must never be committed:

```bash
export OPENAI_API_KEY="..."
```

SWE-bench runs require Docker.

For the Codex runner, also install the OpenAI Codex Python SDK (`openai_codex`)
and place a Linux x86_64 `codex` binary at `bin/codex-linux-x86_64` (or set
`CONFIG["CODEX_BIN"]`). The binary is copied into each task container. On Apple
Silicon, the SDK's bundled macOS binary does not work inside the amd64 container.

## Running

All phases are launched from the repository root:

```bash
python run.py <phase>
```

| Phase | Module | Description |
|---|---|---|
| `minisweagent_lite` | `src/runners/minisweagent_swebench_lite.py` | mini-swe-agent on SWE-bench Lite. |
| `minisweagent_pro` | `src/runners/minisweagent_swebench_pro.py` | mini-swe-agent on SWE-bench Pro V2. |
| `codex_lite` | `src/runners/codex_swebench_lite.py` | Codex CLI on SWE-bench Lite. |
| `eval_lite` | `src/evaluation/swebench_lite.py` | Evaluate SWE-bench Lite patches. |
| `eval_pro` | `src/evaluation/swebench_pro.py` | Evaluate SWE-bench Pro V2 patches with the official verifiers. |

Each runner and evaluator is configured by editing the `CONFIG` dict at the top
of its module, which contains the complete evaluated configuration. Before
running, set `DATASET_JSON` to the per-repository file to run (for example
`data/swe_bench_lite/django__django.json` or
`data/swe_bench_pro/ansible__ansible.json`), and set `RUN_NAME` and
`RESULTS_ROOT`. Relative paths in a runner's `CONFIG` resolve against the
repository root. Remaining command-line options are
forwarded to the phase. For example:

```bash
python run.py eval_lite --limit 10
python run.py eval_pro --dry-run
```

With `RUN_MODE="per_instance"` (the default), each instance runs in a
subprocess that re-launches the runner module with a per-instance watchdog
timeout.

### Output layout

Runners write to `<RESULTS_ROOT>/<RUN_NAME>/<VARIANT>/`: the run manifest and
instance lists (`run_manifest.json`, `instances.json`, `instances_selected.json`)
at the variant root, and under `repair/` one `<instance_id>/<instance_id>.traj.json`
and `<instance_id>.patch` per instance plus `preds.json` and
`run_batch_exit_statuses.yaml`. Recovery
trajectories add a top-level `trajectory_recovery` object with every reviewer
decision, the evidence ledger, scope history, and reviewer cost (see
[docs/realtime_trajectory_recovery.md](docs/realtime_trajectory_recovery.md)).

The evaluators read the same layout: set `TARGETS` to `<RESULTS_ROOT>/<RUN_NAME>`
and `VARIANTS` to the variant directories to compare. They resolve relative paths
against the current working directory, so run them from the repository root. The
SWE-bench Pro evaluator requires a local checkout of
[scaleapi/SWE-bench_Pro-os](https://github.com/scaleapi/SWE-bench_Pro-os); set
`TASKS_ROOT` (or `SWE_BENCH_PRO_V2_TASKS`) to its `v2/tasks` directory.

## Runner configuration

The evaluated implementation exposes the following principal recovery controls:

| Option | Evaluated value | Description |
|---|---|---|
| `TRAJECTORY_RECOVERY_MODE` | `"active"` | `off`, `shadow` (record decisions only), or `active`. |
| `RECOVERY_MODEL_NAME` | `None` | Independent reviewer model; `None` reuses `MODEL_NAME`. |
| `RECOVERY_MODEL_TIMEOUT_SECONDS` | `180` | Reviewer call timeout. |
| `RECOVERY_CONTEXT_TOKEN_LIMIT` | `250_000` | Reviewer context ceiling. |
| `RECOVERY_MAX_ACTION_BLOCKS` | `3` | Maximum Edit (mutation) blocks per task. |
| `RECOVERY_MAX_SUBMISSION_BLOCKS` | `2` | Maximum Completion (submission) blocks per task. |
| `RECOVERY_REPEAT_THRESHOLD` | `2` | Prior matching explorations before a repetition review. |
| `RECOVERY_FAIL_OPEN` | `True` | Reviewer/API failures never break the solver run. |
| `RECOVERY_REVIEW_BRANCHES` / `RECOVERY_GATE_MUTATIONS` / `RECOVERY_GATE_SUBMISSION` | `True` | Enable the Direction, Edit, and Completion reviews (see the Table 2 mapping above). |

`DISABLE_NETWORK` and `STRIP_GIT_HISTORY` control the anti-leakage sandbox
(network cut, and git history reduced to a single base commit). In the SWE-bench
Lite runner both default to automatic: they are enabled for models listed in
`STRICT_SANDBOX_MODEL_MARKERS` (e.g. DeepSeek) and disabled otherwise. The
SWE-bench Pro runner cuts the network by default. In the mini-swe-agent runners,
`STRIP_GIT_HISTORY` is implemented by `src/adapters/minisweagent/docker.py`, which
the runners register as mini-swe-agent's `"docker"` environment; with stripping
disabled it behaves exactly like the stock environment.

## Important semantic detail

TrajRec uses logical pruning. A branch labelled `IRRELEVANT` can be excluded
from subsequent reviewer evidence packs, but the repair agent's native
conversation history is never deleted or rewritten. When an action is blocked,
the same agent receives a failed-tool observation and focused recovery feedback
and then continues in its existing context.

The Codex adapter reuses the same recovery policy and controller behavior:
`src/adapters/codex/controller.py` subclasses `RecoveryAgent` and drives it from
native Codex lifecycle hooks (`SessionStart`, `PreToolUse`, `PostToolUse`, `Stop`)
delivered by `hook_bridge.py` through a bind-mounted IPC directory. The benchmark
checkout is left unmodified; only the Codex binary and the hook files are copied
into the task container.

## Implementation Details Beyond the Paper

The following behaviors of the evaluated implementation are not described in the
paper. They apply only when recovery is active (`TRAJECTORY_RECOVERY_MODE="active"`
or `CODEX_RECOVERY_MODE="active"`); BASE runs use the stock agents.

- **Solver task note (mini-swe-agent).** A short fixed paragraph describing how to
  respond to recovery feedback is appended to the solver's task
  (`RecoveryAgent.run`).
- **Sequential tool calls (mini-swe-agent).** `parallel_tool_calls` is set to
  `False` so each proposed action can be reviewed before it executes; BASE keeps
  the mini-swe-agent default.
- **Deterministic invariants.** A solver `git commit` is always withheld, because
  the benchmark requires an uncommitted patch. With the Completion review enabled,
  the submitted `patch.txt` is rebuilt against the initial repository revision, and
  an empty patch is withheld as a bounded Completion block without a reviewer call.
  Both adapters apply these invariants.
- **Budgets.** The per-instance step and cost limits apply to the solver; reviewer
  calls are tracked separately (`reviewer_calls` and `reviewer_cost` in each
  trajectory's `trajectory_recovery` object) and are not charged to those limits.
- **Direction and repetition reviews.** Prunes and repetition blocks follow the
  reviewer's decision once a review is triggered. They are not subject to the Edit
  and Completion evidence checks, and the number of distinct-scope prunes is not
  capped. The repetition check also covers repeated or oscillating edits, so it is
  already active in the + Direction ablation tier.
- **State refresh.** Each review replaces `recovery_state` with the reviewer's
  updated findings and obligations; a failed-open review or a deterministic
  invariant therefore resets them, while the evidence ledger is kept.

## Tests

```bash
python -m pytest
```

The tests use mini-swe-agent's deterministic test models and a local
environment; they need neither Docker nor API credentials.

## Third-party code and data

The `third_party_licenses/` directory contains the licenses of the third-party
projects this package builds on or redistributes data from:

| File | Project | Used for |
|---|---|---|
| `mini-swe-agent-LICENSE.md` | [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (MIT) | The integration code builds on it. It is installed as a dependency, not vendored. |
| `SWE-bench-LICENSE.md` | [SWE-bench](https://github.com/SWE-bench/SWE-bench) (MIT) | Source of the SWE-bench Lite instances in `data/swe_bench_lite/`; its harness is used by the Lite evaluator. |
| `SWE-bench_Pro-LICENSE.md` | [SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os) (MIT) | Source of the SWE-bench Pro instances in `data/swe_bench_pro/`; its official verifiers are used by the Pro evaluator. |

Benchmark instances are derived from issues, pull requests, and code of public
open-source repositories. Their problem statements, patches, and tests remain
subject to the licenses of those original repositories. This repository does not
vendor mini-swe-agent, SWE-bench, SWE-bench Pro, or OpenAI Codex.
