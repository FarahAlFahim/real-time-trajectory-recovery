# Real-time trajectory recovery

This document describes how TrajRec reviews an in-progress repair trajectory. The
recovery path is opt-in in the mini-swe-agent runners
(`src/runners/minisweagent_swebench_lite.py`, `src/runners/minisweagent_swebench_pro.py`)
and in the Codex runner (`src/runners/codex_swebench_lite.py`). It does not
generate or prepend an enhanced static report. The solver starts from the original
problem statement, and a separate reviewer can intervene while the trajectory is
being produced.

## Paper terminology

| Paper commitment | Reviewer phase | Enabled by |
|---|---|---|
| DIRECTION | `branch_review`, plus the auxiliary `repeat_exploration` | `RECOVERY_REVIEW_BRANCHES` (repetition review runs whenever recovery is active) |
| EDIT | `before_action` | `RECOVERY_GATE_MUTATIONS` |
| COMPLETION | `before_submission` | `RECOVERY_GATE_SUBMISSION` |

The progressive ablation in the paper enables these one at a time:
`ablation_direction` (Direction), `ablation_direction_consistency` (Direction +
Edit), and `realtime_recovery` (Direction + Edit + Completion, the full system).
The Codex adapter (`src/adapters/codex/controller.py`) applies the same phases
through native Codex hooks: Direction and Edit at `PreToolUse`, and Completion at
`Stop`.

## Modes

Set `TRAJECTORY_RECOVERY_MODE` in the runner's `CONFIG`:

- `"off"` uses mini-swe-agent's stock `process_instance()` exactly as before.
- `"shadow"` records what the reviewer would have done without changing solver
  behavior. Use this to measure false-positive interventions before an active
  experiment.
- `"active"` permits focused recovery messages and preemptive action gates.

For clean result separation, use `VARIANT="realtime_recovery"` for active runs.
The wrapper creates that result directory alongside the existing variants.

In active mode, the mini-swe-agent runners also set `parallel_tool_calls=False`,
so each proposed action can be reviewed before it executes, and append a short
fixed note about recovery feedback to the solver's task.

Typical active settings:

```python
"VARIANT": "realtime_recovery",
"TRAJECTORY_RECOVERY_MODE": "active",
"RECOVERY_MODEL_NAME": None,  # Reuse MODEL_NAME.
"RECOVERY_CONTEXT_TOKEN_LIMIT": 250_000,
"RECOVERY_MAX_SUBMISSION_BLOCKS": 2,
"RECOVERY_MAX_ACTION_BLOCKS": 3,
"RECOVERY_REPEAT_THRESHOLD": 2,
"RECOVERY_FAIL_OPEN": True,
"RECOVERY_REVIEW_BRANCHES": True,
"RECOVERY_GATE_MUTATIONS": True,
"RECOVERY_GATE_SUBMISSION": True,
```

Run the wrapper normally:

```bash
python run.py minisweagent_lite
```

## Online checkpoints and scope boundaries

The recovery path doesn't judge a new file before it has been read. It tracks an
active exploration scope and reviews that completed scope when the solver tries
to enter another explored file. Mutations and submissions use their own single
review phase instead of first triggering a redundant branch review:

```text
read A -> continue within A -> propose B
                              |
                              +-- review observed scope A before executing B
```

1. **At an explored-scope boundary.** `KEEP` retains relevant findings and
   permits movement. `PRUNE` explicitly labels the completed branch
   `IRRELEVANT` and returns a focused redirection; it never deletes trajectory
   history or evidence. Related producers, consumers, readers, writers, tests,
   parents, and lifecycle operations remain one causal branch even across
   files, so relevant but incomplete exploration is never blocked at a file
   boundary. An unresolved overall repair doesn't by itself make a completed
   scope relevant. Directory-only navigation isn't treated as a branch, there
   is no global cap on distinct branch prunes, and each concrete file is
   reviewed at most once per workspace revision.
2. **On repeated exploration.** An exact command is reviewed on its third use;
   sustained revisits to the same file/scope are also reviewed. A repeat is
   allowed when it reads a new range, asks a new question, validates a new
   hypothesis, or answers prior feedback. Otherwise it is interrupted and the
   ledger directs the solver to an unresolved obligation. Cumulative repetition
   history is never reset by a mutation. Each touched file receives one
   post-mutation validation allowance; later revisits are again subject to the
   cumulative detector. Executed mutation signatures are also retained, so a
   repeated edit or an A -> B -> A oscillation is reviewed before execution.
3. **Before a likely source mutation.** The reviewer may prevent a premature
   edit. Patch packaging, `git add`, `git diff > patch.txt`, temporary repro
   files, and other bookkeeping aren't mutation-gated. Once the configured
   action-block limit is reached, later edits execute without additional
   reviewer calls. A blocked tool call receives a normal tool-result message,
   preserving the model/tool protocol and returning control to the solver.
4. **Before submission.** Submission is allowed by default. A hard block needs a
   stable blocker ID, concrete evidence, a demonstrated path or owning handling
   point, an unresolved report-grounded obligation, and an executable next
   focus. Vague claims that more exploration may help are advisory only.

Mutation and submission reviews also receive a compact semantic-risk context.
In addition to broad matching/normalization changes, it detects changes that
make a count or layout variable. Exact code constraints are captured when they
are observed, with file provenance, and later reactivated only when the proposed
diff touches the relevant file. The nearest hardcoded offset or literal index is
prioritized over generic framework paths; an allow decision must reconcile that
one concrete handling point. This catches, for example, making a collection's
cardinality configurable while leaving its reader offset hardcoded without
turning the review into an exhaustive audit of unrelated variants. Prose and
comments are excluded from the deterministic constraint index. Merely copying a
constraint into the review's evidence list is not reconciliation; the reviewer
must account for it in its reasoning, demonstrated path, or owning handling
point. The deterministic guard redirects only once for an unchanged workspace
and mutation target; later blocking requires a fresh, concrete reviewer finding.

In active mode, a solver-issued `git commit` is deterministically blocked. The
benchmark contract requires an uncommitted patch, and moving `HEAD` would
otherwise hide a valid repair from ordinary patch packaging. This invariant
does not consume the semantic mutation-block budget and is inactive in the
stock recovery-off path.

Submission blocks are separately limited (two by default). After a block, the
next audit receives the prior blocker plus all intervening solver actions and
explanations. The solver can resolve the concern or explain with repository/test
evidence why it is irrelevant. When the limit is reached, submission proceeds.
When the standard submission command reads `patch.txt`, recovery first rebuilds
that artifact relative to the immutable repository revision captured before the
solver's first action, including committed, staged, and unstaged solver changes
and excluding the artifact itself. This deterministic packaging repair doesn't
use a semantic block. Thus a commit that somehow predates or bypasses the active
guard still cannot erase the submitted patch. If the rebuilt artifact is empty,
it is handled as a normal bounded submission blocker rather than silently
submitting no patch.

The native solver message history is never deleted or rewritten. The reviewer
doesn't repeatedly receive that entire accumulated history: each review gets
the retained structured summary, observations produced since the previous
review, the latest reviewer state per relevant file, and the small exact-code
constraint index. Reviewer decisions remain in an append-only ledger for the
saved audit trajectory, but older paraphrases from the same scope are not fed
back into every online call. Scopes labelled `IRRELEVANT` are excluded unless
they become active again. There are no task-independent per-category item
limits. The configured total context budget (250,000 tokens by default) remains
a final ceiling; the trajectory records total, relevant, superseded, irrelevant,
nonlocal, and included entry counts. On true overflow, large tool observations are
omitted and solver thoughts are retained; arbitrary beginning/end slices aren't
used. Recovery is therefore a logical prune that remains auditable without
creating invalid tool-call histories or repeatedly charging for stale context.

## Output and evaluation

Recovery trajectories use the normal `.traj.json` location and include a
top-level `trajectory_recovery` object with:

- mode and reviewer model;
- every recommended and effective decision;
- phase, reason, evidence, and next focus;
- retained findings, behavioral specification, and unresolved obligations;
- append-only evidence ledger with scope provenance;
- scope history with explicit relevance labels, irrelevant scopes (plus the
  legacy `pruned_scopes` alias), and repeated-exploration counts both cumulative
  and since the latest mutation;
- mutation-signature counts, pending one-shot validation allowances, and an
  evidence-selection digest;
- submission blockers and solver feedback since the last block;
- separate branch/repetition/action/submission counts, reviewer calls, and
  reviewer cost.

The trajectory is checkpointed after every solver step. Reviewer failures are
fail-open by default and are recorded with `failed_open: true`; set
`RECOVERY_FAIL_OPEN=False` for strict debugging.

For experiments, first run `shadow` on a representative subset, label whether
each suggested intervention was warranted, then run `active` with the same
solver configuration and original reports. Report solver cost and reviewer cost
separately.
