# Repair Results

This directory contains the aggregate Pass@1 results reported in the paper
(Tables 1–3). Results are organized by benchmark, repair agent, and model.

```text
.
├── swe_bench_lite/
│   ├── mini_swe_agent_v2/
│   │   ├── gpt_5_mini/
│   │   │   └── combined_eval_summary.json   # Tables 1 and 2
│   │   └── deepseek_flash/
│   │       └── combined_eval_summary.json   # Table 3, model generalization
│   └── codex/
│       └── gpt_5_mini/
│           └── combined_eval_summary.json   # Table 3, agent generalization
└── swe_bench_pro/
    └── mini_swe_agent_v2/
        └── gpt_5_mini/
            └── combined_eval_summary.json   # Table 3, benchmark generalization
```

Each `combined_eval_summary.json` reports the benchmark, repair agent, model,
evaluated variants, the overall resolved count and Pass@1 per variant
(`per_variant`), and the resolved count and Pass@1 per repository and variant
(`per_repository`). The SWE-bench Lite summaries cover all 300 instances. The
SWE-bench Pro summary (`"dataset_name": "SWE-bench Pro"`) covers the 100-task
repository-stratified SWE-bench Pro V2 sample listed in
[../data/swe_bench_pro_evaluation_subset_100.json](../data/swe_bench_pro_evaluation_subset_100.json).

## Variants

Variant keys are the runner's `VARIANT` names. Their labels in the summaries map
to the paper as follows:

| Variant key | Summary label | Paper name |
|---|---|---|
| `original` | Original | BASE |
| `ablation_direction` | Direction Recovery | + Direction |
| `ablation_direction_consistency` | Direction Recovery + Edit Consistency | + Edit |
| `realtime_recovery` | Full TrajRec | + Completion (Full TrajRec) / TrajRec |
| `codex_original` | Codex Original | BASE (Codex) |
| `codex_realtime_recovery` | Codex + Full TrajRec | TrajRec (Codex) |

## Results

| Benchmark | Agent | Model | Variants | BASE | TrajRec |
|---|---|---|---|---|---|
| SWE-bench Lite (300) | Mini-SWE-Agent V2 | GPT-5-mini | all four tiers | 123 (41.0%) | 183 (61.0%) |
| SWE-bench Lite (300) | Mini-SWE-Agent V2 | DeepSeek-Flash | BASE, TrajRec | 217 (72.3%) | 225 (75.0%) |
| SWE-bench Lite (300) | Codex | GPT-5-mini | BASE, TrajRec | 157 (52.3%) | 174 (58.0%) |
| SWE-bench Pro V2 (100) | Mini-SWE-Agent V2 | GPT-5-mini | BASE, TrajRec | 53 (53.0%) | 66 (66.0%) |

The intermediate ablation tiers on SWE-bench Lite with GPT-5-mini resolve 147
(49.0%, + Direction) and 168 (56.0%, + Edit) tasks.

Per-instance outcomes, patches, and trajectories, as well as the token and cost
measurements behind Table 4, are not included in this directory.
