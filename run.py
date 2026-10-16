#!/usr/bin/env python3
"""CLI entry point for the real-time trajectory recovery (TrajRec) artifact.

Usage:
    python run.py <phase> [options]

Phases:
    minisweagent_lite     Run mini-swe-agent on SWE-bench Lite (Original, ablations, Realtime Recovery)
    minisweagent_pro      Run mini-swe-agent on SWE-bench Pro V2
    codex_lite            Run Codex CLI on SWE-bench Lite (Original, Realtime Recovery)
    eval_lite             Evaluate generated patches on SWE-bench Lite
    eval_pro              Evaluate generated patches on SWE-bench Pro V2

Each runner and evaluator is configured by editing the CONFIG dict at the top of
its module. Remaining command-line options are forwarded to the selected phase,
which executes exactly as if its module were run as a script.
"""

import runpy
import sys

PHASES = {
    "minisweagent_lite": "src.runners.minisweagent_swebench_lite",
    "minisweagent_pro": "src.runners.minisweagent_swebench_pro",
    "codex_lite": "src.runners.codex_swebench_lite",
    "eval_lite": "src.evaluation.swebench_lite",
    "eval_pro": "src.evaluation.swebench_pro",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    phase = sys.argv[1]
    if phase not in PHASES:
        print(f"Unknown phase: {phase}. Choose one of: {', '.join(PHASES)}", file=sys.stderr)
        sys.exit(1)

    # Drop the phase so the module sees the same argv as a direct script invocation.
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    runpy.run_module(PHASES[phase], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
