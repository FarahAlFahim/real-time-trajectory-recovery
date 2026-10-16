#!/usr/bin/env python3
"""eval_swebench_resolved.py

Docker-based SWE-bench evaluation for SWE-agent outputs.

This script mirrors the "one-click" UX and output layout of
`Agentless/eval_agentless_swebench.py`, but uses SWE-agent artifacts as the patch
source.

Key feature (variants instance-first)
------------------------------------
When CONFIG['VARIANTS'] and CONFIG['OUTPUT_ROOT'] are set, this script evaluates
multiple variants *per instance*, pulling each SWE-bench instance image at most
once per instance and (optionally) deleting that image immediately after the
instance finishes (saving disk space).

Output layout (matches Agentless evaluator)
------------------------------------------
Per variant:
    <OUTPUT_ROOT>/<variant>/eval_summary.json
    <OUTPUT_ROOT>/<variant>/repair/eval_work/<instance_id>/docker_output.log

Combined:
    <OUTPUT_SUMMARY_JSON>

Patch candidates
---------------
For a SWE-agent run dir (results/<RUN>/<VARIANT>):
- Candidate #1 comes from repair/preds.json if present.
- Additional candidates come from repair/<instance_id>/*.patch.
The script reports both pass@1 (resolved_at_1) and pass@any (resolved_any)
when CONFIG['EVALUATE_ANY_REPAIR'] is enabled.

Requirements:
- Docker available locally.
- SWE-bench instance images available/pullable.

Notes on "pytest"
-----------------
This evaluator runs tests *inside* the SWE-bench instance Docker image. The test
runner invoked in the container is `pytest -q` over FAIL_TO_PASS and PASS_TO_PASS
files (as listed in instances.jsonl).

You can optionally switch to a more SWE-bench-faithful "harness-style" mode that:
- runs the repo/version-specific SWE-bench `test_cmd` (plus directives)
- grades using SWE-bench harness log parsers over FAIL_TO_PASS/PASS_TO_PASS
This requires the `swebench` harness to be importable in the Python environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.docker import prune, stop_and_remove_containers_for_image


# =========================
# ONE-CLICK CONFIG (edit these)
# =========================

CONFIG: Dict[str, Any] = {
    # Base result folders. In variants mode, each variant is searched under
    # TARGET/VARIANT, in TARGET list order (fallbacks).
    "TARGETS": [
        # "results/gpt_5_mini/psf__requests"
        # "results/minimax2.5/pallets__flask"
        "results_trajrec/deepseek_flash/pallets__flask"
        # "results_trajrec/gpt_5_mini/matplotlib__matplotlib"
        # "results_trajrec/codex/psf__requests"
    ],
    # Evaluate these variant subfolders under each TARGET.
    "VARIANTS": [
        "original",
        "realtime_recovery",
        # "ablation_direction",
        # "ablation_direction_consistency",
        # "codex_original",
        # "codex_realtime_recovery",
    ],
    # Output paths are derived from the (single) TARGETS entry + this suffix, right after
    # this dict. Change OUTPUT_SUFFIX here to write to a different sibling folder
    # (e.g. "_tests1", "_tests2") without touching TARGETS:
    #   OUTPUT_ROOT         = <TARGETS[0]><OUTPUT_SUFFIX>        (per-variant summaries/logs)
    #   OUTPUT_SUMMARY_JSON = <OUTPUT_ROOT>/combined_eval_summary.json
    "OUTPUT_SUFFIX": "_tests",
    "KEEP_LOGS": True,
    # Write JSON summaries progressively every N instances (1 = after each instance).
    "PROGRESS_WRITE_EVERY": 1,
    "LIMIT": None,

    # Optional: evaluate only these instance_ids (in this exact order) when non-empty.
    # Example: ["scikit-learn__scikit-learn-10949"]
    "INSTANCE_IDS": [
        # "sympy__sympy-14024",
        # "sympy__sympy-17022",
        # "sympy__sympy-19487",
        # "psf__requests-863",
        # "psf__requests-3362",
    ],
    # If True, evaluate all patch candidates found for an instance and compute pass@any.
    "EVALUATE_ANY_REPAIR": True,

    # Which instances to treat as the expected evaluation set:
    # - "selected": use instances_selected.jsonl when present (falls back to instances.jsonl)
    # - "all": always use instances.jsonl (ignores instances_selected.jsonl)
    "INSTANCE_LIST": "all",

    # Where to source per-instance metadata needed for docker evaluation:
    # - "instances_jsonl": use instances.jsonl only
    # - "dataset": build metadata from SWE-bench dataset rows (requires swebench)
    # - "auto": prefer instances.jsonl but fall back to dataset when missing
    #
    # If your instances.jsonl is partially wrong/incomplete, set this to "dataset".
    "META_SOURCE": "auto",

    # Evaluation style:
    # - "pytest": run `pytest -q` over FAIL_TO_PASS and PASS_TO_PASS and score by exit codes.
    # - "harness": run SWE-bench `test_cmd` + directives and grade via SWE-bench harness parsers.
    #   (falls back to "pytest" per-instance if the harness/dataset metadata isn't available)
    "EVAL_STYLE": "harness", # "harness" or "pytest"
    # SWE-bench dataset (used to derive test_cmd + directives in EVAL_STYLE="harness")
    "DATASET_NAME": "princeton-nlp/SWE-bench_Lite",
    "SPLIT": "test",
    # Pull missing SWE-bench instance images.
    "PULL_IMAGES": True,
    # Delete the instance image after the instance finishes (after all variants).
    "DELETE_IMAGES_AFTER_EVAL": False, # True | False
    # Docker run timeout per candidate evaluation (seconds).
    "TIMEOUT": 1800,
}

# Derive output paths from the (single) primary target + OUTPUT_SUFFIX, so you only edit
# TARGETS (and OUTPUT_SUFFIX when you want a different sibling folder).
# e.g. TARGETS=["results_trajrec/gpt_5_mini/django__django"], OUTPUT_SUFFIX="_tests"
#   -> OUTPUT_ROOT         = "results_trajrec/gpt_5_mini/django__django_tests"
#   -> OUTPUT_SUMMARY_JSON = "results_trajrec/gpt_5_mini/django__django_tests/combined_eval_summary.json"
_primary_target = CONFIG["TARGETS"][0].rstrip("/")
CONFIG["OUTPUT_ROOT"] = f"{_primary_target}{CONFIG['OUTPUT_SUFFIX']}"
CONFIG["OUTPUT_SUMMARY_JSON"] = f"{CONFIG['OUTPUT_ROOT']}/combined_eval_summary.json"


@dataclass(frozen=True)
class VariantTarget:
    label: str
    source_run_dirs: List[Path]
    output_run_dir: Path


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Best-effort atomic JSON write (write temp file then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _resolve_existing_dir(path: str) -> Path:
    p = Path(os.path.expanduser(path)).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Directory does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {p}")
    return p


def _build_variant_targets_from_config(
    *, base_targets: List[str], variants: List[str], output_root: str
) -> List[VariantTarget]:
    bases = [_resolve_existing_dir(t) for t in base_targets]
    out_root = Path(os.path.expanduser(output_root)).resolve()
    targets: List[VariantTarget] = []
    for v in variants:
        v = str(v).strip()
        if not v:
            continue
        src_dirs = [b / v for b in bases]
        out_dir = out_root / v
        targets.append(VariantTarget(label=v, source_run_dirs=src_dirs, output_run_dir=out_dir))
    return targets


@dataclass(frozen=True)
class InstanceMeta:
    instance_id: str
    image_name: str
    base_commit: str
    test_patch: str
    fail_to_pass: List[str]
    pass_to_pass: List[str]


def swebench_image_name(instance_id: str) -> str:
    # Docker doesn't allow double underscore, so replace it with _1776_.
    iid_docker_compatible = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{iid_docker_compatible}:latest".lower()


def _instance_meta_from_dataset_row(row: Optional[Dict[str, Any]], instance_id: str) -> Optional[InstanceMeta]:
    """Best-effort build InstanceMeta from a SWE-bench dataset row."""

    if not row:
        return None

    iid = str(row.get("instance_id") or "")
    if not iid.strip():
        iid = str(row.get("instance_id") or instance_id or "")
    if not iid.strip():
        return None

    base_commit = str(row.get("base_commit") or "")
    if not base_commit.strip():
        return None

    extra = row.get("extra_fields") if isinstance(row.get("extra_fields"), dict) else {}
    if not isinstance(extra, dict):
        extra = {}

    test_patch_raw = extra.get("test_patch") or row.get("test_patch")
    test_patch = str(test_patch_raw) if isinstance(test_patch_raw, str) else ""
    if not test_patch.strip():
        return None

    # These can be lists or JSON-encoded list strings.
    try:
        fail_to_pass = _parse_json_list_field(
            extra.get("FAIL_TO_PASS") or row.get("FAIL_TO_PASS"), "FAIL_TO_PASS", iid
        )
        pass_to_pass = _parse_json_list_field(
            extra.get("PASS_TO_PASS") or row.get("PASS_TO_PASS"), "PASS_TO_PASS", iid
        )
    except Exception:
        return None

    if not fail_to_pass:
        return None
    if pass_to_pass is None:
        pass_to_pass = []

    image_name = row.get("image_name")
    if not isinstance(image_name, str) or not image_name.strip():
        image_name = swebench_image_name(iid)

    return InstanceMeta(
        instance_id=iid,
        image_name=str(image_name),
        base_commit=base_commit,
        test_patch=test_patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
    )


def _load_dataset_meta_map(*, dataset_name: str, split: str) -> Dict[str, Dict[str, Any]]:
    """Load SWE-bench dataset rows keyed by instance_id.

    Only used for EVAL_STYLE="harness".
    """

    from swebench.harness.constants import KEY_INSTANCE_ID  # pyright: ignore[reportMissingImports]
    from swebench.harness.utils import load_swebench_dataset  # pyright: ignore[reportMissingImports]

    dataset = load_swebench_dataset(dataset_name, split)
    meta: Dict[str, Dict[str, Any]] = {}
    for row in dataset:
        iid = row.get(KEY_INSTANCE_ID)
        if isinstance(iid, str) and iid:
            meta[iid] = row
    return meta


def _try_get_test_cmd_and_directives_from_dataset_row(
    row: Optional[Dict[str, Any]], instance_id: str
) -> Tuple[str, List[str]]:
    """Best-effort derive SWE-bench test_cmd and directives from a dataset row."""

    if not row:
        return "", []
    try:
        from swebench.harness.constants import (  # pyright: ignore[reportMissingImports]
            MAP_REPO_VERSION_TO_SPECS,
        )
    except Exception:
        return "", []

    repo = str(row.get("repo") or "")
    version = str(row.get("version") or "")
    try:
        test_cmd = str(MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"])
    except Exception:
        test_cmd = ""

    directives: list[str] = []
    for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        raw = row.get(field, [])
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = []
        if isinstance(raw, list):
            directives.extend(str(x) for x in raw)

    directives = [str(x) for x in directives if str(x).strip()]
    return test_cmd, directives


def _extract_full_test_cmd_from_eval_script(
    row: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Extract the full test command from the SWE-bench test_spec eval_script.

    The official SWE-bench harness uses file-level test directives (e.g.,
    ``pytest -rA lib/matplotlib/tests/test_axes.py``) rather than individual
    test IDs.  Some test IDs in the dataset contain truncated parametrized
    names (``test_foo[param`` with no closing ``]``), which cause pytest
    collection errors when passed directly.  Extracting the command from the
    eval_script avoids this problem entirely.
    """
    if not row:
        return None
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec  # pyright: ignore[reportMissingImports]
    except Exception:
        return None

    try:
        ts = make_test_spec(row)
        script = ts.eval_script
    except Exception:
        return None

    # The eval_script wraps the test command between markers:
    #   : '>>>>> Start Test Output'
    #   <test command>
    #   : '>>>>> End Test Output'
    lines = script.split("\n")
    in_test_block = False
    test_cmds: list[str] = []
    for line in lines:
        if ">>>>> Start Test Output" in line:
            in_test_block = True
            continue
        if ">>>>> End Test Output" in line:
            break
        if in_test_block:
            stripped = line.strip()
            if stripped:
                test_cmds.append(stripped)

    if test_cmds:
        return " && ".join(test_cmds) if len(test_cmds) > 1 else test_cmds[0]
    return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _read_instances_rows(path: Path) -> List[Dict[str, Any]]:
    """Read instances rows from either JSONL (dict per line) or JSON (list of dicts)."""

    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    # Best-effort fallback
    return _read_jsonl(path)


def _read_instance_ids_from_instances_jsonl(path: Path) -> List[str]:
    """Read instance_ids from an instances* file, preserving first-seen order."""
    rows = _read_instances_rows(path)
    out: List[str] = []
    seen: set = set()
    for r in rows:
        iid = r.get("instance_id")
        if not isinstance(iid, str) or not iid.strip():
            continue
        if iid in seen:
            continue
        seen.add(iid)
        out.append(iid)
    return out


def _choose_instances_list_file(*, run_dirs: List[Path], mode: str) -> Optional[Path]:
    """Choose the instances* file that defines the expected evaluation set."""

    mode = str(mode or "").strip().lower()
    if mode not in {"selected", "all"}:
        mode = "selected"

    if mode == "selected":
        for rd in run_dirs:
            if not rd.exists():
                continue
            for name in ("instances_selected.jsonl", "instances_selected.json"):
                p = rd / name
                if p.exists():
                    return p

    for rd in run_dirs:
        if not rd.exists():
            continue
        for name in ("instances.jsonl", "instances.json"):
            p = rd / name
            if p.exists():
                return p
    return None


def _index_instances_rows(instances_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    """Map instance_id -> raw row (first seen) from instances.{jsonl,json}."""
    rows = _read_instances_rows(instances_jsonl)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        iid = r.get("instance_id")
        if not isinstance(iid, str) or not iid.strip():
            continue
        if iid in out:
            continue
        out[iid] = r
    return out


def _missing_meta_note_from_row(row: Optional[Dict[str, Any]]) -> str:
    """Human-readable note explaining which required fields are missing."""
    if not row:
        return "Missing instances row (no metadata found for this instance_id)."

    missing: List[str] = []
    image_name = row.get("image_name")
    base_commit = row.get("base_commit")
    extra = row.get("extra_fields") if isinstance(row.get("extra_fields"), dict) else {}
    if not isinstance(extra, dict):
        extra = {}

    test_patch = extra.get("test_patch")
    if not isinstance(test_patch, str) or not test_patch.strip():
        test_patch = row.get("test_patch")

    if not isinstance(image_name, str) or not image_name.strip():
        missing.append("image_name")
    if not isinstance(base_commit, str) or not base_commit.strip():
        missing.append("base_commit")
    if not isinstance(test_patch, str) or not test_patch.strip():
        missing.append("test_patch")

    # These can be lists or JSON-encoded list strings.
    try:
        f2p_raw = extra.get("FAIL_TO_PASS")
        if f2p_raw is None:
            f2p_raw = row.get("FAIL_TO_PASS")
        f2p = _parse_json_list_field(f2p_raw, "FAIL_TO_PASS", str(row.get("instance_id") or ""))
    except Exception:
        f2p = []
        missing.append("FAIL_TO_PASS (parse error)")
    try:
        p2p_raw = extra.get("PASS_TO_PASS")
        if p2p_raw is None:
            p2p_raw = row.get("PASS_TO_PASS")
        p2p = _parse_json_list_field(p2p_raw, "PASS_TO_PASS", str(row.get("instance_id") or ""))
    except Exception:
        p2p = []
        missing.append("PASS_TO_PASS (parse error)")

    if not f2p:
        missing.append("FAIL_TO_PASS")
    if not p2p:
        missing.append("PASS_TO_PASS")

    missing_unique = []
    seen = set()
    for m in missing:
        if m in seen:
            continue
        seen.add(m)
        missing_unique.append(m)
    if not missing_unique:
        return "Missing required metadata (unknown reason)."
    return "Missing required instances metadata: " + ", ".join(missing_unique)


def _load_preds(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected preds.json to be a dict keyed by instance_id: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = v
    return out


def _parse_json_list_field(raw: Any, field_name: str, instance_id: str) -> List[str]:
    """SWE-bench JSON often stores lists as JSON-encoded strings."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except Exception as e:
            raise ValueError(
                f"{instance_id}: could not parse {field_name} JSON string"
            ) from e
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        raise ValueError(f"{instance_id}: {field_name} parsed to non-list")
    raise ValueError(f"{instance_id}: {field_name} has unsupported type {type(raw)}")


def _load_instances_meta(instances_jsonl: Path) -> Dict[str, InstanceMeta]:
    rows = _read_instances_rows(instances_jsonl)
    meta: Dict[str, InstanceMeta] = {}

    for r in rows:
        iid = r.get("instance_id")
        if not isinstance(iid, str) or not iid.strip():
            continue
        image_name = r.get("image_name")
        base_commit = r.get("base_commit")
        extra = r.get("extra_fields") if isinstance(r.get("extra_fields"), dict) else {}
        if not isinstance(extra, dict):
            extra = {}

        test_patch = extra.get("test_patch")
        if not isinstance(test_patch, str) or not test_patch.strip():
            test_patch = r.get("test_patch")
        if not isinstance(test_patch, str):
            test_patch = ""

        f2p_raw = extra.get("FAIL_TO_PASS")
        if f2p_raw is None:
            f2p_raw = r.get("FAIL_TO_PASS")
        p2p_raw = extra.get("PASS_TO_PASS")
        if p2p_raw is None:
            p2p_raw = r.get("PASS_TO_PASS")

        fail_to_pass = _parse_json_list_field(f2p_raw, "FAIL_TO_PASS", iid)
        pass_to_pass = _parse_json_list_field(p2p_raw, "PASS_TO_PASS", iid)

        # Be tolerant: some instances.jsonl files may be missing fields for some rows.
        # We skip those rows rather than aborting evaluation for the entire target.
        if not isinstance(image_name, str) or not image_name:
            print(f"[WARN] {iid}: missing image_name in instances; skipping")
            continue
        if not isinstance(base_commit, str) or not base_commit:
            print(f"[WARN] {iid}: missing base_commit in instances; skipping")
            continue
        if not test_patch.strip():
            print(f"[WARN] {iid}: missing test_patch; skipping")
            continue
        if not fail_to_pass:
            print(f"[WARN] {iid}: missing FAIL_TO_PASS; skipping")
            continue
        if pass_to_pass is None:
            pass_to_pass = []

        meta[iid] = InstanceMeta(
            instance_id=iid,
            image_name=image_name,
            base_commit=base_commit,
            test_patch=test_patch,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )

    return meta


def _docker_run_eval(
    *,
    meta: InstanceMeta,
    model_patch: str,
    keep_logs: bool,
    work_dir: Path,
    timeout: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Run evaluation for one instance. Returns (resolved, details)."""

    # We mount temp files to avoid quoting issues with huge diffs.
    inst_dir = work_dir / meta.instance_id
    inst_dir.mkdir(parents=True, exist_ok=True)

    patch_file = inst_dir / "model_patch.diff"
    test_patch_file = inst_dir / "test_patch.diff"

    patch_text = model_patch or ""
    if patch_text and not patch_text.endswith("\n"):
        patch_text += "\n"
    patch_file.write_text(patch_text, encoding="utf-8")

    test_text = meta.test_patch or ""
    if test_text and not test_text.endswith("\n"):
        test_text += "\n"
    test_patch_file.write_text(test_text, encoding="utf-8")

    # Docker bind mounts require absolute host paths.
    patch_file_abs = patch_file.resolve()
    test_patch_file_abs = test_patch_file.resolve()

    fail_args = " ".join(shlex.quote(x) for x in meta.fail_to_pass)
    pass_args = " ".join(shlex.quote(x) for x in meta.pass_to_pass)

    # When PASS_TO_PASS is empty, skip the p2p pytest run entirely and treat
    # it as passed (exit 0).  Running `pytest -q` with no args would execute
    # the entire test suite, which is not what we want.
    if pass_args.strip():
        p2p_cmd = f"pytest -q {pass_args}"
    else:
        p2p_cmd = "echo 'PASS_TO_PASS is empty, skipping'"

    # A single bash -lc to keep conda activation consistent with SWE-bench images.
    bash_script = "\n".join(
        [
            "set +e",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "cd /testbed",
            "git config --global --add safe.directory /testbed",
            "git reset --hard >/dev/null",
            f"git reset --hard {shlex.quote(meta.base_commit)} >/dev/null",
            "git apply --recount /tmp/test_patch.diff",
            "if [ ! -s /tmp/model_patch.diff ]; then echo 'EMPTY_PATCH=1'; ec_model=99; else git apply -v --recount /tmp/model_patch.diff 2>/dev/null; ec_model=$?; fi",
            "if [ $ec_model -ne 0 ]; then patch --batch --fuzz=5 -p1 -i /tmp/model_patch.diff; ec_model=$?; fi",
            "if [ $ec_model -ne 0 ]; then echo MODEL_PATCH_APPLY_FAILED=$ec_model; FAIL_TO_PASS_exit=99; PASS_TO_PASS_exit=99; RESOLVED=0; exit 0; fi",
            "pytest -q " + fail_args,
            "ec1=$?",
            p2p_cmd,
            "ec2=$?",
            "echo FAIL_TO_PASS_exit=$ec1",
            "echo PASS_TO_PASS_exit=$ec2",
            "if [ $ec1 -eq 0 ] && [ $ec2 -eq 0 ]; then echo RESOLVED=1; else echo RESOLVED=0; fi",
            "exit 0",
        ]
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        "PYTHONIOENCODING=utf-8",
        "-v",
        f"{patch_file_abs}:/tmp/model_patch.diff:ro",
        "-v",
        f"{test_patch_file_abs}:/tmp/test_patch.diff:ro",
        meta.image_name,
        "bash",
        "-lc",
        bash_script,
    ]

    docker_log_path = inst_dir / "docker_output.log"
    test_output_path = inst_dir / "test_output.txt"
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(timeout) if timeout else None,
            check=False,
        )
        output = proc.stdout or ""
        timed_out = False
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or "") if isinstance(e.stdout, str) else ""
        proc = subprocess.CompletedProcess(cmd, returncode=124, stdout=output)
        timed_out = True

    resolved: Optional[bool] = None
    fail_exit: Optional[int] = None
    pass_exit: Optional[int] = None
    model_apply_failed: Optional[int] = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("MODEL_PATCH_APPLY_FAILED="):
            try:
                model_apply_failed = int(line.split("=", 1)[1])
            except Exception:
                model_apply_failed = 1
        if line.startswith("FAIL_TO_PASS_exit="):
            try:
                fail_exit = int(line.split("=", 1)[1])
            except Exception:
                pass
        if line.startswith("PASS_TO_PASS_exit="):
            try:
                pass_exit = int(line.split("=", 1)[1])
            except Exception:
                pass
        if line.startswith("RESOLVED="):
            resolved = line.split("=", 1)[1].strip() == "1"

    fail_to_pass_passed = (fail_exit == 0)
    pass_to_pass_passed = (pass_exit == 0)

    if resolved is None:
        # Conservative fallback: require both exits present and == 0.
        resolved = bool(fail_to_pass_passed and pass_to_pass_passed)

    details: Dict[str, Any] = {
        "instance_id": meta.instance_id,
        "image_name": meta.image_name,
        "base_commit": meta.base_commit,
        "resolved": resolved,
        "timed_out": timed_out,
        # Useful secondary metrics (we still compute resolved from both).
        "fail_to_pass_passed": fail_to_pass_passed,
        "pass_to_pass_passed": pass_to_pass_passed,
        "fail_to_pass_exit": fail_exit,
        "pass_to_pass_exit": pass_exit,
        "model_patch_apply_failed": model_apply_failed,
        "docker_returncode": proc.returncode,
    }

    if keep_logs:
        docker_log_path.write_text(output, encoding="utf-8")
        test_output_path.write_text(output, encoding="utf-8")
        (inst_dir / "report.json").write_text(json.dumps(details, indent=2, sort_keys=True), encoding="utf-8")
    return resolved, details


def _fallback_parse_pytest_failures(test_output: str) -> Optional[set]:
    """Parse FAILED and ERROR test/file names from pytest short test summary.

    When the SWE-bench log parser returns an incomplete status map (e.g. for
    tox-wrapped pytest output that lacks per-test PASSED lines), this function
    extracts the set of FAILED and ERROR entries from the ``short test summary
    info`` section that pytest always emits.

    ERROR entries indicate collection/import failures (e.g. SyntaxError in the
    patched code) and use file-level paths rather than ``file::test`` IDs.
    Callers should use :func:`_test_in_failure_set` for matching, which handles
    both exact and prefix matching.

    Returns a set of failed/errored identifiers, or None if the summary section
    cannot be found (meaning we should not override the harness result).
    """
    # Strip ANSI escape codes for reliable matching.
    clean = re.sub(r"\x1b\[[0-9;]*m", "", test_output)
    clean = re.sub(r"\[([0-9]+)m", "", clean)

    # Locate the "short test summary info" block.
    summary_start = clean.find("short test summary info")
    if summary_start == -1:
        return None

    summary_section = clean[summary_start:]
    failed_tests: set = set()
    for line in summary_section.split("\n"):
        line = line.strip()
        # pytest summary lines look like:
        #   "FAILED tests/foo.py::test_bar - ..."
        #   "ERROR tests/foo.py - ..."  (collection/import errors)
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            parts = line.replace(" - ", " ").split()
            if len(parts) >= 2:
                failed_tests.add(parts[1])
    return failed_tests


def _test_in_failure_set(test_id: str, failed_set: set) -> bool:
    """Check if *test_id* matches any entry in *failed_set*.

    Supports both exact matches (for FAILED lines that carry full
    ``file::class::method`` test IDs) and prefix matches (for ERROR lines
    that carry file-level paths like ``tests/foo.py``).  A file-level ERROR
    means every test in that file failed to collect.
    """
    if test_id in failed_set:
        return True
    for entry in failed_set:
        if "::" not in entry and test_id.startswith(entry + "::"):
            return True
    return False


def _docker_run_eval_harness_style(
    *,
    meta: InstanceMeta,
    model_patch: str,
    keep_logs: bool,
    work_dir: Path,
    timeout: int,
    dataset_row: Optional[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    """Run SWE-bench harness-style evaluation in a prebuilt instance image.

    This is closer to SWE-bench official scoring than exit-code-based pytest runs.
    If the harness isn't available or test_cmd can't be derived, callers should
    fall back to `_docker_run_eval`.
    """

    # Prefer the full test command from the official SWE-bench eval_script.
    # This uses file-level directives (e.g., "pytest -rA lib/.../test_foo.py")
    # instead of individual test IDs, avoiding issues with truncated parametrized
    # test names in the dataset (e.g., "test_bar[param" with no closing "]").
    full_test_cmd = _extract_full_test_cmd_from_eval_script(dataset_row)
    if not full_test_cmd:
        # Fallback to manual construction if eval_script extraction fails.
        test_cmd, test_directives = _try_get_test_cmd_and_directives_from_dataset_row(
            dataset_row, meta.instance_id
        )
        directives_args = " ".join(shlex.quote(x) for x in (test_directives or []))
        full_test_cmd = f"{test_cmd} {directives_args}".strip() if test_cmd else ""

    inst_dir = work_dir / meta.instance_id
    inst_dir.mkdir(parents=True, exist_ok=True)

    patch_file = inst_dir / "model_patch.diff"
    test_patch_file = inst_dir / "test_patch.diff"

    patch_text = model_patch or ""
    if patch_text and not patch_text.endswith("\n"):
        patch_text += "\n"
    patch_file.write_text(patch_text, encoding="utf-8")

    test_text = meta.test_patch or ""
    if test_text and not test_text.endswith("\n"):
        test_text += "\n"
    test_patch_file.write_text(test_text, encoding="utf-8")

    bash_script = "\n".join(
        [
            "set +e",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "cd /testbed",
            "git config --global --add safe.directory /testbed",
            "git reset --hard >/dev/null",
            f"git reset --hard {shlex.quote(meta.base_commit)} >/dev/null",
            "git apply --recount /tmp/test_patch.diff",
            "if [ ! -s /tmp/model_patch.diff ]; then echo '>>>>> Patch Apply Failed'; echo 'EMPTY_PATCH=1'; exit 0; fi",
            "git apply -v --recount /tmp/model_patch.diff 2>/dev/null; ec_model=$?",
            "if [ $ec_model -ne 0 ]; then patch --batch --fuzz=5 -p1 -i /tmp/model_patch.diff; ec_model=$?; fi",
            "if [ $ec_model -ne 0 ]; then echo '>>>>> Patch Apply Failed'; echo MODEL_PATCH_APPLY_FAILED=$ec_model; exit 0; fi",
            "echo '>>>>> Applied Patch (pred)'",
            "echo '>>>>> Start Test Output'",
            full_test_cmd if full_test_cmd else "echo 'NO_TEST_CMD=1'",
            "ec=$?",
            "echo '>>>>> End Test Output'",
            "echo TEST_CMD_exit=$ec",
            "exit 0",
        ]
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        "PYTHONIOENCODING=utf-8",
        "-v",
        f"{patch_file.absolute()}:/tmp/model_patch.diff:ro",
        "-v",
        f"{test_patch_file.absolute()}:/tmp/test_patch.diff:ro",
        meta.image_name,
        "bash",
        "-lc",
        bash_script,
    ]

    docker_log_path = inst_dir / "docker_output.log"
    test_output_path = inst_dir / "test_output.txt"

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(timeout) if timeout else None,
            check=False,
        )
        output = proc.stdout or ""
        timed_out = False
        docker_returncode: Optional[int] = proc.returncode
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or "") if isinstance(e.stdout, str) else ""
        timed_out = True
        docker_returncode = None

    # We need a log file for harness parsing; write it regardless of keep_logs,
    # but optionally delete it afterwards.
    test_output_path.write_text(output if not timed_out else ">>>>> Tests Timed Out\n", encoding="utf-8")
    if keep_logs:
        docker_log_path.write_text(output, encoding="utf-8")

    model_apply_failed: Optional[int] = None
    test_cmd_exit: Optional[int] = None
    for line in output.splitlines():
        s = line.strip()
        if s.startswith("MODEL_PATCH_APPLY_FAILED="):
            try:
                model_apply_failed = int(s.split("=", 1)[1])
            except Exception:
                model_apply_failed = 1
        if s.startswith("TEST_CMD_exit="):
            try:
                test_cmd_exit = int(s.split("=", 1)[1])
            except Exception:
                test_cmd_exit = 1

    resolved = False
    fail_to_pass_passed = False
    pass_to_pass_passed = False
    harness_error: Optional[str] = None
    try:
        from swebench.harness.constants import (  # pyright: ignore[reportMissingImports]
            FAIL_TO_PASS as F2P,
            PASS_TO_PASS as P2P,
            KEY_INSTANCE_ID,
        )
        from swebench.harness.grading import get_eval_report  # pyright: ignore[reportMissingImports]
        from swebench.harness.test_spec.test_spec import make_test_spec  # pyright: ignore[reportMissingImports]

        if not dataset_row:
            raise ValueError("dataset_row is required for harness-style grading")
        test_spec = make_test_spec(dataset_row)
        pred = {
            KEY_INSTANCE_ID: meta.instance_id,
            "model_patch": model_patch,
            "model_name_or_path": "swe-agent",
        }
        report_map = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=str(test_output_path),
            include_tests_status=True,
        )
        inst_report = report_map.get(meta.instance_id, {})
        resolved = bool(inst_report.get("resolved"))
        tests_status = inst_report.get("tests_status") or {}
        f2p_failures = (
            tests_status.get(F2P, {}).get("failure", []) if isinstance(tests_status, dict) else []
        )
        p2p_failures = (
            tests_status.get(P2P, {}).get("failure", []) if isinstance(tests_status, dict) else []
        )
        fail_to_pass_passed = (len(f2p_failures) == 0)
        pass_to_pass_passed = (len(p2p_failures) == 0)
    except Exception as e:
        harness_error = f"{type(e).__name__}: {e}"

    # Fallback: when the harness log parser cannot determine individual test
    # statuses (e.g. tox-wrapped pytest output lacks per-test PASSED lines),
    # we use two strategies:
    #  1. test_cmd_exit == 0  →  ALL tests passed, so f2p/p2p tests passed too.
    #  2. test_cmd_exit != 0  →  parse pytest "short test summary info" for
    #     explicit FAILED and ERROR lines; any f2p/p2p test NOT listed there
    #     is inferred as PASSED.  ERROR lines (collection/import failures) use
    #     file-level paths and are matched via prefix against test IDs.
    fallback_used = False
    if not resolved and harness_error is None and not timed_out and model_apply_failed is None:
        if test_cmd_exit == 0:
            # All tests passed — no failures at all.
            resolved = True
            fail_to_pass_passed = True
            pass_to_pass_passed = True
            fallback_used = True
            print(
                f"[fallback] {meta.instance_id}: harness parser returned incomplete status map "
                f"but test_cmd exited 0 (all tests passed) -> resolved=True"
            )
        else:
            failed_set = _fallback_parse_pytest_failures(output)
            if failed_set is not None:
                fb_f2p_ok = all(not _test_in_failure_set(t, failed_set) for t in meta.fail_to_pass)
                fb_p2p_ok = all(not _test_in_failure_set(t, failed_set) for t in meta.pass_to_pass)
                if fb_f2p_ok and fb_p2p_ok:
                    resolved = True
                    fail_to_pass_passed = True
                    pass_to_pass_passed = True
                    fallback_used = True
                    print(
                        f"[fallback] {meta.instance_id}: harness parser missed passing tests; "
                        f"pytest summary confirms no f2p/p2p failures -> resolved=True"
                    )

    details: Dict[str, Any] = {
        "instance_id": meta.instance_id,
        "image_name": meta.image_name,
        "base_commit": meta.base_commit,
        "resolved": bool(resolved),
        "timed_out": timed_out,
        "fail_to_pass_passed": bool(fail_to_pass_passed),
        "pass_to_pass_passed": bool(pass_to_pass_passed),
        "model_patch_apply_failed": model_apply_failed,
        "docker_returncode": docker_returncode,
        "eval_style": "harness",
        "fallback_override": fallback_used,
        "test_cmd": full_test_cmd,
        "test_cmd_exit": test_cmd_exit,
        "harness_error": harness_error,
    }

    # Keep a small structured artifact always; optionally keep logs too.
    (inst_dir / "report.json").write_text(json.dumps(details, indent=2, sort_keys=True), encoding="utf-8")
    if not keep_logs:
        try:
            test_output_path.unlink(missing_ok=True)
        except Exception:
            pass

    return bool(resolved), details


def _read_text_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(errors="ignore")


def _collect_patch_candidates(
    *,
    run_dir: Path,
    instance_id: str,
    preds: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str]]:
    """Return ordered (label, patch_text) candidates.

    Candidate #1 is the patch from preds.json (if present).
    Additional candidates are any *.patch files under repair/<instance_id>/.
    Dedupes identical patch texts.
    """

    candidates: List[Tuple[str, str]] = []
    seen: set = set()

    pred_patch = preds.get(instance_id, {}).get("model_patch")
    if isinstance(pred_patch, str) and pred_patch.strip():
        norm = pred_patch.strip()
        seen.add(norm)
        candidates.append(("preds.json", pred_patch))

    inst_dir = run_dir / "repair" / instance_id
    if inst_dir.is_dir():
        patch_files = sorted(p for p in inst_dir.glob("*.patch") if p.is_file())
        for pf in patch_files:
            txt = _read_text_safely(pf)
            if not isinstance(txt, str) or not txt.strip():
                continue
            norm = txt.strip()
            if norm in seen:
                continue
            seen.add(norm)
            candidates.append((pf.name, txt))

    # If there was no preds.json patch but there is a canonical <iid>.patch, treat
    # the first discovered candidate as patch@1.
    return candidates


def _docker_image_exists(image: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _docker_pull(image: str) -> bool:
    proc = subprocess.run(
        ["docker", "pull", image],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
        text=True,
    )
    return proc.returncode == 0


def _docker_remove_image(image: str) -> bool:
    stop_and_remove_containers_for_image(image)
    proc = subprocess.run(
        ["docker", "image", "rm", "-f", image],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
        text=True,
    )
    prune()
    return proc.returncode == 0


def _have_patches_in_run_dir(run_dir: Path) -> set:
    """Return instance_ids that have any non-empty patch candidate in a run dir."""
    have: set = set()
    repair_dir = run_dir / "repair"
    preds_json = repair_dir / "preds.json"
    if preds_json.exists():
        try:
            data = json.loads(preds_json.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for iid, obj in data.items():
                    if not isinstance(iid, str) or not iid:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    patch = obj.get("model_patch")
                    if isinstance(patch, str) and patch.strip():
                        have.add(iid)
        except Exception:
            pass

    if repair_dir.is_dir():
        for inst_dir in repair_dir.iterdir():
            if not inst_dir.is_dir():
                continue
            iid = inst_dir.name
            try:
                for pf in inst_dir.glob("*.patch"):
                    if not pf.is_file():
                        continue
                    txt = _read_text_safely(pf)
                    if isinstance(txt, str) and txt.strip():
                        have.add(iid)
                        break
            except Exception:
                continue
    return have


def _collect_candidates_across_run_dirs(
    *, run_dirs: List[Path], instance_id: str
) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Return (candidates, chosen_source) merged across run_dir priority order."""
    merged: List[Tuple[str, str]] = []
    seen: set = set()
    chosen_source: Optional[str] = None

    for rd in run_dirs:
        if not rd.exists():
            continue
        preds: Dict[str, Dict[str, Any]] = {}
        preds_path = rd / "repair" / "preds.json"
        if preds_path.exists():
            try:
                preds = _load_preds(preds_path)
            except Exception:
                preds = {}

        cands = _collect_patch_candidates(run_dir=rd, instance_id=instance_id, preds=preds)
        if cands and chosen_source is None:
            chosen_source = str(rd)

        for lbl, patch in cands:
            norm = (patch or "").strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            merged.append((f"{rd.name}/{lbl}", patch))

    return merged, chosen_source


def _format_variant_eval_summary(
    *,
    st: Dict[str, Any],
    expected_instance_ids: List[str],
    generated_at: str,
) -> Dict[str, Any]:
    evaluated = int(st.get("evaluated") or 0)
    total_resolved_at_1 = int(st.get("resolved_at_1") or 0)
    total_pass_to_pass_ok_at_1 = int(st.get("pass_to_pass_ok_at_1") or 0)
    total_fail_to_pass_ok_at_1 = int(st.get("fail_to_pass_ok_at_1") or 0)

    overall_pass_at_1 = (total_resolved_at_1 / evaluated) if evaluated else None
    overall_regression_only = (total_pass_to_pass_ok_at_1 / evaluated) if evaluated else None
    overall_fail_to_pass = (total_fail_to_pass_ok_at_1 / evaluated) if evaluated else None

    label = str(st.get("label") or "variant")
    run_dir = str(st.get("run_dir") or "")
    source_run_dirs = list(st.get("source_run_dirs") or [])

    per_target_entry: Dict[str, Any] = {
        "label": label,
        "run_dir": run_dir,
        "source_run_dirs": source_run_dirs,
        "preds_json": None,
        "instances_jsonl": None,
        "evaluated": evaluated,
        "pass_to_pass_ok_at_1": total_pass_to_pass_ok_at_1,
        "fail_to_pass_ok_at_1": total_fail_to_pass_ok_at_1,
        "regression_only_pass_rate": overall_regression_only,
        "fail_to_pass_pass_rate": overall_fail_to_pass,
        "resolved_at_1": total_resolved_at_1,
        "pass_at_1": overall_pass_at_1,
        "per_instance": st.get("per_instance", {}),
        "missing_repairs": sorted(st.get("missing_repairs", [])),
        "expected_total": len(expected_instance_ids),
        "progress": st.get("progress", {}),
    }

    # Best-effort fill in preds_json/instances_jsonl from first existing source.
    try:
        for rd in source_run_dirs:
            p = Path(rd) / "repair" / "preds.json"
            if p.exists():
                per_target_entry["preds_json"] = str(p)
                break
        for rd in source_run_dirs:
            for name in ("instances.jsonl", "instances.json"):
                p = Path(rd) / name
                if p.exists():
                    per_target_entry["instances_jsonl"] = str(p)
                    raise StopIteration
    except Exception:
        pass

    return {
        "generated_at": generated_at,
        "targets": source_run_dirs,
        "total_evaluated": evaluated,
        "total_pass_to_pass_ok_at_1": total_pass_to_pass_ok_at_1,
        "total_fail_to_pass_ok_at_1": total_fail_to_pass_ok_at_1,
        "overall_regression_only_pass_rate": overall_regression_only,
        "overall_fail_to_pass_pass_rate": overall_fail_to_pass,
        "total_resolved_at_1": total_resolved_at_1,
        "overall_pass_at_1": overall_pass_at_1,
        "missing_repairs": sorted(st.get("missing_repairs", [])),
        "progress": st.get("progress", {}),
        "per_target": [per_target_entry],
    }


def _label_for_target(target: str) -> str:
    p = Path(target)
    name = p.name
    if name in {"preds.json", "repair"}:
        name = p.parent.name
    if name == "eval_swebench_summary.json":
        name = p.parent.parent.name
    return name or str(p)


def _resolve_target_to_run_dir(target: str) -> Path:
    """Resolve various input styles to results/<RUN>/<VARIANT>."""
    p = Path(os.path.expanduser(target)).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Target does not exist: {p}")

    if p.is_file():
        if p.name == "preds.json":
            # .../<VARIANT>/repair/preds.json
            return p.parent.parent
        if p.name == "eval_swebench_summary.json":
            # .../<VARIANT>/repair/eval_swebench_summary.json
            return p.parent.parent
        if p.suffix == ".patch" and p.parent.is_dir() and p.parent.parent.is_dir():
            # .../<VARIANT>/repair/<instance_id>/<something>.patch
            repair_dir = p.parent.parent
            if repair_dir.name == "repair" and any(
                (repair_dir.parent / n).exists() for n in ("instances.jsonl", "instances.json")
            ):
                return repair_dir.parent
        raise ValueError(f"Unsupported file target (expected preds.json or *.patch): {p}")

    # Directory cases
    if any((p / n).exists() for n in ("instances.jsonl", "instances.json")) and (p / "repair").is_dir():
        return p
    if p.name == "repair" and any((p.parent / n).exists() for n in ("instances.jsonl", "instances.json")):
        return p.parent
    if p.name == "eval_work" and any(
        (p.parent.parent / n).exists() for n in ("instances.jsonl", "instances.json")
    ):
        return p.parent.parent

    raise ValueError(
        "Could not resolve target to a run directory containing instances.(jsonl|json) and repair/: "
        + str(p)
    )


def _evaluate_run_dir(run_dir: Path, *, keep_logs: bool, limit: Optional[int], timeout: int) -> Dict[str, Any]:
    instances_jsonl = run_dir / "instances.jsonl"
    instances_json = run_dir / "instances.json"
    instances_selected_jsonl = run_dir / "instances_selected.jsonl"
    instances_selected_json = run_dir / "instances_selected.json"
    preds_json = run_dir / "repair" / "preds.json"

    instances_path = instances_jsonl if instances_jsonl.exists() else instances_json
    if not instances_path.exists():
        raise FileNotFoundError(f"Missing instances file: {instances_jsonl} (or {instances_json})")
    # preds.json is optional; if missing we fall back to per-instance *.patch files.

    meta_by_id = _load_instances_meta(instances_path)
    raw_rows_by_id = _index_instances_rows(instances_path)
    preds: Dict[str, Dict[str, Any]] = {}
    if preds_json.exists():
        preds = _load_preds(preds_json)

    # Define the *expected* instances from instances_selected.jsonl (default)
    # or from instances.jsonl (CONFIG["INSTANCE_LIST"] = "all").
    instances_mode = str(CONFIG.get("INSTANCE_LIST") or "selected")
    expected_file = _choose_instances_list_file(run_dirs=[run_dir], mode=instances_mode) or instances_path
    instance_ids = _read_instance_ids_from_instances_jsonl(expected_file)
    if not instance_ids:
        # Fallback to evaluating what we actually have artifacts for.
        repair_root = run_dir / "repair"
        repair_dirs = sorted([p.name for p in repair_root.iterdir() if p.is_dir()]) if repair_root.is_dir() else []
        if preds:
            instance_ids = sorted(set(repair_dirs) | set(preds.keys()))
        else:
            instance_ids = sorted(set(repair_dirs))

    if limit is not None:
        instance_ids = instance_ids[: int(limit)]

    work_dir = run_dir / "repair" / "eval_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    eval_style = str(CONFIG.get("EVAL_STYLE") or "pytest").strip().lower()
    meta_source = str(CONFIG.get("META_SOURCE") or "instances_jsonl").strip().lower()
    if meta_source not in {"instances_jsonl", "dataset", "auto"}:
        meta_source = "instances_jsonl"
    dataset_meta: Optional[Dict[str, Dict[str, Any]]] = None
    dataset_error: Optional[str] = None
    if eval_style == "harness" or meta_source in {"dataset", "auto"}:
        try:
            dataset_meta = _load_dataset_meta_map(
                dataset_name=str(CONFIG.get("DATASET_NAME") or ""),
                split=str(CONFIG.get("SPLIT") or ""),
            )
        except Exception as e:
            dataset_meta = None
            dataset_error = f"{type(e).__name__}: {e}"

    per_instance: Dict[str, Any] = {}
    missing_repairs: List[str] = []
    resolved_at_1_count = 0
    resolved_any_count = 0
    pass_to_pass_ok_at_1_count = 0
    fail_to_pass_ok_at_1_count = 0
    pass_to_pass_ok_any_count = 0
    fail_to_pass_ok_any_count = 0

    for iid in instance_ids:
        candidates = _collect_patch_candidates(run_dir=run_dir, instance_id=iid, preds=preds)

        # Resolve metadata from instances.jsonl and/or dataset.
        meta: Optional[InstanceMeta] = None
        if meta_source in {"instances_jsonl", "auto"}:
            meta = meta_by_id.get(iid)
        if meta is None and meta_source in {"dataset", "auto"} and dataset_meta is not None:
            meta = _instance_meta_from_dataset_row(dataset_meta.get(iid), iid)

        if meta is None:
            per_instance[iid] = {
                "instance_id": iid,
                "num_candidates": len(candidates),
                "resolved_at_1": False,
                "resolved_any": False,
                "candidate_results": [],
                "note": _missing_meta_note_from_row(raw_rows_by_id.get(iid))
                + (
                    "" if dataset_meta is None else " (dataset fallback attempted but unavailable/insufficient)"
                ),
            }
            continue

        m = meta

        if not candidates:
            missing_repairs.append(iid)
            per_instance[iid] = {
                "instance_id": iid,
                "resolved_at_1": False,
                "resolved_any": False,
                "num_candidates": 0,
                "candidates": [],
                "candidate_results": [],
                "note": "Missing repair: no non-empty patch candidates found (preds.json empty and no *.patch files).",
            }
            continue

        # Evaluate patch@1
        label1, patch1 = candidates[0]
        print(f"[eval] {run_dir.name} {iid} [1/{len(candidates)}] {label1} ...")
        if eval_style == "harness" and dataset_meta is not None:
            row = dataset_meta.get(iid)
            test_cmd, _ = _try_get_test_cmd_and_directives_from_dataset_row(row, iid)
            if test_cmd.strip():
                resolved1, details1 = _docker_run_eval_harness_style(
                    meta=m,
                    model_patch=patch1,
                    keep_logs=keep_logs,
                    work_dir=work_dir,
                    timeout=timeout,
                    dataset_row=row,
                )
            else:
                resolved1, details1 = _docker_run_eval(
                    meta=m,
                    model_patch=patch1,
                    keep_logs=keep_logs,
                    work_dir=work_dir,
                    timeout=timeout,
                )
                details1 = dict(
                    details1,
                    eval_style="pytest",
                    note=(
                        "EVAL_STYLE=harness requested but test_cmd unavailable for this instance_id; "
                        "fell back to pytest exit-code scoring."
                    ),
                )
        else:
            resolved1, details1 = _docker_run_eval(
                meta=m,
                model_patch=patch1,
                keep_logs=keep_logs,
                work_dir=work_dir,
                timeout=timeout,
            )
            if eval_style == "harness" and dataset_meta is None:
                details1 = dict(
                    details1,
                    eval_style="pytest",
                    note=(
                        "EVAL_STYLE=harness requested but dataset/harness unavailable; "
                        f"fell back to pytest. dataset_error={dataset_error}"
                    ),
                )

        resolved_at_1 = bool(resolved1)
        resolved_any = resolved_at_1
        winning_index: Optional[int] = 1 if resolved_at_1 else None
        winning_label: Optional[str] = label1 if resolved_at_1 else None

        pass_to_pass_any = bool(details1.get("pass_to_pass_passed"))
        fail_to_pass_any = bool(details1.get("fail_to_pass_passed"))
        pass_to_pass_any_winner: Optional[int] = 1 if pass_to_pass_any else None
        fail_to_pass_any_winner: Optional[int] = 1 if fail_to_pass_any else None

        candidate_details: List[Dict[str, Any]] = [dict(details1, candidate_label=label1, candidate_index=1)]

        evaluate_any = bool(CONFIG.get("EVALUATE_ANY_REPAIR", True))
        if evaluate_any and (not resolved_any) and len(candidates) > 1:
            for idx, (lbl, patch_txt) in enumerate(candidates[1:], start=2):
                print(f"[eval] {run_dir.name} {iid} [{idx}/{len(candidates)}] {lbl} ...")
                if eval_style == "harness" and dataset_meta is not None:
                    row = dataset_meta.get(iid)
                    test_cmd, _ = _try_get_test_cmd_and_directives_from_dataset_row(row, iid)
                    if test_cmd.strip():
                        res, det = _docker_run_eval_harness_style(
                            meta=m,
                            model_patch=patch_txt,
                            keep_logs=keep_logs,
                            work_dir=work_dir,
                            timeout=timeout,
                            dataset_row=row,
                        )
                    else:
                        res, det = _docker_run_eval(
                            meta=m,
                            model_patch=patch_txt,
                            keep_logs=keep_logs,
                            work_dir=work_dir,
                            timeout=timeout,
                        )
                        det = dict(
                            det,
                            eval_style="pytest",
                            note=(
                                "EVAL_STYLE=harness requested but test_cmd unavailable for this instance_id; "
                                "fell back to pytest exit-code scoring."
                            ),
                        )
                else:
                    res, det = _docker_run_eval(
                        meta=m,
                        model_patch=patch_txt,
                        keep_logs=keep_logs,
                        work_dir=work_dir,
                        timeout=timeout,
                    )
                    if eval_style == "harness" and dataset_meta is None:
                        det = dict(
                            det,
                            eval_style="pytest",
                            note=(
                                "EVAL_STYLE=harness requested but dataset/harness unavailable; "
                                f"fell back to pytest. dataset_error={dataset_error}"
                            ),
                        )
                det = dict(det, candidate_label=lbl, candidate_index=idx)
                candidate_details.append(det)

                if (not pass_to_pass_any) and det.get("pass_to_pass_passed"):
                    pass_to_pass_any = True
                    pass_to_pass_any_winner = idx
                if (not fail_to_pass_any) and det.get("fail_to_pass_passed"):
                    fail_to_pass_any = True
                    fail_to_pass_any_winner = idx

                if res:
                    resolved_any = True
                    winning_index = idx
                    winning_label = lbl
                    break

        per_instance[iid] = {
            "instance_id": iid,
            "num_candidates": len(candidates),
            "resolved_at_1": resolved_at_1,
            "resolved_any": resolved_any,
            "winning_candidate_index": winning_index,
            "winning_candidate_label": winning_label,
            "candidate_results": candidate_details,
            # For convenience: expose patch@1 exits.
            "fail_to_pass_passed_at_1": bool(details1.get("fail_to_pass_passed")),
            "pass_to_pass_passed_at_1": bool(details1.get("pass_to_pass_passed")),
            # Any-candidate diagnostics (may succeed on a different candidate than resolved).
            "pass_to_pass_passed_any": pass_to_pass_any,
            "fail_to_pass_passed_any": fail_to_pass_any,
            "pass_to_pass_any_first_index": pass_to_pass_any_winner,
            "fail_to_pass_any_first_index": fail_to_pass_any_winner,
        }

        resolved_at_1_count += 1 if resolved_at_1 else 0
        resolved_any_count += 1 if resolved_any else 0
        pass_to_pass_ok_at_1_count += 1 if details1.get("pass_to_pass_passed") else 0
        fail_to_pass_ok_at_1_count += 1 if details1.get("fail_to_pass_passed") else 0
        pass_to_pass_ok_any_count += 1 if pass_to_pass_any else 0
        fail_to_pass_ok_any_count += 1 if fail_to_pass_any else 0

    evaluated = len(instance_ids)
    pass1 = (resolved_at_1_count / evaluated) if evaluated else None
    pass_any = (resolved_any_count / evaluated) if evaluated else None
    regression_only_rate = (pass_to_pass_ok_at_1_count / evaluated) if evaluated else None
    fail_to_pass_rate = (fail_to_pass_ok_at_1_count / evaluated) if evaluated else None
    regression_only_any_rate = (pass_to_pass_ok_any_count / evaluated) if evaluated else None
    fail_to_pass_any_rate = (fail_to_pass_ok_any_count / evaluated) if evaluated else None

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "preds_json": str(preds_json) if preds_json.exists() else None,
        "instances_jsonl": str(instances_path),
        "instances_selected_jsonl": str(instances_selected_jsonl)
        if instances_selected_jsonl.exists()
        else (str(instances_selected_json) if instances_selected_json.exists() else None),
        "expected_instances_file": str(expected_file),
        "evaluated": evaluated,
        # SWE-bench primary metrics
        "resolved_at_1": resolved_at_1_count,
        "pass_at_1": pass1,
        "resolved_any": resolved_any_count,
        "pass_at_any": pass_any,
        # Secondary metrics
        "pass_to_pass_ok_at_1": pass_to_pass_ok_at_1_count,
        "fail_to_pass_ok_at_1": fail_to_pass_ok_at_1_count,
        "regression_only_pass_rate": regression_only_rate,
        "fail_to_pass_pass_rate": fail_to_pass_rate,
        "pass_to_pass_ok_any": pass_to_pass_ok_any_count,
        "fail_to_pass_ok_any": fail_to_pass_ok_any_count,
        "regression_only_pass_rate_any": regression_only_any_rate,
        "fail_to_pass_pass_rate_any": fail_to_pass_any_rate,
        "per_instance": per_instance,
        "missing_repairs": sorted(missing_repairs),
        "eval_style": eval_style,
        "dataset_name": str(CONFIG.get("DATASET_NAME") or "") if eval_style == "harness" else None,
        "split": str(CONFIG.get("SPLIT") or "") if eval_style == "harness" else None,
        "dataset_error": dataset_error,
    }

    out_path = run_dir / "repair" / "eval_swebench_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[eval] wrote: {out_path}")
    return summary


def _evaluate_variants_instance_first(
    *,
    variant_targets: List[VariantTarget],
    keep_logs: bool,
    limit: Optional[int],
    pull_images: bool,
    delete_images_after_eval: bool,
    timeout: int,
    progress_write_every: int,
    combined_out_path: Path,
) -> Dict[str, Any]:
    """Evaluate N variants per instance, reusing one pulled image per instance."""

    eval_style = str(CONFIG.get("EVAL_STYLE") or "pytest").strip().lower()
    meta_source = str(CONFIG.get("META_SOURCE") or "instances_jsonl").strip().lower()
    if meta_source not in {"instances_jsonl", "dataset", "auto"}:
        meta_source = "instances_jsonl"

    dataset_meta: Optional[Dict[str, Dict[str, Any]]] = None
    dataset_error: Optional[str] = None
    if eval_style == "harness" or meta_source in {"dataset", "auto"}:
        try:
            dataset_meta = _load_dataset_meta_map(
                dataset_name=str(CONFIG.get("DATASET_NAME") or ""),
                split=str(CONFIG.get("SPLIT") or ""),
            )
        except Exception as e:
            dataset_meta = None
            dataset_error = f"{type(e).__name__}: {e}"

    # Merge metadata from all available source run dirs.
    meta_by_id: Dict[str, InstanceMeta] = {}
    raw_rows_by_id: Dict[str, Dict[str, Any]] = {}
    for vt in variant_targets:
        for rd in vt.source_run_dirs:
            instances_jsonl = rd / "instances.jsonl"
            instances_json = rd / "instances.json"
            instances_path = instances_jsonl if instances_jsonl.exists() else instances_json
            if not instances_path.exists():
                continue
            # Keep raw rows for missing-meta diagnostics.
            try:
                for iid, row in _index_instances_rows(instances_path).items():
                    raw_rows_by_id.setdefault(iid, row)
            except Exception:
                pass
            try:
                m = _load_instances_meta(instances_path)
            except Exception:
                continue
            for iid, imeta in m.items():
                meta_by_id.setdefault(iid, imeta)

    have_by_variant: Dict[str, set] = {}
    for vt in variant_targets:
        have: set = set()
        for rd in vt.source_run_dirs:
            if rd.exists():
                have |= _have_patches_in_run_dir(rd)
        have_by_variant[vt.label] = have

    # Agentless-style expected instance set: prefer instances_selected.jsonl when available.
    all_source_dirs: List[Path] = []
    for vt in variant_targets:
        for rd in vt.source_run_dirs:
            if rd not in all_source_dirs:
                all_source_dirs.append(rd)

    expected_file = _choose_instances_list_file(
        run_dirs=all_source_dirs,
        mode=str(CONFIG.get("INSTANCE_LIST") or "selected"),
    )
    expected_instance_ids: List[str] = []
    if expected_file is not None:
        expected_instance_ids = _read_instance_ids_from_instances_jsonl(expected_file)

    # Fallback: if we still don't have an expected set, use the union of instances
    # that have patch artifacts across any variant.
    if not expected_instance_ids:
        expected_instance_ids = sorted({iid for s in have_by_variant.values() for iid in s})

    # Optional: user-specified instance filter (preserve list order).
    requested_instance_ids = CONFIG.get("INSTANCE_IDS")
    if isinstance(requested_instance_ids, list) and requested_instance_ids:
        requested = [str(x).strip() for x in requested_instance_ids if str(x).strip()]
        if requested:
            allowed = set(expected_instance_ids)
            expected_instance_ids = [iid for iid in requested if iid in allowed]

    if limit is not None:
        expected_instance_ids = expected_instance_ids[: int(limit)]

    # Prepare per-variant state.
    state: Dict[str, Dict[str, Any]] = {}
    for vt in variant_targets:
        vt.output_run_dir.mkdir(parents=True, exist_ok=True)
        (vt.output_run_dir / "repair" / "eval_work").mkdir(parents=True, exist_ok=True)
        missing = [iid for iid in expected_instance_ids if iid not in have_by_variant.get(vt.label, set())]
        state[vt.label] = {
            "label": vt.label,
            "run_dir": str(vt.output_run_dir),
            "source_run_dirs": [str(p) for p in vt.source_run_dirs],
            "per_instance": {},
            "missing_repairs": missing,
            "resolved_at_1": 0,
            "resolved_any": 0,
            "pass_to_pass_ok_at_1": 0,
            "fail_to_pass_ok_at_1": 0,
            "pass_to_pass_ok_any": 0,
            "fail_to_pass_ok_any": 0,
            "evaluated": 0,
            "progress": {
                "completed_instances": 0,
                "total_instances": len(expected_instance_ids),
                "last_instance_id": None,
            },
        }

    combined: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "variants_instance_first",
        "base_targets": [str(Path(os.path.expanduser(t)).resolve()) for t in CONFIG.get("TARGETS", [])],
        "variants": [vt.label for vt in variant_targets],
        "expected_total": len(expected_instance_ids),
        "expected_instances_file": str(expected_file) if expected_file is not None else None,
        "progress": {
            "completed_instances": 0,
            "total_instances": len(expected_instance_ids),
            "last_instance_id": None,
        },
        "per_variant": {},
        "eval_style": eval_style,
        "meta_source": meta_source,
        "dataset_name": str(CONFIG.get("DATASET_NAME") or "") if (eval_style == "harness" or meta_source in {"dataset", "auto"}) else None,
        "split": str(CONFIG.get("SPLIT") or "") if (eval_style == "harness" or meta_source in {"dataset", "auto"}) else None,
        "dataset_error": dataset_error,
    }

    def write_progress(force: bool = False) -> None:
        nonlocal combined
        generated_at = datetime.now(timezone.utc).isoformat()
        combined["generated_at"] = generated_at
        for vt in variant_targets:
            st = state[vt.label]
            variant_summary = _format_variant_eval_summary(
                st=st,
                expected_instance_ids=expected_instance_ids,
                generated_at=generated_at,
            )
            out_path = Path(st["run_dir"]) / "eval_summary.json"
            _atomic_write_json(out_path, variant_summary)

            combined["per_variant"][vt.label] = {
                "total_evaluated": variant_summary.get("total_evaluated"),
                "total_pass_to_pass_ok_at_1": variant_summary.get("total_pass_to_pass_ok_at_1"),
                "total_fail_to_pass_ok_at_1": variant_summary.get("total_fail_to_pass_ok_at_1"),
                "overall_regression_only_pass_rate": variant_summary.get("overall_regression_only_pass_rate"),
                "overall_fail_to_pass_pass_rate": variant_summary.get("overall_fail_to_pass_pass_rate"),
                "total_resolved_at_1": variant_summary.get("total_resolved_at_1"),
                "overall_pass_at_1": variant_summary.get("overall_pass_at_1"),
                "missing_repairs": variant_summary.get("missing_repairs"),
            }

        combined["variants_resolved_comparison"] = {
            iid: {
                vt.label: bool(
                    state[vt.label]
                    .get("per_instance", {})
                    .get(iid, {})
                    .get("resolved_at_1", False)
                )
                for vt in variant_targets
            }
            for iid in expected_instance_ids
        }
        _atomic_write_json(combined_out_path, combined)

    write_progress(force=True)

    for idx_iid, iid in enumerate(expected_instance_ids, start=1):
        combined["progress"]["completed_instances"] = idx_iid - 1
        combined["progress"]["last_instance_id"] = iid

        any_variant_has_patch = any(
            iid in have_by_variant.get(vt.label, set()) for vt in variant_targets
        )

        # If nobody produced a patch, record missing repairs and move on without pulling/running Docker.
        if not any_variant_has_patch:
            for vt in variant_targets:
                st = state[vt.label]
                st["evaluated"] += 1
                st["per_instance"][iid] = {
                    "instance_id": iid,
                    "resolved_at_1": False,
                    "resolved_any": False,
                    "num_candidates": 0,
                    "candidate_results": [],
                    "note": "Missing repair: no non-empty patch candidates found.",
                    "chosen_source_run_dir": None,
                }
            for vt in variant_targets:
                st = state[vt.label]
                st["progress"]["completed_instances"] = idx_iid
                st["progress"]["last_instance_id"] = iid
            combined["generated_at"] = datetime.now(timezone.utc).isoformat()
            combined["progress"]["completed_instances"] = idx_iid
            combined["progress"]["last_instance_id"] = iid
            if progress_write_every and (idx_iid % max(1, int(progress_write_every)) == 0):
                write_progress()
            continue

        meta: Optional[InstanceMeta] = None
        if meta_source in {"instances_jsonl", "auto"}:
            meta = meta_by_id.get(iid)
        if meta is None and meta_source in {"dataset", "auto"} and dataset_meta is not None:
            meta = _instance_meta_from_dataset_row(dataset_meta.get(iid), iid)

        if meta is None:
            # Record per-variant result for missing meta vs missing repair.
            for vt in variant_targets:
                st = state[vt.label]
                st["evaluated"] += 1

                if iid not in have_by_variant.get(vt.label, set()):
                    st["per_instance"][iid] = {
                        "instance_id": iid,
                        "resolved_at_1": False,
                        "resolved_any": False,
                        "num_candidates": 0,
                        "candidate_results": [],
                        "note": "Missing repair: no non-empty patch candidates found.",
                        "chosen_source_run_dir": None,
                    }
                    continue

                candidates, chosen_source = _collect_candidates_across_run_dirs(
                    run_dirs=vt.source_run_dirs,
                    instance_id=iid,
                )
                st["per_instance"][iid] = {
                    "instance_id": iid,
                    "resolved_at_1": False,
                    "resolved_any": False,
                    "num_candidates": len(candidates),
                    "candidate_results": [],
                    "note": _missing_meta_note_from_row(raw_rows_by_id.get(iid))
                    + (
                        "" if dataset_meta is None else " (dataset fallback attempted but unavailable/insufficient)"
                    ),
                    "chosen_source_run_dir": chosen_source,
                }

            for vt in variant_targets:
                st = state[vt.label]
                st["progress"]["completed_instances"] = idx_iid
                st["progress"]["last_instance_id"] = iid
            combined["generated_at"] = datetime.now(timezone.utc).isoformat()
            combined["progress"]["completed_instances"] = idx_iid
            combined["progress"]["last_instance_id"] = iid
            if progress_write_every and (idx_iid % max(1, int(progress_write_every)) == 0):
                write_progress()
            continue

        if not _docker_image_exists(meta.image_name):
            if pull_images:
                print(f"[docker] pulling missing image: {meta.image_name}")
                if not _docker_pull(meta.image_name):
                    # Mark each variant as evaluated: patch variants fail due to pull, others are missing repair.
                    for vt in variant_targets:
                        st = state[vt.label]
                        st["evaluated"] += 1
                        if iid not in have_by_variant.get(vt.label, set()):
                            st["per_instance"][iid] = {
                                "instance_id": iid,
                                "resolved_at_1": False,
                                "resolved_any": False,
                                "num_candidates": 0,
                                "candidate_results": [],
                                "note": "Missing repair: no non-empty patch candidates found.",
                            }
                        else:
                            st["per_instance"][iid] = {
                                "instance_id": iid,
                                "resolved_at_1": False,
                                "resolved_any": False,
                                "num_candidates": 0,
                                "candidate_results": [],
                                "note": f"Failed to pull instance image: {meta.image_name}",
                            }
                    for vt in variant_targets:
                        st = state[vt.label]
                        st["progress"]["completed_instances"] = idx_iid
                        st["progress"]["last_instance_id"] = iid
                    combined["generated_at"] = datetime.now(timezone.utc).isoformat()
                    combined["progress"]["completed_instances"] = idx_iid
                    if progress_write_every and (idx_iid % max(1, int(progress_write_every)) == 0):
                        write_progress()
                    continue
            else:
                for vt in variant_targets:
                    st = state[vt.label]
                    st["evaluated"] += 1
                    if iid not in have_by_variant.get(vt.label, set()):
                        st["per_instance"][iid] = {
                            "instance_id": iid,
                            "resolved_at_1": False,
                            "resolved_any": False,
                            "num_candidates": 0,
                            "candidate_results": [],
                            "note": "Missing repair: no non-empty patch candidates found.",
                        }
                    else:
                        st["per_instance"][iid] = {
                            "instance_id": iid,
                            "resolved_at_1": False,
                            "resolved_any": False,
                            "num_candidates": 0,
                            "candidate_results": [],
                            "note": f"Missing instance image (pull disabled): {meta.image_name}",
                        }
                for vt in variant_targets:
                    st = state[vt.label]
                    st["progress"]["completed_instances"] = idx_iid
                    st["progress"]["last_instance_id"] = iid
                combined["generated_at"] = datetime.now(timezone.utc).isoformat()
                combined["progress"]["completed_instances"] = idx_iid
                if progress_write_every and (idx_iid % max(1, int(progress_write_every)) == 0):
                    write_progress()
                continue

        try:
            for vt in variant_targets:
                st = state[vt.label]
                st["evaluated"] += 1

                if iid not in have_by_variant.get(vt.label, set()):
                    st["per_instance"][iid] = {
                        "instance_id": iid,
                        "resolved_at_1": False,
                        "resolved_any": False,
                        "num_candidates": 0,
                        "candidate_results": [],
                        "note": "Missing repair: no non-empty patch candidates found.",
                        "chosen_source_run_dir": None,
                    }
                    continue

                work_dir = vt.output_run_dir / "repair" / "eval_work"
                work_dir.mkdir(parents=True, exist_ok=True)

                candidates, chosen_source = _collect_candidates_across_run_dirs(
                    run_dirs=vt.source_run_dirs,
                    instance_id=iid,
                )

                if not candidates:
                    st["per_instance"][iid] = {
                        "instance_id": iid,
                        "resolved_at_1": False,
                        "resolved_any": False,
                        "num_candidates": 0,
                        "candidate_results": [],
                        "note": "Missing repair: no non-empty patch candidates found.",
                        "chosen_source_run_dir": chosen_source,
                    }
                    continue

                label1, patch1 = candidates[0]
                print(f"[eval] {vt.label} {iid} [1/{len(candidates)}] {label1} ...")
                if eval_style == "harness" and dataset_meta is not None:
                    row = dataset_meta.get(iid)
                    test_cmd, _ = _try_get_test_cmd_and_directives_from_dataset_row(row, iid)
                    if test_cmd.strip():
                        resolved1, details1 = _docker_run_eval_harness_style(
                            meta=meta,
                            model_patch=patch1,
                            keep_logs=keep_logs,
                            work_dir=work_dir,
                            timeout=timeout,
                            dataset_row=row,
                        )
                    else:
                        resolved1, details1 = _docker_run_eval(
                            meta=meta,
                            model_patch=patch1,
                            keep_logs=keep_logs,
                            work_dir=work_dir,
                            timeout=timeout,
                        )
                        details1 = dict(
                            details1,
                            eval_style="pytest",
                            note=(
                                "EVAL_STYLE=harness requested but test_cmd unavailable for this instance_id; "
                                "fell back to pytest exit-code scoring."
                            ),
                        )
                else:
                    resolved1, details1 = _docker_run_eval(
                        meta=meta,
                        model_patch=patch1,
                        keep_logs=keep_logs,
                        work_dir=work_dir,
                        timeout=timeout,
                    )
                    if eval_style == "harness" and dataset_meta is None:
                        details1 = dict(
                            details1,
                            eval_style="pytest",
                            note=(
                                "EVAL_STYLE=harness requested but dataset/harness unavailable; "
                                f"fell back to pytest. dataset_error={dataset_error}"
                            ),
                        )

                resolved_at_1 = bool(resolved1)
                resolved_any = resolved_at_1
                winning_index: Optional[int] = 1 if resolved_at_1 else None
                winning_label: Optional[str] = label1 if resolved_at_1 else None

                pass_to_pass_any = bool(details1.get("pass_to_pass_passed"))
                fail_to_pass_any = bool(details1.get("fail_to_pass_passed"))
                pass_to_pass_any_winner: Optional[int] = 1 if pass_to_pass_any else None
                fail_to_pass_any_winner: Optional[int] = 1 if fail_to_pass_any else None

                candidate_details: List[Dict[str, Any]] = [
                    dict(details1, candidate_label=label1, candidate_index=1)
                ]

                evaluate_any = bool(CONFIG.get("EVALUATE_ANY_REPAIR", True))
                if evaluate_any and (not resolved_any) and len(candidates) > 1:
                    for idx, (lbl, patch_txt) in enumerate(candidates[1:], start=2):
                        print(f"[eval] {vt.label} {iid} [{idx}/{len(candidates)}] {lbl} ...")
                        if eval_style == "harness" and dataset_meta is not None:
                            row = dataset_meta.get(iid)
                            test_cmd, _ = _try_get_test_cmd_and_directives_from_dataset_row(row, iid)
                            if test_cmd.strip():
                                res, det = _docker_run_eval_harness_style(
                                    meta=meta,
                                    model_patch=patch_txt,
                                    keep_logs=keep_logs,
                                    work_dir=work_dir,
                                    timeout=timeout,
                                    dataset_row=row,
                                )
                            else:
                                res, det = _docker_run_eval(
                                    meta=meta,
                                    model_patch=patch_txt,
                                    keep_logs=keep_logs,
                                    work_dir=work_dir,
                                    timeout=timeout,
                                )
                                det = dict(
                                    det,
                                    eval_style="pytest",
                                    note=(
                                        "EVAL_STYLE=harness requested but test_cmd unavailable for this instance_id; "
                                        "fell back to pytest exit-code scoring."
                                    ),
                                )
                        else:
                            res, det = _docker_run_eval(
                                meta=meta,
                                model_patch=patch_txt,
                                keep_logs=keep_logs,
                                work_dir=work_dir,
                                timeout=timeout,
                            )
                            if eval_style == "harness" and dataset_meta is None:
                                det = dict(
                                    det,
                                    eval_style="pytest",
                                    note=(
                                        "EVAL_STYLE=harness requested but dataset/harness unavailable; "
                                        f"fell back to pytest. dataset_error={dataset_error}"
                                    ),
                                )
                        det = dict(det, candidate_label=lbl, candidate_index=idx)
                        candidate_details.append(det)

                        if (not pass_to_pass_any) and det.get("pass_to_pass_passed"):
                            pass_to_pass_any = True
                            pass_to_pass_any_winner = idx
                        if (not fail_to_pass_any) and det.get("fail_to_pass_passed"):
                            fail_to_pass_any = True
                            fail_to_pass_any_winner = idx

                        if res:
                            resolved_any = True
                            winning_index = idx
                            winning_label = lbl
                            break

                st["per_instance"][iid] = {
                    "instance_id": iid,
                    "num_candidates": len(candidates),
                    "resolved_at_1": resolved_at_1,
                    "resolved_any": resolved_any,
                    "winning_candidate_index": winning_index,
                    "winning_candidate_label": winning_label,
                    "candidate_results": candidate_details,
                    "fail_to_pass_passed_at_1": bool(details1.get("fail_to_pass_passed")),
                    "pass_to_pass_passed_at_1": bool(details1.get("pass_to_pass_passed")),
                    "pass_to_pass_passed_any": pass_to_pass_any,
                    "fail_to_pass_passed_any": fail_to_pass_any,
                    "pass_to_pass_any_first_index": pass_to_pass_any_winner,
                    "fail_to_pass_any_first_index": fail_to_pass_any_winner,
                    "chosen_source_run_dir": chosen_source,
                }

                st["resolved_at_1"] += 1 if resolved_at_1 else 0
                st["resolved_any"] += 1 if resolved_any else 0
                st["pass_to_pass_ok_at_1"] += 1 if details1.get("pass_to_pass_passed") else 0
                st["fail_to_pass_ok_at_1"] += 1 if details1.get("fail_to_pass_passed") else 0
                st["pass_to_pass_ok_any"] += 1 if pass_to_pass_any else 0
                st["fail_to_pass_ok_any"] += 1 if fail_to_pass_any else 0
        finally:
            if delete_images_after_eval and _docker_image_exists(meta.image_name):
                print(f"[docker] removing image after eval (all variants): {meta.image_name}")
                _docker_remove_image(meta.image_name)

        for vt in variant_targets:
            st = state[vt.label]
            st["progress"]["completed_instances"] = idx_iid
            st["progress"]["last_instance_id"] = iid
        combined["generated_at"] = datetime.now(timezone.utc).isoformat()
        combined["progress"]["completed_instances"] = idx_iid
        combined["progress"]["last_instance_id"] = iid
        if progress_write_every and (idx_iid % max(1, int(progress_write_every)) == 0):
            write_progress()

    combined["generated_at"] = datetime.now(timezone.utc).isoformat()
    combined["progress"]["completed_instances"] = len(expected_instance_ids)
    combined["eval_style"] = eval_style
    if eval_style == "harness":
        combined["dataset_name"] = str(CONFIG.get("DATASET_NAME") or "")
        combined["split"] = str(CONFIG.get("SPLIT") or "")
        combined["dataset_error"] = dataset_error
    write_progress(force=True)
    return combined


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "targets",
        nargs="*",
        help="Optional targets (otherwise uses CONFIG['TARGETS']).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override CONFIG['LIMIT'].",
    )
    ap.add_argument(
        "--no-logs",
        action="store_true",
        help="Disable per-instance docker output logs (overrides CONFIG['KEEP_LOGS']).",
    )
    args = ap.parse_args()

    keep_logs = bool(CONFIG.get("KEEP_LOGS", True)) and (not bool(args.no_logs))
    limit = args.limit if args.limit is not None else CONFIG.get("LIMIT")
    timeout = int(CONFIG.get("TIMEOUT") or 0)

    variants = list(CONFIG.get("VARIANTS", []) or [])
    output_root = CONFIG.get("OUTPUT_ROOT")
    use_variants_mode = bool(variants) and bool(output_root) and (not bool(args.targets))

    if use_variants_mode:
        base_targets = list(CONFIG.get("TARGETS", []) or [])
        if not base_targets:
            raise SystemExit("No base targets provided. Edit CONFIG['TARGETS'] to point at results/<RUN> folders.")

        variant_targets = _build_variant_targets_from_config(
            base_targets=[str(x) for x in base_targets],
            variants=[str(x) for x in variants],
            output_root=str(output_root),
        )

        combined_out_path = Path(str(CONFIG.get("OUTPUT_SUMMARY_JSON") or "results/sweagent_combined_eval_summary.json"))
        pull_images = bool(CONFIG.get("PULL_IMAGES", True))
        delete_images_after_eval = bool(CONFIG.get("DELETE_IMAGES_AFTER_EVAL", True))
        progress_write_every = int(CONFIG.get("PROGRESS_WRITE_EVERY") or 0)

        combined = _evaluate_variants_instance_first(
            variant_targets=variant_targets,
            keep_logs=keep_logs,
            limit=limit,
            pull_images=pull_images,
            delete_images_after_eval=delete_images_after_eval,
            timeout=timeout,
            progress_write_every=progress_write_every,
            combined_out_path=combined_out_path,
        )
        print(f"wrote combined summary: {combined_out_path}")
        print(
            f"completed_instances={combined.get('progress', {}).get('completed_instances')} "
            f"expected_total={combined.get('expected_total')}"
        )
        return 0

    # Fallback: legacy "targets" mode (single target = one run_dir).
    targets: List[str] = list(args.targets) if args.targets else list(CONFIG.get("TARGETS", []))
    if not targets:
        raise SystemExit("No targets provided. Edit CONFIG['TARGETS'] or pass targets on the command line.")

    all_target_summaries: List[Dict[str, Any]] = []
    total_evaluated = 0
    total_resolved_at_1 = 0
    total_pass_to_pass_ok_at_1 = 0
    total_fail_to_pass_ok_at_1 = 0

    for t in targets:
        run_dir = _resolve_target_to_run_dir(t)
        label = _label_for_target(t)
        print(f"=== Evaluating target: {label} ({run_dir}) ===")
        summary = _evaluate_run_dir(run_dir, keep_logs=keep_logs, limit=limit, timeout=timeout)
        summary["label"] = label
        all_target_summaries.append(summary)
        total_evaluated += int(summary.get("evaluated") or 0)
        total_resolved_at_1 += int(summary.get("resolved_at_1") or 0)
        total_pass_to_pass_ok_at_1 += int(summary.get("pass_to_pass_ok_at_1") or 0)
        total_fail_to_pass_ok_at_1 += int(summary.get("fail_to_pass_ok_at_1") or 0)

    overall_pass1 = (total_resolved_at_1 / total_evaluated) if total_evaluated else None
    overall_regression_only = (
        (total_pass_to_pass_ok_at_1 / total_evaluated) if total_evaluated else None
    )
    overall_fail_to_pass = (
        (total_fail_to_pass_ok_at_1 / total_evaluated) if total_evaluated else None
    )
    combined_legacy: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": targets,
        "total_evaluated": total_evaluated,
        "total_pass_to_pass_ok_at_1": total_pass_to_pass_ok_at_1,
        "total_fail_to_pass_ok_at_1": total_fail_to_pass_ok_at_1,
        "overall_regression_only_pass_rate": overall_regression_only,
        "overall_fail_to_pass_pass_rate": overall_fail_to_pass,
        "total_resolved_at_1": total_resolved_at_1,
        "overall_pass_at_1": overall_pass1,
        "per_target": all_target_summaries,
    }

    out_path = Path(CONFIG.get("OUTPUT_SUMMARY_JSON", "results/swebench_eval_summary.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined_legacy, indent=2, sort_keys=True), encoding="utf-8")
    print("---")
    print(
        f"total_evaluated={total_evaluated} "
        f"total_resolved_at_1={total_resolved_at_1} overall_pass@1={overall_pass1}"
    )
    print(f"wrote aggregate summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
