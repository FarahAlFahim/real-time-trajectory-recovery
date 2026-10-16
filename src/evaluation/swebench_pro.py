#!/usr/bin/env python3
"""Evaluate mini-swe-agent patches on SWE-bench Pro V2.

This evaluator intentionally does not reuse the SWE-bench Lite harness. Pro V2
ships an instance-specific verifier in ``v2/tasks/<instance_id>/tests``. For
every candidate patch, this script starts the official pristine amd64 image,
applies the patch against the recorded base commit, mounts the unchanged
official verifier at ``/tests``, runs ``/tests/test.sh``, and reads
``/logs/verifier/reward.txt``.

The official task tree is required locally so evaluation is pinned and does
not download mutable verifier files at runtime. Obtain it with::

    git clone https://github.com/scaleapi/SWE-bench_Pro-os.git

Then set ``TASKS_ROOT`` below (or ``SWE_BENCH_PRO_V2_TASKS``) to its
``v2/tasks`` directory.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG: dict[str, Any] = {
    # One or more result roots. Each variant is expected at TARGET/VARIANT.
    "TARGETS": [
        "results_swebench_pro/gpt_5_mini/element-hq__element-web"
    ],
    "VARIANTS": [
        "original",
        "realtime_recovery",
    ],
    # A per-repository JSON file or the directory containing all per-repository JSONs.
    "DATASET_PATH": (
        "data/swe_bench_pro"
    ),
    # Official scaleapi/SWE-bench_Pro-os checkout, specifically v2/tasks.
    "TASKS_ROOT": os.environ.get(
        "SWE_BENCH_PRO_V2_TASKS",
        "SWE-bench_Pro-os/v2/tasks",
    ),
    # Empty means use instances_selected.json from the run. This prevents a
    # one-instance pilot from being scored against every row in the repo.
    "INSTANCES_FILTER": [],
    "LIMIT": None,
    # Pass@1 evaluates the first unique patch (preds.json first). Enabling this
    # also evaluates additional saved *.patch files and reports pass@any.
    "EVALUATE_ANY_REPAIR": False,
    "OUTPUT_SUFFIX": "_tests",
    "KEEP_LOGS": True,
    "PROGRESS_WRITE_EVERY": 1,
    # Match the Lite evaluator: evaluate patches afresh on every invocation.
    "REDO": False,
    "DRY_RUN": False,
    # Official V2 verifier resources.
    "PLATFORM": "linux/amd64",
    "CPUS": "1",
    "MEMORY": "4096m",
    # None uses each official task.toml verifier timeout (normally 3000s).
    "TIMEOUT_SECONDS": None,
    "PULL_IMAGES": True,
    "DELETE_IMAGES_AFTER_EVAL": True,
    "VERIFY_TASK_CHECKSUMS": True,
}


@dataclass(frozen=True)
class InstanceMeta:
    instance_id: str
    image_name: str
    base_commit: str
    repo: str


@dataclass(frozen=True)
class VariantTarget:
    label: str
    source_dirs: tuple[Path, ...]
    output_dir: Path


@dataclass(frozen=True)
class TaskConfig:
    image_name: str
    verifier_timeout: int
    verifier_network: str
    verifier_env: dict[str, str]
    cpus: str
    memory: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_dir(value: str, *, label: str) -> Path:
    path = Path(os.path.expanduser(value)).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return path


def _load_dataset(path_value: str) -> OrderedDict[str, InstanceMeta]:
    path = Path(os.path.expanduser(path_value)).resolve()
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"DATASET_PATH does not exist: {path}")
    if not files:
        raise ValueError(f"No JSON dataset files found under {path}")

    result: OrderedDict[str, InstanceMeta] = OrderedDict()
    required = {"instance_id", "docker_image", "base_commit", "repo"}
    for file_path in files:
        rows = _read_json(file_path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list in {file_path}")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Non-object dataset row in {file_path}")
            missing = sorted(key for key in required if not str(row.get(key, "")).strip())
            if missing:
                raise ValueError(f"{file_path}: dataset row missing {missing}")
            instance_id = str(row["instance_id"]).strip()
            if instance_id in result:
                raise ValueError(f"Duplicate instance_id: {instance_id}")
            image_name = str(row["docker_image"]).strip()
            if not image_name.startswith("ghcr.io/scaleapi/swe-bench_pro-v2:"):
                raise ValueError(f"{instance_id}: unexpected V2 image {image_name!r}")
            if not re.fullmatch(r"[0-9a-fA-F]{40}", str(row["base_commit"]).strip()):
                raise ValueError(f"{instance_id}: invalid base_commit")
            result[instance_id] = InstanceMeta(
                instance_id=instance_id,
                image_name=image_name,
                base_commit=str(row["base_commit"]).strip(),
                repo=str(row["repo"]).strip(),
            )
    return result


def _build_targets() -> list[VariantTarget]:
    bases = [_resolve_dir(str(value), label="TARGET") for value in CONFIG["TARGETS"]]
    variants = [str(value).strip() for value in CONFIG["VARIANTS"] if str(value).strip()]
    if not variants:
        raise ValueError("VARIANTS must contain at least one variant")
    primary = bases[0]
    output_root = Path(str(primary) + str(CONFIG["OUTPUT_SUFFIX"])).resolve()
    return [
        VariantTarget(
            label=variant,
            source_dirs=tuple(base / variant for base in bases),
            output_dir=output_root / variant,
        )
        for variant in variants
    ]


def _ids_from_instances_file(run_dir: Path) -> list[str]:
    for name in ("instances_selected.json", "instances_selected.jsonl"):
        path = run_dir / name
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = _read_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list in {path}")
        return [
            str(row["instance_id"])
            for row in rows
            if isinstance(row, dict) and str(row.get("instance_id", "")).strip()
        ]
    return []


def _load_preds(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "repair" / "preds.json"
    if not path.exists():
        return {}
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object keyed by instance_id in {path}")
    return {str(key): value for key, value in raw.items() if isinstance(value, dict)}


def _collect_patch_candidates(run_dir: Path, instance_id: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    pred = _load_preds(run_dir).get(instance_id, {}).get("model_patch")
    if isinstance(pred, str) and pred.strip():
        seen.add(pred.strip())
        candidates.append(("preds.json", pred))

    bundle_path = run_dir / "repair" / "pro_predictions.json"
    if bundle_path.exists():
        bundle = _read_json(bundle_path)
        if isinstance(bundle, list):
            for row in bundle:
                if not isinstance(row, dict) or row.get("instance_id") != instance_id:
                    continue
                patch = row.get("patch")
                if isinstance(patch, str) and patch.strip() and patch.strip() not in seen:
                    seen.add(patch.strip())
                    candidates.append(("pro_predictions.json", patch))

    instance_dir = run_dir / "repair" / instance_id
    preferred = [instance_dir / f"{instance_id}.patch", instance_dir / "model.patch"]
    extras = sorted(
        path for path in instance_dir.glob("*.patch") if path not in preferred
    ) if instance_dir.is_dir() else []
    for path in preferred + extras:
        if not path.is_file():
            continue
        patch = path.read_text(encoding="utf-8", errors="replace")
        if patch.strip() and patch.strip() not in seen:
            seen.add(patch.strip())
            candidates.append((path.name, patch))
    return candidates


def _choose_instance_ids(
    dataset: OrderedDict[str, InstanceMeta], targets: list[VariantTarget]
) -> list[str]:
    # INSTANCE_IDS is the evaluator's canonical name. Accept INSTANCES_FILTER
    # as an alias so the same explicit ID list can be copied from the runner.
    configured_values = CONFIG.get("INSTANCE_IDS")
    if configured_values is None:
        configured_values = CONFIG.get("INSTANCES_FILTER", [])
    configured = [
        str(value).strip()
        for value in configured_values
        if str(value).strip()
    ]
    if configured:
        ids = configured
    else:
        ids: list[str] = []
        seen: set[str] = set()
        for target in targets:
            for run_dir in target.source_dirs:
                if not run_dir.is_dir():
                    continue
                for instance_id in _ids_from_instances_file(run_dir):
                    if instance_id not in seen:
                        seen.add(instance_id)
                        ids.append(instance_id)
        if not ids:
            for target in targets:
                for run_dir in target.source_dirs:
                    repair = run_dir / "repair"
                    if not repair.is_dir():
                        continue
                    for child in sorted(repair.iterdir()):
                        if child.is_dir() and child.name in dataset and child.name not in seen:
                            seen.add(child.name)
                            ids.append(child.name)
    unknown = [instance_id for instance_id in ids if instance_id not in dataset]
    if unknown:
        raise ValueError(f"Selected instance IDs missing from DATASET_PATH: {unknown[:5]}")
    limit = CONFIG.get("LIMIT")
    return ids[: int(limit)] if limit is not None else ids


def _task_release_root(tasks_root: Path) -> Path:
    if tasks_root.name != "tasks" or tasks_root.parent.name != "v2":
        raise ValueError(f"TASKS_ROOT must point to SWE-bench_Pro-os/v2/tasks: {tasks_root}")
    return tasks_root.parent


def _load_release_checksums(tasks_root: Path) -> dict[str, str]:
    checksum_path = _task_release_root(tasks_root) / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Official V2 checksum file not found: {checksum_path}")
    checksums: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", line)
        if not match:
            raise ValueError(f"Malformed checksum line in {checksum_path}: {line!r}")
        checksums[match.group(2).removeprefix("./")] = match.group(1).lower()
    return checksums


def _verify_task(task_dir: Path, checksums: dict[str, str] | None) -> None:
    required = (
        "task.toml", "tests/test.sh", "tests/run_script.sh", "tests/parser.py",
        "tests/config.json", "tests/test_patch.patch",
    )
    for relative in required:
        if not (task_dir / relative).is_file():
            raise FileNotFoundError(f"Missing official verifier asset: {task_dir / relative}")
    if checksums is None:
        return
    release_root = _task_release_root(task_dir.parent)
    relative_prefix = task_dir.relative_to(release_root).as_posix() + "/"
    expected = {path: digest for path, digest in checksums.items() if path.startswith(relative_prefix)}
    if not expected:
        raise ValueError(f"No SHA256SUMS entries found for {task_dir.name}")
    for relative, digest in expected.items():
        path = release_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Checksummed task file is missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch for official verifier asset: {path}")


def _load_task_config(task_dir: Path) -> TaskConfig:
    raw = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    verifier = raw.get("verifier") or {}
    environment = raw.get("environment") or {}
    verifier_env = verifier.get("env") or {}
    if not isinstance(verifier_env, dict):
        raise ValueError(f"Invalid [verifier.env] in {task_dir / 'task.toml'}")
    image = str(environment.get("docker_image") or "").strip()
    if not image:
        raise ValueError(f"Missing environment.docker_image in {task_dir / 'task.toml'}")
    return TaskConfig(
        image_name=image,
        verifier_timeout=int(float(verifier.get("timeout_sec", 3000))),
        verifier_network=str(verifier.get("network_mode") or "public").strip().lower(),
        verifier_env={str(key): str(value) for key, value in verifier_env.items()},
        cpus=str(environment.get("cpus", CONFIG["CPUS"])),
        memory=f"{int(environment.get('memory_mb', 4096))}m",
    )


def _parse_test_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("Official verifier test list is not a list")
    return [str(item) for item in parsed]


def _official_test_sets(task_dir: Path) -> tuple[list[str], list[str]]:
    config = _read_json(task_dir / "tests" / "config.json")
    if not isinstance(config, dict):
        raise ValueError(f"Invalid verifier config: {task_dir / 'tests/config.json'}")
    return (
        _parse_test_list(config.get("fail_to_pass")),
        _parse_test_list(config.get("pass_to_pass")),
    )


def _official_repo_commit(tasks_root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(tasks_root.parent.parent), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _docker_image_exists(image: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def _pull_image(image: str) -> None:
    cmd = ["docker", "pull", "--platform", str(CONFIG["PLATFORM"]), image]
    if subprocess.run(cmd, check=False).returncode != 0:
        raise RuntimeError(f"Unable to pull Docker image: {image}")


def _remove_image(image: str) -> None:
    subprocess.run(
        ["docker", "image", "rm", "-f", image], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False,
    )


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "candidate"


def _container_name(instance_id: str, variant: str, candidate_index: int) -> str:
    digest = hashlib.sha256(
        f"{instance_id}:{variant}:{candidate_index}:{time.time_ns()}".encode()
    ).hexdigest()[:16]
    return f"sbp-v2-eval-{digest}"


def _cached_candidate_report(
    *, meta: InstanceMeta, patch: str, variant: str, candidate_index: int,
    candidate_label: str, output_dir: Path,
) -> dict[str, Any] | None:
    """Return a reusable report for this exact patch and environment."""

    candidate_dir = output_dir / "repair" / "eval_work" / meta.instance_id / (
        f"candidate_{candidate_index:02d}_{_safe_label(candidate_label)}"
    )
    report_path = candidate_dir / "report.json"
    if not report_path.exists():
        return None
    report = _read_json(report_path)
    patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    if not (
        isinstance(report, dict)
        and report.get("patch_sha256") == patch_sha256
        and report.get("image_name") == meta.image_name
        and report.get("base_commit") == meta.base_commit
        and isinstance(report.get("fail_to_pass_passed"), bool)
        and isinstance(report.get("pass_to_pass_passed"), bool)
    ):
        return None
    report["cached"] = True
    return report


def _evaluate_candidate(
    *, meta: InstanceMeta, task_dir: Path, task_config: TaskConfig, patch: str, variant: str,
    candidate_index: int, candidate_label: str, output_dir: Path, redo: bool,
) -> dict[str, Any]:
    candidate_dir = output_dir / "repair" / "eval_work" / meta.instance_id / (
        f"candidate_{candidate_index:02d}_{_safe_label(candidate_label)}"
    )
    report_path = candidate_dir / "report.json"
    patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    if not redo:
        cached_report = _cached_candidate_report(
            meta=meta,
            patch=patch,
            variant=variant,
            candidate_index=candidate_index,
            candidate_label=candidate_label,
            output_dir=output_dir,
        )
        if cached_report is not None:
            return cached_report

    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    logs_dir = candidate_dir / "logs"
    verifier_logs = logs_dir / "verifier"
    verifier_logs.mkdir(parents=True, exist_ok=True)
    patch_path = candidate_dir / "model.patch"
    patch_path.write_text(patch if patch.endswith("\n") else patch + "\n", encoding="utf-8")

    container = _container_name(meta.instance_id, variant, candidate_index)
    base = meta.base_commit
    shell_script = f"""
set -uo pipefail
repo=''
for candidate in /app /testbed; do
  if git -C "$candidate" rev-parse --show-toplevel >/dev/null 2>&1; then repo="$candidate"; break; fi
done
if [ -z "$repo" ]; then
  echo 'ERROR: repository not found at /app or /testbed'
  mkdir -p /logs/verifier; echo 0 > /logs/verifier/reward.txt
  exit 20
fi
cd "$repo"
git cat-file -e {base}^{{commit}} || {{ echo 0 > /logs/verifier/reward.txt; exit 21; }}
git reset --hard {base} || {{ echo 0 > /logs/verifier/reward.txt; exit 22; }}
git clean -fd || {{ echo 0 > /logs/verifier/reward.txt; exit 23; }}
if [ ! -s /workspace/model.patch ]; then
  echo 'ERROR: empty model patch'; echo 0 > /logs/verifier/reward.txt; exit 24
fi
git apply --check /workspace/model.patch || {{ echo 'ERROR: model patch check failed'; echo 0 > /logs/verifier/reward.txt; exit 25; }}
git apply --verbose /workspace/model.patch || {{ echo 'ERROR: model patch apply failed'; echo 0 > /logs/verifier/reward.txt; exit 26; }}
bash /tests/test.sh
""".strip()
    command = [
        "docker", "run", "--rm", "--name", container,
        "--platform", str(CONFIG["PLATFORM"]), "--cpus", task_config.cpus,
        "--memory", task_config.memory, "-e", "PYTHONIOENCODING=utf-8",
        "-v", f"{patch_path.resolve()}:/workspace/model.patch:ro",
        "-v", f"{(task_dir / 'tests').resolve()}:/tests:ro",
        "-v", f"{logs_dir.resolve()}:/logs", meta.image_name,
        "bash", "-c", shell_script,
    ]
    if task_config.verifier_network in {"none", "no-network", "offline"}:
        command[5:5] = ["--network", "none"]
    for key, value in sorted(task_config.verifier_env.items()):
        command[5:5] = ["-e", f"{key}={value}"]

    timeout = CONFIG["TIMEOUT_SECONDS"]
    timeout_seconds = int(timeout) if timeout is not None else task_config.verifier_timeout

    started = _utc_now()
    timed_out = False
    try:
        proc = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=timeout_seconds, check=False,
        )
        output = proc.stdout or ""
        returncode: int | None = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        output = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        returncode = None
        subprocess.run(
            ["docker", "rm", "-f", container], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=False,
        )

    reward_path = verifier_logs / "reward.txt"
    reward = reward_path.read_text(encoding="utf-8").strip() if reward_path.exists() else None
    fail_to_pass, pass_to_pass = _official_test_sets(task_dir)
    output_path = verifier_logs / "output.json"
    passed_tests: set[str] = set()
    if output_path.exists():
        parsed_output = _read_json(output_path)
        if isinstance(parsed_output, dict):
            passed_tests = {
                str(test.get("name", ""))
                for test in parsed_output.get("tests", [])
                if isinstance(test, dict) and test.get("status") == "PASSED"
            }
    fail_to_pass_passed = bool(fail_to_pass) and set(fail_to_pass) <= passed_tests
    pass_to_pass_passed = set(pass_to_pass) <= passed_tests
    resolved = reward == "1" and returncode == 0 and not timed_out
    report: dict[str, Any] = {
        "instance_id": meta.instance_id, "variant": variant,
        "candidate_index": candidate_index, "candidate_label": candidate_label,
        "resolved": resolved, "reward": reward, "timed_out": timed_out,
        "docker_returncode": returncode, "image_name": meta.image_name,
        "patch_sha256": patch_sha256, "verifier_timeout_seconds": timeout_seconds,
        "verifier_network": task_config.verifier_network,
        "fail_to_pass_passed": fail_to_pass_passed,
        "pass_to_pass_passed": pass_to_pass_passed,
        "fail_to_pass_count": len(fail_to_pass),
        "pass_to_pass_count": len(pass_to_pass),
        "base_commit": meta.base_commit, "repo": meta.repo,
        "started_at_utc": started, "ended_at_utc": _utc_now(), "cached": False,
    }
    if CONFIG["KEEP_LOGS"]:
        (candidate_dir / "docker_output.log").write_text(output, encoding="utf-8")
    _atomic_write_json(report_path, report)
    return report


def _empty_variant_state(target: VariantTarget, expected: list[str]) -> dict[str, Any]:
    return {
        "variant": target.label, "expected_instance_ids": expected,
        "total_expected": len(expected), "evaluated_instances": 0,
        "missing_patch_instances": [], "resolved_at_1_instances": [],
        "resolved_any_instances": [], "instance_results": OrderedDict(),
        "fail_to_pass_ok_at_1": 0, "pass_to_pass_ok_at_1": 0,
        "progress": {
            "completed_instances": 0,
            "total_instances": len(expected),
            "last_instance_id": None,
        },
    }


def _write_variant_summary(target: VariantTarget, state: dict[str, Any]) -> None:
    total = int(state["total_expected"])
    resolved_at_1 = len(state["resolved_at_1_instances"])
    resolved_any = len(state["resolved_any_instances"])
    evaluated = int(state["evaluated_instances"])
    pass_to_pass = int(state["pass_to_pass_ok_at_1"])
    fail_to_pass = int(state["fail_to_pass_ok_at_1"])
    generated_at = _utc_now()
    per_target = {
        "label": target.label,
        "run_dir": str(target.output_dir),
        "source_run_dirs": [str(path) for path in target.source_dirs],
        "preds_json": next(
            (str(path / "repair" / "preds.json") for path in target.source_dirs if (path / "repair" / "preds.json").exists()),
            None,
        ),
        "instances_jsonl": next(
            (
                str(path / name)
                for path in target.source_dirs
                for name in ("instances.jsonl", "instances.json")
                if (path / name).exists()
            ),
            None,
        ),
        "evaluated": evaluated,
        "pass_to_pass_ok_at_1": pass_to_pass,
        "fail_to_pass_ok_at_1": fail_to_pass,
        "regression_only_pass_rate": pass_to_pass / evaluated if evaluated else None,
        "fail_to_pass_pass_rate": fail_to_pass / evaluated if evaluated else None,
        "resolved_at_1": resolved_at_1,
        "pass_at_1": resolved_at_1 / evaluated if evaluated else None,
        "per_instance": state["instance_results"],
        "missing_repairs": sorted(state["missing_patch_instances"]),
        "expected_total": total,
        "progress": state["progress"],
    }
    summary = {
        "generated_at": generated_at,
        "targets": [str(path) for path in target.source_dirs],
        "total_evaluated": evaluated,
        "total_pass_to_pass_ok_at_1": pass_to_pass,
        "total_fail_to_pass_ok_at_1": fail_to_pass,
        "overall_regression_only_pass_rate": pass_to_pass / evaluated if evaluated else None,
        "overall_fail_to_pass_pass_rate": fail_to_pass / evaluated if evaluated else None,
        "total_resolved_at_1": resolved_at_1,
        "overall_pass_at_1": resolved_at_1 / evaluated if evaluated else None,
        "missing_repairs": sorted(state["missing_patch_instances"]),
        "progress": state["progress"],
        "per_target": [per_target],
    }
    _atomic_write_json(target.output_dir / "eval_summary.json", summary)


def _source_run_dir(target: VariantTarget, instance_id: str) -> Path | None:
    for run_dir in target.source_dirs:
        if _collect_patch_candidates(run_dir, instance_id):
            return run_dir
    for run_dir in target.source_dirs:
        if run_dir.is_dir():
            return run_dir
    return None


def _expected_instances_file(targets: list[VariantTarget]) -> Path | None:
    for target in targets:
        for run_dir in target.source_dirs:
            for name in ("instances_selected.json", "instances_selected.jsonl"):
                path = run_dir / name
                if path.exists():
                    return path
    return None


def _write_combined_summary(
    output_root: Path,
    states: dict[str, dict[str, Any]],
    *,
    targets: list[VariantTarget],
    expected_instance_ids: list[str],
) -> None:
    completed = max(
        (int(state["progress"]["completed_instances"]) for state in states.values()),
        default=0,
    )
    last_instance = next(
        (
            state["progress"]["last_instance_id"]
            for state in states.values()
            if state["progress"]["last_instance_id"] is not None
        ),
        None,
    )
    combined = {
        "base_targets": [str(Path(value).expanduser().resolve()) for value in CONFIG["TARGETS"]],
        "dataset_error": None,
        "dataset_name": "ScaleAI/SWE-bench_Pro",
        "eval_style": "official_v2_verifier",
        "expected_instances_file": (
            str(_expected_instances_file(targets)) if _expected_instances_file(targets) else None
        ),
        "expected_total": len(expected_instance_ids),
        "generated_at": _utc_now(),
        "meta_source": "dataset_json+official_v2_tasks",
        "mode": "variants_instance_first",
        "per_variant": {
            label: {
                "missing_repairs": sorted(state["missing_patch_instances"]),
                "overall_fail_to_pass_pass_rate": (
                    state["fail_to_pass_ok_at_1"] / state["evaluated_instances"]
                    if state["evaluated_instances"] else None
                ),
                "overall_pass_at_1": (
                    len(state["resolved_at_1_instances"]) / state["evaluated_instances"]
                    if state["evaluated_instances"] else None
                ),
                "overall_regression_only_pass_rate": (
                    state["pass_to_pass_ok_at_1"] / state["evaluated_instances"]
                    if state["evaluated_instances"] else None
                ),
                "total_evaluated": state["evaluated_instances"],
                "total_fail_to_pass_ok_at_1": state["fail_to_pass_ok_at_1"],
                "total_pass_to_pass_ok_at_1": state["pass_to_pass_ok_at_1"],
                "total_resolved_at_1": len(state["resolved_at_1_instances"]),
            }
            for label, state in states.items()
        },
        "progress": {
            "completed_instances": completed,
            "last_instance_id": last_instance,
            "total_instances": len(expected_instance_ids),
        },
        "split": "test",
        "variants": [target.label for target in targets],
        "variants_resolved_comparison": {
            instance_id: {
                target.label: bool(
                    states[target.label]["instance_results"]
                    .get(instance_id, {})
                    .get("resolved_at_1", False)
                )
                for target in targets
            }
            for instance_id in expected_instance_ids
        },
    }
    _atomic_write_json(output_root / "combined_eval_summary.json", combined)


def evaluate(*, redo: bool, dry_run: bool) -> None:
    dataset = _load_dataset(str(CONFIG["DATASET_PATH"]))
    targets = _build_targets()
    tasks_root = _resolve_dir(str(CONFIG["TASKS_ROOT"]), label="TASKS_ROOT")
    _task_release_root(tasks_root)
    checksums = _load_release_checksums(tasks_root) if CONFIG["VERIFY_TASK_CHECKSUMS"] else None
    official_commit = _official_repo_commit(tasks_root)
    selected = _choose_instance_ids(dataset, targets)
    if not selected:
        raise ValueError("No instances selected for evaluation")

    # Inventory every selected variant-instance pair before starting Docker so
    # empty or missing patches are visible immediately rather than gradually.
    patch_inventory: dict[
        tuple[str, str], tuple[Path | None, list[tuple[str, str]]]
    ] = {}
    missing_patches: list[tuple[str, str]] = []
    for instance_id in selected:
        for target in targets:
            run_dir = _source_run_dir(target, instance_id)
            candidates = (
                _collect_patch_candidates(run_dir, instance_id) if run_dir else []
            )
            patch_inventory[(target.label, instance_id)] = (run_dir, candidates)
            if not candidates:
                missing_patches.append((target.label, instance_id))

    total_pairs = len(selected) * len(targets)
    print(
        f"Patch preflight: {total_pairs - len(missing_patches)}/{total_pairs} "
        "variant-instance pair(s) have a non-empty patch."
    )
    if missing_patches:
        print("Missing/empty patches detected before evaluation:")
        for variant, instance_id in missing_patches:
            print(f"  {variant}: {instance_id}")

    if dry_run:
        print(f"SWE-bench Pro V2 dry run: {len(selected)} instance(s)")
        for instance_id in selected:
            meta = dataset[instance_id]
            task_dir = tasks_root / instance_id
            _verify_task(task_dir, checksums)
            task_config = _load_task_config(task_dir)
            if task_config.image_name != meta.image_name:
                raise ValueError(
                    f"{instance_id}: dataset and official task images do not match"
                )
            for target in targets:
                _, candidates = patch_inventory[(target.label, instance_id)]
                print(
                    f"  {instance_id} [{target.label}]: "
                    f"{len(candidates)} unique patch candidate(s)"
                )
                if not candidates:
                    raise ValueError(
                        f"{instance_id} [{target.label}]: no non-empty patch found"
                    )
        print("Dry run passed; no Docker containers started and no summaries written.")
        return

    output_root = targets[0].output_dir.parent
    states = {target.label: _empty_variant_state(target, selected) for target in targets}
    # Reset stale summaries immediately at the start of a real evaluation,
    # matching the Lite evaluator's zero-progress initialization behavior.
    for target in targets:
        _write_variant_summary(target, states[target.label])
    _write_combined_summary(
        output_root,
        states,
        targets=targets,
        expected_instance_ids=selected,
    )
    print(f"SWE-bench Pro V2 instances selected: {len(selected)}")
    print(f"Official tasks: {tasks_root}")
    if official_commit:
        print(f"Official task repository commit: {official_commit}")

    for position, instance_id in enumerate(selected, start=1):
        meta = dataset[instance_id]
        task_dir = tasks_root / instance_id
        _verify_task(task_dir, checksums)
        task_config = _load_task_config(task_dir)
        if task_config.image_name != meta.image_name:
            raise ValueError(
                f"{instance_id}: dataset image {meta.image_name!r} does not match "
                f"official task image {task_config.image_name!r}"
            )
        print(f"[{position}/{len(selected)}] {instance_id}")
        instance_needs_docker = False
        for target in targets:
            _, candidates = patch_inventory[(target.label, instance_id)]
            selected_candidates = (
                candidates if CONFIG["EVALUATE_ANY_REPAIR"] else candidates[:1]
            )
            for index, (label, patch) in enumerate(selected_candidates, start=1):
                if redo or _cached_candidate_report(
                    meta=meta,
                    patch=patch,
                    variant=target.label,
                    candidate_index=index,
                    candidate_label=label,
                    output_dir=target.output_dir,
                ) is None:
                    instance_needs_docker = True
                    break
            if instance_needs_docker:
                break

        if instance_needs_docker and not _docker_image_exists(meta.image_name):
            if not CONFIG["PULL_IMAGES"]:
                raise RuntimeError(f"Docker image missing and PULL_IMAGES=False: {meta.image_name}")
            _pull_image(meta.image_name)

        for target in targets:
            state = states[target.label]
            state["progress"]["completed_instances"] = position
            state["progress"]["last_instance_id"] = instance_id
            run_dir, candidates = patch_inventory[(target.label, instance_id)]
            if not candidates:
                state["missing_patch_instances"].append(instance_id)
                state["instance_results"][instance_id] = {
                    "resolved_at_1": False, "resolved_any": False,
                    "missing_patch": True, "candidates": [],
                }
                # An empty/missing submission is a scored Pass@1 failure, not
                # an instance to remove from the denominator.
                state["evaluated_instances"] += 1
                print(f"  {target.label}: missing patch")
                _write_variant_summary(target, state)
                continue

            selected_candidates = candidates if CONFIG["EVALUATE_ANY_REPAIR"] else candidates[:1]
            candidate_reports: list[dict[str, Any]] = []
            if dry_run:
                for index, (label, patch) in enumerate(selected_candidates, start=1):
                    candidate_reports.append({
                        "candidate_index": index, "candidate_label": label,
                        "patch_bytes": len(patch.encode("utf-8")),
                        "resolved": False, "dry_run": True,
                    })
            else:
                for index, (label, patch) in enumerate(selected_candidates, start=1):
                    candidate_reports.append(_evaluate_candidate(
                        meta=meta, task_dir=task_dir, task_config=task_config,
                        patch=patch, variant=target.label,
                        candidate_index=index, candidate_label=label,
                        output_dir=target.output_dir, redo=redo,
                    ))

            resolved_at_1 = bool(candidate_reports and candidate_reports[0].get("resolved"))
            resolved_any = any(bool(report.get("resolved")) for report in candidate_reports)
            if resolved_at_1:
                state["resolved_at_1_instances"].append(instance_id)
            if resolved_any:
                state["resolved_any_instances"].append(instance_id)
            state["evaluated_instances"] += 1
            first = candidate_reports[0]
            if first.get("fail_to_pass_passed"):
                state["fail_to_pass_ok_at_1"] += 1
            if first.get("pass_to_pass_passed"):
                state["pass_to_pass_ok_at_1"] += 1
            state["instance_results"][instance_id] = {
                "instance_id": instance_id,
                "num_candidates": len(candidates),
                "resolved_at_1": resolved_at_1,
                "resolved_any": resolved_any,
                "winning_candidate_index": 1 if resolved_at_1 else None,
                "winning_candidate_label": first.get("candidate_label") if resolved_at_1 else None,
                "candidate_results": candidate_reports,
                "fail_to_pass_passed_at_1": bool(first.get("fail_to_pass_passed")),
                "pass_to_pass_passed_at_1": bool(first.get("pass_to_pass_passed")),
                "chosen_source_run_dir": str(run_dir),
            }
            print(f"  {target.label}: resolved_at_1={resolved_at_1}")
            _write_variant_summary(target, state)

        if CONFIG["DELETE_IMAGES_AFTER_EVAL"] and not dry_run:
            _remove_image(meta.image_name)
        if position % max(1, int(CONFIG["PROGRESS_WRITE_EVERY"])) == 0:
            for target in targets:
                _write_variant_summary(target, states[target.label])
            _write_combined_summary(
                output_root,
                states,
                targets=targets,
                expected_instance_ids=selected,
            )

    for target in targets:
        _write_variant_summary(target, states[target.label])
    _write_combined_summary(
        output_root,
        states,
        targets=targets,
        expected_instance_ids=selected,
    )
    print(f"Combined summary: {output_root / 'combined_eval_summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redo", action="store_true", help="Re-evaluate cached candidates")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without Docker evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(redo=bool(args.redo or CONFIG["REDO"]), dry_run=bool(args.dry_run or CONFIG["DRY_RUN"]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
