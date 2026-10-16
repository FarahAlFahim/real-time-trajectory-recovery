from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from minisweagent.environments import get_environment_class
from minisweagent.run.benchmarks.swebench import get_sb_environment

import src.runners.minisweagent_swebench_lite  # noqa: F401  (registers the environment)
from src.adapters.minisweagent.docker import DockerEnvironment


@pytest.fixture
def docker_calls(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="container-id\n", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_runner_routes_docker_environment_class() -> None:
    assert get_environment_class("docker") is DockerEnvironment


def test_strip_git_history_runs_after_container_start(docker_calls) -> None:
    env = DockerEnvironment(image="example:latest", cwd="/testbed", strip_git_history=True)

    assert env.container_id == "container-id"
    assert [cmd[1] for cmd in docker_calls] == ["run", "exec"]
    assert docker_calls[1] == [
        "docker",
        "exec",
        "container-id",
        "bash",
        "-c",
        "set -e; cd /testbed; rm -rf .git; git init -q; git add -A; "
        "git -c user.email=agent@example.com -c user.name=agent commit -q --allow-empty -m base",
    ]


def test_default_matches_stock_docker_environment(docker_calls) -> None:
    env = DockerEnvironment(image="example:latest", cwd="/testbed")

    assert env.config.strip_git_history is False
    assert [cmd[1] for cmd in docker_calls] == ["run"]


def test_swebench_environment_keeps_instance_image(docker_calls) -> None:
    config = {"environment": {"environment_class": "docker", "cwd": "/testbed", "strip_git_history": True}}
    env = get_sb_environment(config, {"instance_id": "pallets__flask-4045", "image_name": "example:latest"})

    assert isinstance(env, DockerEnvironment)
    assert env.config.image == "example:latest"
    assert env.config.strip_git_history is True
