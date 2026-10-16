"""mini-swe-agent Docker environment with optional git-history stripping (anti-leakage).

SWE-bench images ship the ground-truth fix commit inside the repository. With
``strip_git_history=True`` the repository under ``cwd`` is reset to a single synthetic
base commit at container startup, so the agent cannot recover the solution via
``git log --all`` / ``git show <sha>``. With the default ``False`` this environment
behaves exactly like mini-swe-agent's stock ``DockerEnvironment``.
"""

from __future__ import annotations

import shlex
import subprocess

from minisweagent import environments
from minisweagent.environments.docker import DockerEnvironment as _StockDockerEnvironment
from minisweagent.environments.docker import DockerEnvironmentConfig as _StockDockerEnvironmentConfig


# A locally patched mini-swe-agent may already strip history in its own _start_container().
_STOCK_STRIPS_GIT_HISTORY = "strip_git_history" in _StockDockerEnvironmentConfig.model_fields


class DockerEnvironmentConfig(_StockDockerEnvironmentConfig):
    strip_git_history: bool = False
    """Replace the repo's git history under `cwd` with a single synthetic base commit at startup.
    Used for benchmarks (e.g. SWE-bench) whose images ship the ground-truth fix commit inside the
    repo: this removes those objects/refs so the agent cannot recover the solution via
    `git log --all` / `git show <sha>`. `git diff`/`git apply` (editing and submission) still work
    against the base commit. Default is False (no change to normal behavior)."""


class DockerEnvironment(_StockDockerEnvironment):
    def __init__(self, *, config_class: type = DockerEnvironmentConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _start_container(self):
        super()._start_container()
        if self.config.strip_git_history and not _STOCK_STRIPS_GIT_HISTORY:
            self._strip_git_history()

    def _strip_git_history(self):
        """Reset the repo under `cwd` to a single synthetic commit of the current working tree.

        Removes all prior history, refs, remotes and their objects (including any ground-truth
        fix commit the benchmark image ships) so the agent cannot recover the solution from git.
        The working tree is unchanged, and `git diff`/`git apply` against this base still work.
        """
        cwd = shlex.quote(self.config.cwd)
        script = (
            f"set -e; cd {cwd}; "
            "rm -rf .git; "
            "git init -q; "
            "git add -A; "
            'git -c user.email=agent@example.com -c user.name=agent commit -q --allow-empty -m base'
        )
        self.logger.info("Stripping git history in container to a single base commit")
        subprocess.run(
            [self.config.executable, "exec", self.container_id, "bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,
            check=True,
        )


def use_as_docker_environment() -> None:
    """Resolve mini-swe-agent's ``environment_class: "docker"`` to this environment.

    The class is registered under the existing ``"docker"`` name (rather than selected by its
    import path) because mini-swe-agent's SWE-bench runner only assigns the instance image when
    ``environment_class == "docker"``.
    """
    environments._ENVIRONMENT_MAPPING["docker"] = f"{__name__}.{DockerEnvironment.__name__}"
