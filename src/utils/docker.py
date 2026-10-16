"""Docker container/image cleanup helpers shared by runners and evaluators."""

from __future__ import annotations

import subprocess


def stop_and_remove_containers_for_image(image: str) -> None:
    """Stop and force-remove ALL containers (running + stopped) created from *image*."""
    img = str(image).strip()
    if not img:
        return
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "-q", "--filter", f"ancestor={img}"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return
    for cid in [x for x in out.splitlines() if x.strip()]:
        subprocess.run(["docker", "rm", "-f", cid], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def remove_minisweagent_containers() -> None:
    """Force-remove all containers whose name starts with 'minisweagent-'."""
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "-q", "--filter", "name=minisweagent-"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return
    for cid in [x for x in out.splitlines() if x.strip()]:
        subprocess.run(["docker", "rm", "-f", cid], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prune() -> None:
    """Remove dangling images, stopped containers, and unused build cache."""
    subprocess.run(
        ["docker", "system", "prune", "-f"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def image_rm(image: str, *, force: bool) -> None:
    img = str(image).strip()
    if not img:
        return
    cmd = ["docker", "image", "rm"]
    if force:
        cmd.append("-f")
    cmd.append(img)
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
