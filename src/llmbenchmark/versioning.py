"""Content fingerprints and lineage: what makes results reproducible.

A result is only meaningful if you know exactly what produced it. Prompts,
datasets and evaluation configs are identified by a hash of their content,
so "same prompt" and "same dataset" are checkable, not a naming convention.
Lineage records the environment (SDK, Python, packages, git commit).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
from importlib import metadata
from typing import Any

from .schemas import SCHEMA_VERSION, Lineage

logger = logging.getLogger("llmbenchmark")

# Packages whose version can change results; recorded when installed.
_TRACKED_PACKAGES = ("llmbenchmark", "anthropic", "openai", "deepeval", "httpx", "pydantic")


def content_hash(obj: Any) -> str:
    """sha256 hex of canonical JSON (sorted keys, no whitespace)."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def capture_lineage(*, sdk_version: str, capture_git: bool = True) -> Lineage:
    """Best effort: never raises (it runs while recording)."""
    lineage = Lineage(sdk_version=sdk_version, schema_version=SCHEMA_VERSION)
    try:
        lineage.python = platform.python_version()
        for name in _TRACKED_PACKAGES:
            try:
                lineage.packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        if capture_git:
            lineage.git_commit, lineage.git_dirty = _git_state()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("llmbenchmark: lineage capture failed: %s", exc)
    return lineage


def _git_state() -> tuple[str | None, bool | None]:
    """(commit, dirty) of the git repo containing the current directory."""
    try:
        commit = _git("rev-parse", "HEAD")
        if commit is None:
            return None, None
        # A repo's own .git/config can make `git status` run commands
        # (core.fsmonitor). The notebook may sit in an untrusted checkout, so
        # turn those hooks off and skip submodules (which have their own config).
        status = _git(
            "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false",
            "status", "--porcelain", "--ignore-submodules=all",
        )
        return commit, (bool(status) if status is not None else None)
    except Exception:
        return None, None


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
