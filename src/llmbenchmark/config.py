"""SDK configuration.

Precedence (highest first): arguments passed to ``llmbenchmark.init()`` ->
``LLMBENCHMARK_*`` environment variables -> defaults below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

_ENV_PREFIX = "LLMBENCHMARK_"


@dataclass
class Config:
    endpoint: str = "http://localhost:8000"
    # repr=False: init() returns the Config, and a notebook that displays it
    # would otherwise write the key into the saved cell output.
    api_key: str | None = field(default=None, repr=False)
    project: str = "default"
    # Flush buffered call records to the backend every N records.
    batch_size: int = 50
    timeout: float = 10.0
    # Retries for retryable failures (network errors, 408/429/5xx).
    max_retries: int = 3
    # Where records go if the backend stays unreachable after retries.
    fallback_dir: Path = Path(".llmbenchmark")
    # Set LLMBENCHMARK_ENABLED=false to make every SDK call a local no-op send.
    enabled: bool = True
    # Record the git commit (and whether the working tree is dirty) with each run.
    capture_git: bool = True


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"expected one of {sorted(_TRUE | _FALSE - {''})}")


# Explicit per-field converters for env-var strings (no guessing from annotations).
_CONVERTERS: dict[str, Callable[[str], Any]] = {
    "endpoint": str,
    "api_key": str,
    "project": str,
    "batch_size": int,
    "timeout": float,
    "max_retries": int,
    "fallback_dir": Path,
    "enabled": _to_bool,
    "capture_git": _to_bool,
}


def _validate(config: Config) -> None:
    if config.batch_size < 1:
        raise ValueError(f"llmbenchmark batch_size must be >= 1, got {config.batch_size}")
    if config.max_retries < 0:
        raise ValueError(f"llmbenchmark max_retries must be >= 0, got {config.max_retries}")
    if config.timeout <= 0:
        raise ValueError(f"llmbenchmark timeout must be > 0, got {config.timeout}")


def load_config(**overrides) -> Config:
    """Build a Config from env vars, then apply non-None keyword overrides."""
    names = {f.name for f in fields(Config)}
    unknown = set(overrides) - names
    if unknown:
        raise TypeError(f"Unknown llmbenchmark config option(s): {sorted(unknown)}")

    values: dict[str, Any] = {}
    for name in names:
        var = _ENV_PREFIX + name.upper()
        raw = os.environ.get(var)
        if raw is None:
            continue
        try:
            values[name] = _CONVERTERS[name](raw)
        except ValueError as exc:
            raise ValueError(f"{var}={raw!r} is invalid: {exc}") from None
    values.update({k: v for k, v in overrides.items() if v is not None})
    if "fallback_dir" in values:
        values["fallback_dir"] = Path(values["fallback_dir"])
    config = Config(**values)
    config.endpoint = config.endpoint.rstrip("/")
    _validate(config)
    return config
