"""llmbenchmark: record LLM calls and agent traces, and evaluate them.

Everything importable from this module is the public API. Anything else
(``llmbenchmark.transport``, ``llmbenchmark.providers``, ...) is internal and may
change between minor versions.

    import llmbenchmark
    llmbenchmark.init(endpoint="http://localhost:8000", project="qa-bench")
    prompt = llmbenchmark.Prompt("qa", system="Answer briefly.", template="{input}", version="v1")
    with llmbenchmark.start_run(name="claude-baseline", model="claude-opus-5-5",
                                generation_params={"max_tokens": 1024}, prompt=prompt) as run:
        with run.track(input=question) as t:
            t.response = client.messages.create(...)
"""

from __future__ import annotations

import atexit
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .assets import Dataset, Prompt
from .compare import Comparison, compare, get_baseline, set_baseline
from .config import Config, load_config
from .evaluation import Evaluation
from .run import Run, TraceHandle, TrackedCall
from .schemas import DatasetItem, Message, Record, RunMode, RunMetric, Score, Span, ToolCall
from .tracing import observe, propagate, span
from .transport import APIError, Transport

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # setup
    "init",
    "get_config",
    "replay",
    # runs and recording
    "start_run",
    "Run",
    "TrackedCall",
    "TraceHandle",
    "span",
    "observe",
    "propagate",
    # versioned inputs
    "Prompt",
    "Dataset",
    "DatasetItem",
    # results
    "Evaluation",
    "compare",
    "Comparison",
    "set_baseline",
    "get_baseline",
    "APIError",
    # data types
    "Record",
    "Message",
    "ToolCall",
    "Span",
    "Score",
    "RunMetric",
]

_config: Config | None = None
_transport: Transport | None = None
_open_runs: "weakref.WeakSet[Run]" = weakref.WeakSet()


def init(
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    project: str | None = None,
    **options: Any,
) -> Config:
    """Configure the SDK. Unset arguments fall back to LLMBENCHMARK_* env vars."""
    global _config, _transport
    config = load_config(endpoint=endpoint, api_key=api_key, project=project, **options)
    old = _transport
    _config, _transport = config, Transport(config)
    # Re-running the init cell must not break runs started earlier: they keep
    # their own transport (finished runs too: you can still evaluate them), so
    # only close the old one when no run you still hold uses it.
    if old is not None and not any(r._transport is old for r in list(_open_runs)):
        old.close()
    return _config


def get_config() -> Config:
    _ensure_init()
    assert _config is not None
    return _config


def start_run(
    name: str | None = None,
    *,
    mode: RunMode = "managed",
    model: str | None = None,
    provider: str | None = None,
    generation_params: dict[str, Any] | None = None,
    prompt: Prompt | None = None,
    dataset: Dataset | None = None,
    project: str | None = None,
    tags: Sequence[str] | None = None,
) -> Run:
    """Create a run on the backend and return it. Use it as a context manager.

    mode="managed" (default): you will evaluate with llmbenchmark.eval; model,
    generation_params and prompt are expected (a warning names any missing).
    mode="external": you evaluate elsewhere and upload results with
    run.evaluation(...) or run.log_results(...).
    """
    _ensure_init()
    assert _config is not None and _transport is not None
    run = Run(
        _transport, _config, name=name, project=project, mode=mode, model=model, provider=provider,
        generation_params=generation_params, prompt=prompt, dataset=dataset, tags=tags,
    )
    _open_runs.add(run)
    return run.start()


def replay(file: str | Path) -> tuple[int, int]:
    """Re-send records saved to a fallback .jsonl file. Returns (sent, failed)."""
    _ensure_init()
    assert _transport is not None
    return _transport.replay(file)


def _ensure_init() -> None:
    if _transport is None:
        init()  # env-var / default configuration


@atexit.register
def _finish_open_runs() -> None:
    # Don't lose buffered records if the kernel shuts down mid-run.
    for run in list(_open_runs):
        try:
            run.flush()
            run.flush_evaluations()
        except Exception:
            pass
