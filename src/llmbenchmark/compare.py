"""Compare two evaluations item by item, and manage per-project baselines.

    diff = llmbenchmark.compare(base=old_eval, head=new_eval)
    diff.summary()          # per metric: base vs head mean / pass rate, regressions
    diff.regressions        # items that passed before and fail now
    diff.warnings           # e.g. "datasets differ": not a like-for-like comparison

These are explicit reads, so they raise ``llmbenchmark.APIError`` on failure
(unlike recording, which never raises).
The server does the join; items are matched on dataset item id (or input).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import BaselineSet, is_valid_id
from .transport import APIError


@dataclass
class Comparison:
    base: dict[str, Any]
    head: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    differences: dict[str, Any] = field(default_factory=dict)
    items: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def regressions(self) -> list[dict[str, Any]]:
        return [i for i in self.items if i["status"] == "regressed"]

    @property
    def improvements(self) -> list[dict[str, Any]]:
        return [i for i in self.items if i["status"] == "improved"]

    def summary(self):
        import pandas as pd

        return pd.DataFrame(self.metrics).set_index("metric") if self.metrics else pd.DataFrame()

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.items)

    def __repr__(self) -> str:
        return (f"<llmbenchmark.Comparison metrics={len(self.metrics)} items={len(self.items)} "
                f"regressions={len(self.regressions)} warnings={len(self.warnings)}>")


def compare(base: Any = None, head: Any = None) -> Comparison:
    """Compare ``head`` against ``base`` (evaluation ids, Evaluation or EvalReport objects).

    Leave ``base`` out to compare against the project's baseline.
    """
    transport = _transport()
    params = {"head": _evaluation_id(head)}
    if base is not None:
        params["base"] = _evaluation_id(base)
    data = transport.get("/v1/compare", **params)
    return Comparison(**data)


def set_baseline(evaluation: Any, *, project: str | None = None) -> dict[str, Any]:
    """Make ``evaluation`` (and its run) the baseline new runs are compared against."""
    transport = _transport()
    evaluation_id = _evaluation_id(evaluation)
    info = transport.get(f"/v1/evaluations/{evaluation_id}")
    if project is None:
        project = transport.get(f"/v1/runs/{info['run_id']}")["project"]
    body = BaselineSet(project=project, run_id=info["run_id"], evaluation_id=evaluation_id)
    return transport.put("/v1/baselines", body)


def get_baseline(project: str | None = None) -> dict[str, Any] | None:
    import llmbenchmark

    transport = _transport()
    try:
        return transport.get("/v1/baselines", project=project or llmbenchmark.get_config().project)
    except APIError as exc:
        if exc.status_code == 404:
            return None
        raise


def _evaluation_id(value: Any) -> str:
    if value is None:
        raise ValueError("an evaluation (id, Evaluation or EvalReport) is required")
    candidate = getattr(value, "evaluation_id", None) or getattr(value, "id", None) or value
    if not is_valid_id(candidate):
        raise ValueError(f"not a valid evaluation id: {candidate!r}")
    return candidate


def _transport():
    import llmbenchmark

    llmbenchmark._ensure_init()
    return llmbenchmark._transport
