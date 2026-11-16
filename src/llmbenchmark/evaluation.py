"""Evaluations: one immutable, versioned scoring pass over a run.

Every score belongs to an Evaluation, which records *who* scored (evaluator
name + version), *how* (metrics, thresholds, criteria, judge model) and a
hash of that config. Re-scoring a run never overwrites: it creates a new
Evaluation (optionally ``supersedes=`` the old one). Once finalized, an
evaluation accepts no new scores.

Bring-your-own-evaluation users create one directly:

    with run.evaluation(evaluator="team-x-rubric", version="1.2",
                        metrics={"accuracy": {"threshold": 1}}) as ev:
        rec = run.log(input=q, output=answer, expected_output=gold)
        ev.score(rec, "accuracy", score=1.0, reason="matches gold")
        ev.metric("accuracy", 0.87, n=500)        # run-level number
    # leaving the block finalizes the evaluation

``llmbenchmark.eval.evaluate_run()`` creates one for DeepEval scoring.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

from .schemas import EvalSource, EvaluationCreate, EvaluationFinalize, MetricSpec, Record, RunMetric, Score
from .transport import Transport
from .versioning import content_hash

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger("llmbenchmark")

MetricsArg = Mapping[str, Mapping[str, Any] | None] | Sequence[MetricSpec | Mapping[str, Any] | str] | None


class Evaluation:
    def __init__(
        self,
        transport: Transport,
        run_id: str,
        *,
        evaluator: str,
        version: str | None = None,
        metrics: MetricsArg = None,
        judge_model: str | None = None,
        supersedes: str | None = None,
        source: EvalSource = "external",
        before_send: Callable[[], Any] | None = None,
        batch_size: int = 50,
        upload: bool = True,
    ) -> None:
        if not evaluator:
            raise ValueError("an evaluation needs an evaluator name")
        specs = _metric_specs(metrics)
        config = {
            "evaluator": evaluator,
            "evaluator_version": version,
            "judge_model": judge_model,
            "metrics": [s.model_dump() for s in specs],
        }
        self.info = EvaluationCreate(
            run_id=run_id,
            source=source,
            evaluator=evaluator,
            evaluator_version=version,
            judge_model=judge_model,
            metrics=specs,
            config_hash=content_hash(config),
            supersedes=supersedes,
        )
        self._transport = transport
        self._before_send = before_send
        self._batch_size = batch_size
        self._upload = upload
        self._thresholds = {s.name: s.threshold for s in specs}
        self.scores: list[Score] = []
        self.metrics: list[RunMetric] = []
        self._score_buffer: list[Score] = []
        self._metric_buffer: list[RunMetric] = []
        self._lock = threading.Lock()
        self._started = False
        self.finalized = False
        self.uploaded = True  # becomes False if any send fell back to the file

    @property
    def id(self) -> str:
        return self.info.id

    @property
    def config_hash(self) -> str:
        return self.info.config_hash

    def __repr__(self) -> str:
        state = "finalized" if self.finalized else "open"
        return (f"<llmbenchmark.Evaluation id={self.id} evaluator={self.info.evaluator!r} "
                f"scores={len(self.scores)} {state}>")

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> Evaluation:
        if not self._started:
            self._started = True
            self._prepare()
            if self._upload:
                self._track(self._transport.create_evaluation(self.info))
        return self

    def __enter__(self) -> Evaluation:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finalize()
        else:
            # Keep what was scored, but don't freeze a half-done evaluation.
            self.flush()
            logger.warning("llmbenchmark: evaluation %s left open because of %s", self.id, exc_type.__name__)

    def finalize(self) -> None:
        """Send everything and freeze the evaluation (no more scores)."""
        if self.finalized:
            return
        self.start()
        self.flush()
        self.finalized = True
        if self._upload:
            manifest = EvaluationFinalize(score_ids=[s.id for s in self.scores],
                                          metric_ids=[m.id for m in self.metrics])
            self._track(self._transport.finalize_evaluation(self.info.run_id, self.id, manifest))

    def flush(self) -> None:
        self._prepare()
        with self._lock:
            scores, self._score_buffer = self._score_buffer, []
            metrics, self._metric_buffer = self._metric_buffer, []
        if not self._upload:
            return
        if scores:
            self._track(self._transport.send_scores(self.info.run_id, self.id, scores))
        if metrics:
            self._track(self._transport.send_metrics(self.info.run_id, self.id, metrics))

    # ---- recording -------------------------------------------------------

    def score(
        self,
        record: Record | str,
        metric: str,
        score: float | bool | None = None,
        *,
        passed: bool | None = None,
        threshold: float | None = None,
        reason: str | None = None,
        error: str | None = None,
    ) -> Score:
        """Record one metric's verdict on one record.

        ``passed`` defaults to ``score >= threshold`` when both are known
        (the threshold comes from the evaluation's metric config if not given).
        A boolean score means pass/fail and is stored as 1.0 / 0.0.
        """
        if self.finalized:
            raise RuntimeError(f"evaluation {self.id} is finalized; create a new evaluation to re-score")
        self.start()
        record_id = record.id if isinstance(record, Record) else str(record)
        if _is_bool(score):  # Python or numpy bool: pass/fail
            passed = bool(score) if passed is None else passed
            score = 1.0 if score else 0.0
        value = _to_float(score)
        if score is not None and value is None and error is None:
            error = f"score is not a finite number: {score!r}"
        passed = None if passed is None else bool(passed)
        threshold = _to_float(threshold) if threshold is not None else self._thresholds.get(metric)
        if passed is None and value is not None and threshold is not None:
            passed = value >= threshold
        item = Score(
            evaluation_id=self.id,
            record_id=record_id,
            metric=metric,
            score=value,
            passed=passed,
            threshold=threshold,
            reason=None if reason is None else str(reason),
            error=error,
        )
        with self._lock:
            self.scores.append(item)
            self._score_buffer.append(item)
            full = len(self._score_buffer) >= self._batch_size
        if full:
            self.flush()
        return item

    def metric(self, name: str, value: float, *, n: int | None = None, details: dict[str, Any] | None = None) -> RunMetric:
        """Record a run-level number (e.g. accuracy over the whole dataset)."""
        if self.finalized:
            raise RuntimeError(f"evaluation {self.id} is finalized; create a new evaluation to re-score")
        self.start()
        number = _to_float(value)
        if number is None:
            raise ValueError(f"metric {name!r} needs a finite numeric value, got {value!r}")
        from .run import _jsonable  # local: run imports this module

        item = RunMetric(evaluation_id=self.id, name=name, value=number, n=n, details=_jsonable(details or {}))
        with self._lock:
            self.metrics.append(item)
            self._metric_buffer.append(item)
        return item

    # ---- analysis ----------------------------------------------------------

    def to_dataframe(self):
        """One row per score."""
        import pandas as pd

        return pd.DataFrame([s.model_dump() for s in self.scores])

    def summary(self):
        """Mean score, pass rate, count and errors per metric (from this session's scores)."""
        return summarize(self.scores)

    # ---- internals ---------------------------------------------------------

    def _prepare(self) -> None:
        # Scores reference records, so the run's buffered records go first.
        if self._before_send is not None:
            try:
                self._before_send()
            except Exception as exc:  # pragma: no cover - flush never raises
                logger.warning("llmbenchmark: could not flush records: %s", exc)

    def _track(self, ok: bool) -> None:
        self.uploaded = self.uploaded and ok


def resolve_supersedes(value: Any, run_id: str, transport: Transport) -> str | None:
    """The id of the evaluation ``value`` replaces, checked to belong to ``run_id``.

    Accepts an Evaluation, an EvalReport, or an id. A mismatch raises here
    (explicit misuse) instead of failing later on the server, which would send
    the whole new evaluation to the fallback file.
    """
    from .schemas import is_valid_id
    from .transport import APIError

    if value is None:
        return None
    if isinstance(value, Evaluation):
        candidate, owner = value.id, value.info.run_id
    else:
        candidate = getattr(value, "evaluation_id", None) or getattr(value, "id", None) or value
        owner = getattr(value, "run_id", None)
    if not is_valid_id(candidate):
        raise ValueError(f"not a valid evaluation id: {candidate!r}")
    if owner is None:  # a bare id: ask the backend which run it belongs to
        try:
            owner = transport.get(f"/v1/evaluations/{candidate}")["run_id"]
        except APIError as exc:
            if exc.status_code == 404:
                raise ValueError(f"supersedes: evaluation {candidate} not found") from None
            raise
        except Exception as exc:  # offline: can't check now; the server still will
            logger.warning("llmbenchmark: could not verify supersedes=%s (%s)", candidate, exc)
            return candidate
    if owner != run_id:
        raise ValueError(f"supersedes: evaluation {candidate} belongs to run {owner}, not {run_id}")
    return candidate


def summarize(scores: Sequence[Score]):
    import pandas as pd

    df = pd.DataFrame([s.model_dump() for s in scores])
    if df.empty:
        return df
    return df.groupby("metric").agg(
        mean_score=("score", "mean"),
        pass_rate=("passed", lambda s: s.dropna().astype(float).mean() if s.notna().any() else float("nan")),
        n=("record_id", "count"),
        errors=("error", lambda s: int(s.notna().sum())),
    )


def _metric_specs(metrics: MetricsArg) -> list[MetricSpec]:
    if metrics is None:
        return []
    if isinstance(metrics, Mapping):
        return [_spec({"name": name, **dict(cfg or {})}) for name, cfg in metrics.items()]
    specs = []
    for m in metrics:
        if isinstance(m, MetricSpec):
            specs.append(m)
        elif isinstance(m, str):
            specs.append(MetricSpec(name=m))
        else:
            specs.append(_spec(dict(m)))
    return specs


def _spec(data: dict[str, Any]) -> MetricSpec:
    # Unknown keys (e.g. "higher_is_better") are kept in params, not dropped.
    known = {"name", "threshold", "criteria", "params"}
    params = {**(data.get("params") or {}), **{k: v for k, v in data.items() if k not in known}}
    fields = {k: v for k, v in data.items() if k in known and k != "params"}
    if fields.get("threshold") is not None:
        threshold = _to_float(fields["threshold"])
        if threshold is None:
            raise ValueError(f"metric {data.get('name')!r} threshold must be a finite number")
        fields["threshold"] = threshold
    return MetricSpec(**fields, params=params)


def _to_float(value: Any) -> float | None:
    """A finite float, or None. NaN/inf can't be sent as JSON."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_bool(value: Any) -> bool:
    kind = type(value)
    return isinstance(value, bool) or (kind.__module__ == "numpy" and kind.__name__ in ("bool", "bool_"))
