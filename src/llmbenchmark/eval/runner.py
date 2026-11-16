"""Managed evaluation: run DeepEval metrics over a run's records and store the
results as one versioned Evaluation.

    report = evaluate_run(run, metrics=[AnswerRelevancyMetric(model=judge()), correctness(model=judge())])
    report.summary()

Before spending judge tokens, every (record, metric) pair is checked:
- metrics that can't apply to a record kind are skipped (e.g. a multi-turn
  metric on a single prompt, a trace metric on a non-agent record);
- pairs missing a field the metric needs (e.g. expected_output) are recorded
  as a score with ``error`` set, or, with ``strict=True``, the call raises.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

from ..evaluation import Evaluation, resolve_supersedes, summarize
from ..run import Run
from ..schemas import MetricSpec, Record, Score, Span, is_valid_id
from .adapters import (
    applies_to,
    is_conversational,
    missing_fields,
    required_fields,
    to_conversational_test_case,
    to_test_case,
)


@dataclass
class EvalReport:
    run_id: str
    evaluation_id: str
    scores: list[Score]
    skipped_record_ids: list[str] = field(default_factory=list)  # errored / no output
    skipped: list[dict[str, Any]] = field(default_factory=list)  # (record, metric) pairs not scored
    uploaded: bool = False
    evaluation: Evaluation | None = None

    def to_dataframe(self):
        """One row per (record, metric)."""
        import pandas as pd

        return pd.DataFrame([s.model_dump() for s in self.scores])

    def summary(self):
        """Mean score, pass rate, count and errors per metric."""
        return summarize(self.scores)


def evaluate_run(
    run: Run | str,
    metrics: Sequence[Any],
    *,
    records: Sequence[Record] | None = None,
    upload: bool = True,
    strict: bool = False,
    supersedes: Any = None,
    max_concurrent: int = 10,
    print_results: bool = False,
) -> EvalReport:
    """Score a run's records with DeepEval ``metrics`` as a new, finalized Evaluation.

    Args:
        run: a ``Run`` from this session, or a run id (records and spans are
            then fetched from the backend).
        metrics: DeepEval metric instances. Choose their judge when you build
            them, e.g. ``AnswerRelevancyMetric(model=llmbenchmark.eval.judge())``.
        records: evaluate these records instead of all of the run's records.
        upload: send the evaluation and scores to the backend.
        strict: raise before judging if any (record, metric) pair is missing a
            field the metric needs, instead of recording it as an error.
        supersedes: an earlier evaluation (id, Evaluation or EvalReport) this
            one replaces, e.g. after fixing a rubric.
        max_concurrent: parallel judge calls (lower it if you hit rate limits).
    """
    from .. import __version__

    run_id, transport, all_records, spans, flush = _resolve(run)
    metrics = list(metrics)
    candidates = list(records) if records is not None else all_records
    eligible = [r for r in candidates if _evaluable(r)]
    skipped_ids = [r.id for r in candidates if not _evaluable(r)]

    plan: dict[str, list[int]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    incomplete: list[tuple[Record, int, list[str]]] = []
    for r in eligible:
        for i, m in enumerate(metrics):
            if not applies_to(m, r):
                skipped.append({"record_id": r.id, "metric": _name(m), "reason": f"not applicable to {r.kind} records"})
                continue
            missing = missing_fields(m, r)
            if missing:
                incomplete.append((r, i, missing))
            else:
                plan[r.id].append(i)

    if strict and incomplete:
        examples = "; ".join(f"{_name(metrics[i])} needs {', '.join(f)}" for _, i, f in incomplete[:3])
        raise ValueError(
            f"{len(incomplete)} (record, metric) pair(s) are missing required fields ({examples}). "
            "Add the fields (e.g. expected_output) or pass strict=False to record them as errors."
        )

    ev = Evaluation(
        transport, run_id,
        evaluator="llmbenchmark", version=__version__,
        metrics=[_spec(m) for m in metrics], judge_model=_judge_name(metrics),
        supersedes=resolve_supersedes(supersedes, run_id, transport), source="llmbenchmark",
        before_send=flush, batch_size=transport.config.batch_size, upload=upload,
    ).start()

    for record, i, missing in incomplete:
        ev.score(record, _name(metrics[i]), error=f"missing required field(s): {', '.join(missing)}")
        skipped.append({"record_id": record.id, "metric": _name(metrics[i]),
                        "reason": f"missing {', '.join(missing)}"})

    # One DeepEval call per (test-case type, metric subset), so every record
    # is scored only by the metrics that apply to it.
    groups: dict[tuple[bool, tuple[int, ...]], list[Record]] = defaultdict(list)
    for r in eligible:
        indices = plan.get(r.id, [])
        single = tuple(i for i in indices if not is_conversational(metrics[i]))
        multi = tuple(i for i in indices if is_conversational(metrics[i]))
        if single:
            groups[(False, single)].append(r)
        if multi:
            groups[(True, multi)].append(r)

    run_configs = dict(
        async_config=_config(AsyncConfig, max_concurrent=max_concurrent),
        display_config=_config(DisplayConfig, print_results=print_results, show_indicator=True,
                               inspect_after_run=False),
        # One failing metric/test-case pair is recorded, not fatal.
        error_config=_config(ErrorConfig, ignore_errors=True),
    )
    for (conversational, indices), group in groups.items():
        cases = [to_conversational_test_case(r) if conversational else to_test_case(r, spans.get(r.trace_id or ""))
                 for r in group]
        result = _run_deepeval(test_cases=cases, metrics=[metrics[i] for i in indices], **run_configs)
        by_id = {r.id: r for r in group}
        for k, test_result in enumerate(result.test_results):
            record = by_id.get(test_result.name)
            if record is None:  # fall back to position if the name didn't round-trip
                idx = test_result.index if getattr(test_result, "index", None) is not None else k
                record = group[idx]
            for md in test_result.metrics_data or []:
                # DeepEval reports success=False for a metric that errored; an error
                # is not a verdict, so keep it out of pass rates (like missing fields).
                passed = None if md.error else md.success
                ev.score(record, md.name, md.score, passed=passed, threshold=md.threshold,
                         reason=md.reason, error=md.error)

    ev.finalize()
    return EvalReport(
        run_id=run_id, evaluation_id=ev.id, scores=list(ev.scores), skipped_record_ids=skipped_ids,
        skipped=skipped, uploaded=upload and ev.uploaded, evaluation=ev,
    )


# ---- helpers ---------------------------------------------------------------------


def _evaluable(r: Record) -> bool:
    if r.error:
        return False
    return bool(r.output) or (r.kind == "conversation" and bool(r.messages))


def _resolve(run: Run | str) -> tuple[str, Any, list[Record], dict[str, list[Span]], Callable[[], Any] | None]:
    if isinstance(run, Run):
        return run.id, run._transport, list(run.records), dict(run.spans), run.flush
    if not is_valid_id(run):
        raise ValueError(f"not a valid run id: {run!r}")
    import llmbenchmark

    llmbenchmark._ensure_init()
    transport = llmbenchmark._transport
    records = transport.get_records(run)
    spans: dict[str, list[Span]] = defaultdict(list)
    if any(r.kind == "agent" for r in records):
        for raw in transport.get_all(f"/v1/runs/{run}/spans"):
            span = Span.model_validate(raw)
            spans[span.trace_id].append(span)
    return run, transport, records, dict(spans), None


def _name(metric: Any) -> str:
    try:
        return str(metric.__name__)
    except AttributeError:
        return type(metric).__name__


def _spec(metric: Any) -> MetricSpec:
    criteria = getattr(metric, "criteria", None)
    steps = getattr(metric, "evaluation_steps", None)
    params: dict[str, Any] = {"class": type(metric).__name__}
    if is_conversational(metric):
        params["conversational"] = True
    else:
        params["fields"] = required_fields(metric)
    if isinstance(steps, list):
        params["evaluation_steps"] = [str(s) for s in steps]
    if getattr(metric, "strict_mode", False):
        params["strict_mode"] = True
    threshold = getattr(metric, "threshold", None)
    return MetricSpec(
        name=_name(metric),
        threshold=float(threshold) if isinstance(threshold, (int, float)) else None,
        criteria=criteria if isinstance(criteria, str) else None,
        params=params,
    )


def _judge_name(metrics: Sequence[Any]) -> str | None:
    names = sorted({m.evaluation_model for m in metrics if isinstance(getattr(m, "evaluation_model", None), str)})
    return ",".join(names) or None


def _run_deepeval(**kwargs):
    """Call deepeval.evaluate, staying clear of an already-running event loop.

    In Jupyter an asyncio loop is already running in the main thread. DeepEval
    then patches it with nest_asyncio and nests into it, which breaks on
    Python 3.12+ ("Timeout should be used inside a task"). A worker thread has
    no running loop, so DeepEval creates a clean one of its own there.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # plain script / pytest: no loop running
        return evaluate(**kwargs)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="llmbenchmark-eval") as pool:
        return pool.submit(evaluate, **kwargs).result()


def _config(cls, **kwargs):
    """Build a DeepEval config, dropping options this DeepEval version lacks."""
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in kwargs.items() if k in known})
