"""A Run groups the records of one experiment (one model + params + prompt
version over one dataset version).

Recording never raises into the notebook (AGENTS.md invariant 1): values are
sanitized at log time, sends retry and fall back to a local file, and
``track()``/``trace()`` record a minimal error record if building the real
one fails.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from . import providers
from .assets import Dataset, Prompt, _rows, _to_item, is_missing
from .config import Config
from .evaluation import Evaluation, MetricsArg, resolve_supersedes
from .schemas import (
    DatasetItem,
    Message,
    Record,
    RecordKind,
    RunCreate,
    RunMode,
    RunStatus,
    RunUpdate,
    Span,
    ToolCall,
    limit_depth,
    new_id,
    utcnow,
)
from .tracing import SpanHandle, _current_span, _current_trace, _TraceState, span
from .transport import Transport
from .versioning import capture_lineage

logger = logging.getLogger("llmbenchmark")

# Fields managed mode expects, so evaluations are reproducible.
_MANAGED_REQUIRED = ("model", "generation_params", "prompt")


@dataclass
class TrackedCall:
    """Handle yielded by ``Run.track()``. Set ``response`` (or ``output``) on it."""

    response: Any = None
    output: str | None = None
    tools_called: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    record: Record | None = None  # filled in when the block exits


@dataclass
class TraceHandle:
    """Handle yielded by ``Run.trace()``. Set ``output`` to the agent's final answer."""

    trace_id: str
    record_id: str
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    record: Record | None = None
    spans: list[Span] = field(default_factory=list)


class Run:
    def __init__(
        self,
        transport: Transport,
        config: Config,
        *,
        name: str | None = None,
        project: str | None = None,
        mode: RunMode = "managed",
        model: str | None = None,
        provider: str | None = None,
        generation_params: dict[str, Any] | None = None,
        prompt: Prompt | None = None,
        dataset: Dataset | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        from . import __version__

        if prompt is not None and not isinstance(prompt, Prompt):
            raise TypeError("prompt must be an llmbenchmark.Prompt")
        if dataset is not None and not isinstance(dataset, Dataset):
            raise TypeError("dataset must be an llmbenchmark.Dataset")
        self._transport = transport
        self._config = config
        self.prompt = prompt
        self.dataset = dataset
        self.info = RunCreate(
            project=project or config.project,
            name=name,
            mode=mode,
            model=model,
            provider=provider,
            generation_params=_jsonable(generation_params or {}),
            prompt=prompt.ref() if prompt else None,
            dataset=dataset.ref() if dataset else None,
            tags=list(tags or []),
            lineage=capture_lineage(sdk_version=__version__, capture_git=config.capture_git),
        )
        if mode == "managed":
            missing = [f for f in _MANAGED_REQUIRED if not getattr(self.info, f)]
            if missing:
                self.info.lineage.incomplete = missing
                logger.warning(
                    "llmbenchmark: run %r is missing %s; results will be hard to reproduce. "
                    "Pass them to start_run(), or use mode='external' if you only upload results.",
                    name or self.id, ", ".join(missing),
                )
        self.records: list[Record] = []  # everything logged, kept for evaluation
        self.spans: dict[str, list[Span]] = {}  # trace_id -> spans
        self.evaluations: list[Evaluation] = []
        self._buffer: list[Record] = []
        self._lock = threading.Lock()  # notebooks often fan out with threads
        # Held from taking the buffer until it is sent (or saved to the fallback
        # file): anything that depends on earlier records (a trace's spans, an
        # evaluation's scores) flushes first and so waits for them.
        self._send_lock = threading.RLock()
        self._finished = False

    @property
    def id(self) -> str:
        return self.info.id

    def __repr__(self) -> str:
        return (f"<llmbenchmark.Run id={self.id} name={self.info.name!r} "
                f"mode={self.info.mode} records={len(self.records)}>")

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> Run:
        if self.prompt is not None:
            self._transport.send_prompt(self.prompt.to_wire(), self.id)
        if self.dataset is not None:
            self._transport.send_dataset(self.dataset.to_wire(), self.dataset.items, self.id)
        self._transport.create_run(self.info)
        return self

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish("failed" if exc_type else "completed")

    def finish(self, status: RunStatus = "completed") -> None:
        """Send remaining records and close the run. A finished run is immutable."""
        if self._finished:
            return
        self._finished = True
        self.flush()
        self.flush_evaluations()
        self.info.status = status
        self.info.ended_at = utcnow()
        # The manifest lets the server accept late retries of exactly these ids.
        manifest = RunUpdate(
            status=status,
            record_ids=[r.id for r in self.records],
            span_ids=[s.id for spans in self.spans.values() for s in spans],
        )
        self._transport.update_run(self.id, manifest)

    def flush_evaluations(self) -> None:
        """Send buffered scores of evaluations that were never finalized."""
        for ev in list(self.evaluations):
            if not ev.finalized:
                logger.warning("llmbenchmark: evaluation %s was never finalized; sending its scores, "
                               "but it stays open (call ev.finalize() or use `with run.evaluation(...)`)", ev.id)
                ev.flush()

    def flush(self) -> None:
        with self._send_lock:
            with self._lock:
                batch, self._buffer = self._buffer, []
            self._transport.send_records(self.id, batch)

    # ---- recording --------------------------------------------------------

    def log(
        self,
        *,
        input: Any = None,
        output: Any = None,
        response: Any = None,
        system: str | None = None,
        messages: Sequence[Message | Mapping[str, Any]] | None = None,
        item: DatasetItem | Mapping[str, Any] | None = None,
        expected_output: Any = None,
        expected_tools: Sequence[str] | None = None,
        tools_called: Sequence[ToolCall | Mapping[str, Any]] | None = None,
        context: Any = None,
        retrieval_context: Any = None,
        model: str | None = None,
        provider: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        sample_index: int = 0,
        kind: RecordKind | None = None,
        trace_id: str | None = None,
        id: str | None = None,
    ) -> Record:
        """Record one example.

        Give the input as one of:
        - ``input="..."`` (optionally with ``system="..."``): a single prompt,
        - ``messages=[{"role": ..., "content": ...}, ...]``: a conversation,
        - ``item=<DatasetItem>``: a dataset item (fills input, system and
          expected values; explicit arguments still win).

        Pass the raw provider ``response`` to fill model, output, token counts
        and tool calls automatically. Explicit values win over extracted ones.

        Raises RuntimeError on a finished run: its records are frozen, so the
        record could never be stored (start a new run instead).
        """
        if self._finished:
            raise RuntimeError(f"run {self.id} is finished; its records are immutable. Start a new run.")
        ds_item = _to_item(item) if item is not None and not isinstance(item, DatasetItem) else item
        msgs = _messages(messages if messages is not None else (ds_item.messages if ds_item else None))
        if system is None:
            system = ds_item.system if ds_item and ds_item.system else _system_from(msgs)
        system = None if system is None else _content_text(system)  # e.g. Anthropic content blocks
        if input is None:
            input = ds_item.input if ds_item else _last_user(msgs)
        if input is None:
            raise ValueError("log() needs input=..., messages=[...] or item=...")

        extracted = providers.extract(response)
        explicit = dict(model=model, provider=provider, output=output,
                        input_tokens=input_tokens, output_tokens=output_tokens)
        fields_ = {**extracted, **{k: v for k, v in explicit.items() if v is not None}}
        for key in ("input_tokens", "output_tokens"):
            fields_[key] = _int_or_none(fields_.get(key))
        for key in ("model", "provider"):
            if fields_.get(key) is not None:
                fields_[key] = str(fields_[key])
        tools = tools_called if tools_called is not None else fields_.pop("tools_called", None)
        fields_.pop("tools_called", None)
        fields_["metadata"] = _jsonable({**extracted.get("metadata", {}), **(metadata or {})})
        fields_["output"] = _opt_text(fields_.get("output"))
        # Fall back to the run's model/provider when the response didn't say.
        if fields_.get("model") is None:
            fields_["model"] = self.info.model
        if fields_.get("provider") is None:
            fields_["provider"] = self.info.provider

        if kind is None:
            if trace_id is not None:
                kind = "agent"
            elif msgs and any(m.role == "assistant" for m in msgs):
                kind = "conversation"
            else:
                kind = "single"

        record = Record(
            id=id or new_id(),
            run_id=self.id,
            kind=kind,
            dataset_item_id=ds_item.id if ds_item else None,
            sample_index=_int_or_none(sample_index) or 0,
            system=system,
            input=_to_text(input),
            messages=msgs,
            expected_output=_opt_text(expected_output if expected_output is not None
                                      else (ds_item.expected_output if ds_item else None)),
            expected_tools=_str_list(expected_tools) if expected_tools is not None
            else (ds_item.expected_tools if ds_item else None),
            tools_called=_tool_calls(tools),
            context=_opt_texts(context if context is not None else (ds_item.context if ds_item else None)),
            retrieval_context=_opt_texts(retrieval_context if retrieval_context is not None
                                         else (ds_item.retrieval_context if ds_item else None)),
            trace_id=trace_id,
            latency_ms=_float_or_none(latency_ms),
            error=None if error is None else str(error),
            tags=_str_list(tags) or [],
            **fields_,
        )
        with self._lock:
            self.records.append(record)
            self._buffer.append(record)
            full = len(self._buffer) >= self._config.batch_size
        if full:
            self.flush()
        return record

    @contextmanager
    def track(
        self,
        *,
        input: Any = None,
        system: str | None = None,
        messages: Sequence[Message | Mapping[str, Any]] | None = None,
        item: DatasetItem | Mapping[str, Any] | None = None,
        expected_output: Any = None,
        expected_tools: Sequence[str] | None = None,
        context: Any = None,
        retrieval_context: Any = None,
        model: str | None = None,
        provider: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        sample_index: int = 0,
        raise_errors: bool = True,
    ) -> Iterator[TrackedCall]:
        """Time an LLM call and record it when the ``with`` block exits.

        Exceptions are recorded in ``error``; they are re-raised unless
        ``raise_errors=False`` (handy for long benchmark loops). Interrupts
        (``KeyboardInterrupt``) are recorded and always re-raised. Recording
        itself never raises.
        """
        handle = TrackedCall(metadata=dict(metadata or {}))
        error: str | None = None
        start = time.perf_counter()
        try:
            yield handle
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if raise_errors or not isinstance(exc, Exception):
                raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            handle.record = self._safe_log(
                dict(
                    input=input, system=system, messages=messages, item=item,
                    output=handle.output, response=handle.response, tools_called=handle.tools_called,
                    expected_output=expected_output, expected_tools=expected_tools,
                    context=context, retrieval_context=retrieval_context,
                    model=model, provider=provider, metadata=handle.metadata, tags=tags,
                    sample_index=sample_index, latency_ms=latency_ms, error=error,
                ),
                error,
            )

    @contextmanager
    def trace(
        self,
        *,
        input: Any = None,
        system: str | None = None,
        item: DatasetItem | Mapping[str, Any] | None = None,
        expected_output: Any = None,
        expected_tools: Sequence[str] | None = None,
        name: str = "agent",
        metadata: dict[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        sample_index: int = 0,
        raise_errors: bool = True,
    ) -> Iterator[TraceHandle]:
        """Record one agent task as a trace of spans plus one ``kind="agent"`` record.

        Inside the block, ``llmbenchmark.span(...)`` and ``@llmbenchmark.observe``
        add steps to this trace. Tool spans become the record's ``tools_called``.
        """
        handle = TraceHandle(trace_id=new_id(), record_id=new_id(), metadata=dict(metadata or {}))
        state = _TraceState(self.id, handle.trace_id, handle.record_id)
        trace_token = _current_trace.set(state)
        span_token = _current_span.set(None)
        root_input = input if input is not None else (_to_item(item).input if item is not None else None)
        root: SpanHandle | None = None
        error: str | None = None
        start = time.perf_counter()
        try:
            with span(name, kind="agent", input=root_input) as root:
                yield handle
                root.output = handle.output
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if raise_errors or not isinstance(exc, Exception):
                raise
        finally:
            _current_span.reset(span_token)
            _current_trace.reset(trace_token)
            latency_ms = (time.perf_counter() - start) * 1000
            spans = state.close()  # spans finishing later are dropped with a warning
            handle.spans = spans
            self.spans[handle.trace_id] = spans
            llm = [s for s in spans if s.kind == "llm"]
            tools = [
                {"name": s.name, "arguments": s.input if isinstance(s.input, dict) else {"input": s.input},
                 "output": s.output}
                for s in sorted(spans, key=lambda s: s.started_at) if s.kind == "tool"
            ]
            handle.record = self._safe_log(
                dict(
                    id=handle.record_id, trace_id=handle.trace_id, kind="agent",
                    input=input, system=system, item=item, output=handle.output,
                    expected_output=expected_output, expected_tools=expected_tools,
                    tools_called=tools,  # [] (not None) when the agent used no tools
                    model=next((s.model for s in llm if s.model), None),
                    input_tokens=_sum([s.input_tokens for s in llm]),
                    output_tokens=_sum([s.output_tokens for s in llm]),
                    metadata=handle.metadata, tags=tags, sample_index=sample_index,
                    latency_ms=latency_ms, error=error,
                ),
                error,
            )
            # Spans reference the record, so send the record first.
            self.flush()
            self._transport.send_spans(self.id, spans)

    # ---- evaluations --------------------------------------------------------

    def evaluation(
        self,
        evaluator: str,
        *,
        version: str | None = None,
        metrics: MetricsArg = None,
        judge_model: str | None = None,
        supersedes: str | Evaluation | None = None,
    ) -> Evaluation:
        """Start a scoring pass for results you computed yourself (see Evaluation)."""
        supersedes = resolve_supersedes(supersedes, self.id, self._transport)
        ev = Evaluation(
            self._transport, self.id, evaluator=evaluator, version=version, metrics=metrics,
            judge_model=judge_model, supersedes=supersedes, source="external",
            before_send=self.flush, batch_size=self._config.batch_size,
        )
        self.evaluations.append(ev)
        return ev.start()

    def log_results(
        self,
        rows: Any,
        *,
        evaluator: str,
        scores: Mapping[str, str] | Sequence[str],
        version: str | None = None,
        input_col: str = "input",
        output_col: str = "output",
        expected_col: str | None = "expected_output",
        system_col: str | None = None,
        item_id_col: str | None = None,
        reason_cols: Mapping[str, str] | None = None,
        metrics: MetricsArg = None,
        aggregates: Mapping[str, float] | None = None,
    ) -> Evaluation:
        """Upload results scored elsewhere: one row per example (list of dicts or DataFrame).

        ``scores`` maps metric name -> column (or lists columns named after
        their metrics). Returns the finalized Evaluation.
        """
        if self._finished:
            raise RuntimeError(f"run {self.id} is finished; call log_results() inside the run's `with` block.")
        score_cols = dict(scores) if isinstance(scores, Mapping) else {c: c for c in scores}
        reason_cols = dict(reason_cols or {})
        ev = self.evaluation(evaluator, version=version, metrics=metrics or list(score_cols))
        rows = _rows(rows)
        for row in rows:
            item = None
            if item_id_col and row.get(item_id_col) is not None:
                item = DatasetItem(id=str(row[item_id_col]), input=_to_text(row[input_col]))
            record = self.log(
                input=row[input_col], item=item, output=row.get(output_col),
                expected_output=row.get(expected_col) if expected_col else None,
                system=row.get(system_col) if system_col else None,
            )
            for metric, col in score_cols.items():
                reason_col = reason_cols.get(metric)
                ev.score(record, metric, row.get(col), reason=row.get(reason_col) if reason_col else None)
        for name, value in (aggregates or {}).items():
            ev.metric(name, value, n=len(rows))
        ev.finalize()
        return ev

    # ---- helpers --------------------------------------------------------

    def to_dataframe(self):
        """All logged records as a pandas DataFrame (requires pandas)."""
        import pandas as pd

        return pd.DataFrame([r.model_dump() for r in self.records])

    def _safe_log(self, kwargs: dict[str, Any], error: str | None) -> Record | None:
        """log(), but recording problems become an error record instead of an exception."""
        if self._finished:
            logger.error("llmbenchmark: run %s is already finished; this call was not recorded", self.id)
            return None
        try:
            return self.log(**kwargs)
        except Exception as exc:
            logger.warning("llmbenchmark: could not record call: %s", exc, exc_info=True)
            problem = f"llmbenchmark could not record this call: {type(exc).__name__}: {exc}"
            base = {k: kwargs.get(k) for k in ("id", "trace_id", "latency_ms", "kind")}
            base = {k: v for k, v in base.items() if v is not None}
            base["input"] = _to_text(kwargs.get("input") if kwargs.get("input") is not None else "")
            base["error"] = f"{error}; {problem}" if error else problem
            # Keep what we can (output, model, tokens); drop the rest if even that fails.
            richer = {k: kwargs.get(k) for k in ("output", "model", "provider", "input_tokens", "output_tokens")}
            for attempt in ({**base, **{k: v for k, v in richer.items() if v is not None}}, base):
                try:
                    return self.log(**attempt)
                except Exception:
                    continue
            logger.error("llmbenchmark: could not record call at all")  # pragma: no cover
            return None


# ---- value helpers -----------------------------------------------------------


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _opt_text(value: Any) -> str | None:
    return None if is_missing(value) else _to_text(value)


def _opt_texts(values: Any) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return [values]
    return [_to_text(v) for v in values if not is_missing(v)]


def _jsonable(value: dict[str, Any]) -> dict[str, Any]:
    """A JSON-safe copy: anything json can't encode becomes its str(), and
    nesting deeper than MAX_JSON_DEPTH is cut (the server refuses it)."""
    try:
        return json.loads(json.dumps(limit_depth(value), ensure_ascii=False, default=str))
    except (TypeError, ValueError, RecursionError):
        return {str(k): str(v)[:1000] for k, v in value.items()}


def _jsonable_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        return json.loads(json.dumps(limit_depth(value), ensure_ascii=False, default=str))
    except (TypeError, ValueError, RecursionError):
        return str(value)[:1000]


def _content_text(content: Any) -> str:
    """Message content as text: provider content-block lists are joined."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") in ("text", "input_text", "output_text"):
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return _to_text(content)


_ROLE_ALIASES = {"developer": "system", "human": "user", "ai": "assistant", "model": "assistant"}


def _messages(messages: Sequence[Message | Mapping[str, Any]] | None) -> list[Message] | None:
    if messages is None:
        return None
    out = []
    for m in messages:
        if isinstance(m, Message):
            out.append(m)
            continue
        data = dict(m)
        role = _ROLE_ALIASES.get(str(data.get("role")), data.get("role"))
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        out.append(Message(
            role=role,
            content=_content_text(data.get("content")),
            name=data.get("name"),
            tool_calls=_tool_calls(data.get("tool_calls")),
        ))
    return out


def _system_from(messages: list[Message] | None) -> str | None:
    if not messages:
        return None
    parts = [m.content for m in messages if m.role == "system"]
    return "\n".join(parts) if parts else None


def _last_user(messages: list[Message] | None) -> str | None:
    for m in reversed(messages or []):
        if m.role == "user":
            return m.content
    return None


def _tool_calls(raw: Any) -> list[ToolCall] | None:
    if raw is None:
        return None  # unknown; [] means "called no tools", which metrics can score
    out = []
    for t in raw:
        if isinstance(t, ToolCall):
            out.append(t)
            continue
        data = dict(t) if isinstance(t, Mapping) else {"name": str(t)}
        fn = data.get("function")
        if isinstance(fn, Mapping):  # OpenAI message.tool_calls shape
            data = {"name": fn.get("name"), "arguments": providers._json_args(fn.get("arguments")), "id": data.get("id")}
        args = data.get("arguments", data.get("input", {}))
        out.append(ToolCall(
            name=str(data.get("name") or "unknown"),
            arguments=_jsonable(args) if isinstance(args, Mapping) else {"input": _jsonable_value(args)},
            output=_jsonable_value(data.get("output")),
            id=data.get("id"),
        ))
    return out


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number == number and number not in (float("inf"), float("-inf")) else None


def _float_or_none(value: Any) -> float | None:
    try:
        number = None if value is None else float(value)
    except (TypeError, ValueError):
        return None
    return number if number is not None and number == number and abs(number) != float("inf") else None


def _str_list(values: Any) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return [values]
    return [str(v) for v in values if v is not None]


def _sum(values: list[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None
