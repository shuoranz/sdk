"""Agent tracing: record every step of an agent run as a tree of spans.

    with run.trace(input=task) as tr:              # one agent task -> one Record
        with llmbenchmark.span("planner", kind="agent"):
            with llmbenchmark.span("plan", kind="llm") as s:
                s.response = client.messages.create(...)   # model/tokens extracted
            search(query)                          # @llmbenchmark.observe(kind="tool")
        tr.output = answer

The current trace and span live in ``contextvars``, so nesting works across
function calls and asyncio tasks automatically. Plain threads start with an
empty context; wrap the function with ``llmbenchmark.propagate(fn)`` when
submitting it to a thread pool.

Like the rest of the SDK, tracing never raises into the agent: a span outside
any trace is a no-op, and a failure while recording a span is logged.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from . import providers
from .schemas import Span, SpanKind, new_id, utcnow

logger = logging.getLogger("llmbenchmark")

F = TypeVar("F", bound=Callable[..., Any])


class SpanHandle:
    """Yielded by ``span()``. Set ``output`` (or ``response`` for LLM calls)."""

    def __init__(self, name: str, kind: SpanKind, input: Any, attributes: dict[str, Any]) -> None:
        self.id = new_id()
        self.name = name
        self.kind = kind
        self.input = input
        self.output: Any = None
        self.response: Any = None  # raw provider response, for kind="llm"
        self.model: str | None = None
        self.provider: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.attributes = dict(attributes)

    def __repr__(self) -> str:
        return f"<llmbenchmark.SpanHandle {self.kind}:{self.name}>"


class _TraceState:
    """Spans collected for one trace (one agent Record)."""

    def __init__(self, run_id: str, trace_id: str, record_id: str) -> None:
        self.run_id = run_id
        self.trace_id = trace_id
        self.record_id = record_id
        self.spans: list[Span] = []
        self.closed = False
        self._lock = threading.Lock()

    def add(self, span: Span) -> bool:
        """Add a finished span; False if the trace already ended (span dropped)."""
        with self._lock:
            if self.closed:
                return False
            self.spans.append(span)
            return True

    def close(self) -> list[Span]:
        with self._lock:
            self.closed = True
            return list(self.spans)


_current_trace: contextvars.ContextVar[_TraceState | None] = contextvars.ContextVar(
    "llmbenchmark_trace", default=None
)
_current_span: contextvars.ContextVar[SpanHandle | None] = contextvars.ContextVar(
    "llmbenchmark_span", default=None
)


@contextmanager
def span(
    name: str,
    *,
    kind: SpanKind = "other",
    input: Any = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[SpanHandle]:
    """Record one step of the current trace. Exceptions are recorded and re-raised."""
    handle = SpanHandle(name, kind, input, attributes or {})
    state = _current_trace.get()
    if state is None:
        logger.debug("llmbenchmark: span %r outside a trace; not recorded", name)
        yield handle
        return
    parent = _current_span.get()
    token = _current_span.set(handle)
    started = utcnow()
    error: str | None = None
    try:
        yield handle
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _current_span.reset(token)
        try:
            if not state.add(_build_span(handle, state, parent, started, error)):
                logger.warning("llmbenchmark: span %r finished after its trace ended (e.g. a background "
                               "task outlived `run.trace()`); it was not recorded", name)
        except Exception as exc:  # recording must never break the agent
            logger.warning("llmbenchmark: could not record span %r: %s", name, exc)


def observe(name: str | None = None, *, kind: SpanKind = "other") -> Callable[[F], F]:
    """Decorator: record each call of the function as a span (sync or async).

    The span's input is the call's arguments by name; its output is the return value.
    """

    def decorate(fn: F) -> F:
        span_name = name or fn.__name__
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            signature = None

        def arguments(args: tuple, kwargs: dict) -> Any:
            if signature is None:
                return {"args": list(args), "kwargs": kwargs}
            try:
                bound = signature.bind_partial(*args, **kwargs)
            except TypeError:
                return {"args": list(args), "kwargs": kwargs}
            return {k: v for k, v in bound.arguments.items() if k != "self"}

        if inspect.isgeneratorfunction(fn):
            # The span covers the whole iteration, not just creating the generator.
            @functools.wraps(fn)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                with span(span_name, kind=kind, input=arguments(args, kwargs)) as s:
                    count = 0
                    for item in fn(*args, **kwargs):
                        count += 1
                        yield item
                    s.output = {"yielded": count}

            return gen_wrapper  # type: ignore[return-value]

        if inspect.isasyncgenfunction(fn):
            @functools.wraps(fn)
            async def agen_wrapper(*args: Any, **kwargs: Any) -> Any:
                with span(span_name, kind=kind, input=arguments(args, kwargs)) as s:
                    count = 0
                    async for item in fn(*args, **kwargs):
                        count += 1
                        yield item
                    s.output = {"yielded": count}

            return agen_wrapper  # type: ignore[return-value]

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with span(span_name, kind=kind, input=arguments(args, kwargs)) as s:
                    result = await fn(*args, **kwargs)
                    s.output = result
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name, kind=kind, input=arguments(args, kwargs)) as s:
                result = fn(*args, **kwargs)
                s.output = result
                return result

        return wrapper  # type: ignore[return-value]

    return decorate


def propagate(fn: F) -> F:
    """Carry the current trace into another thread.

        pool.submit(llmbenchmark.propagate(call_tool), args)

    Each call runs in a fresh copy of the caller's context, so concurrent
    calls never share (or fight over) one context.
    """
    context = contextvars.copy_context()

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return context.copy().run(fn, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


# ---- internals -------------------------------------------------------------


def _build_span(
    handle: SpanHandle, state: _TraceState, parent: SpanHandle | None, started, error: str | None
) -> Span:
    from .run import _jsonable_value  # local import: run imports this module

    extracted = providers.extract(handle.response) if handle.response is not None else {}
    output = handle.output if handle.output is not None else extracted.get("output")
    return Span(
        id=handle.id,
        run_id=state.run_id,
        trace_id=state.trace_id,
        record_id=state.record_id,
        parent_id=parent.id if parent is not None else None,
        kind=handle.kind,
        name=handle.name,
        input=_jsonable_value(handle.input),
        output=_jsonable_value(output),
        provider=handle.provider or extracted.get("provider"),
        model=handle.model or extracted.get("model"),
        input_tokens=_int_or_none(handle.input_tokens if handle.input_tokens is not None else extracted.get("input_tokens")),
        output_tokens=_int_or_none(handle.output_tokens if handle.output_tokens is not None else extracted.get("output_tokens")),
        started_at=started,
        ended_at=utcnow(),
        error=error,
        attributes=_jsonable_value(handle.attributes) or {},
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
