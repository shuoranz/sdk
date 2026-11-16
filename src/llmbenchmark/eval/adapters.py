"""Convert llmbenchmark records (and agent spans) into DeepEval test cases.

- single / agent record -> ``LLMTestCase`` (agent: plus tools_called and the
  span tree as DeepEval's trace dict, which trace-based metrics such as
  TaskCompletion, StepEfficiency and PlanAdherence read)
- conversation record   -> ``ConversationalTestCase`` for multi-turn metrics,
  and also ``LLMTestCase`` (last user turn -> output) for single-turn metrics

Also: which fields each metric needs (``missing_fields``), so the runner can
skip or refuse incomplete records before spending judge tokens.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
from deepeval.test_case import ToolCall as DeepEvalToolCall

from ..schemas import Record, Span

try:  # DeepEval 4.x
    from deepeval.metrics.base_metric import BaseConversationalMetric
except ImportError:  # pragma: no cover - older layout
    from deepeval.metrics import BaseConversationalMetric

# Our span kinds -> DeepEval's span "type" values.
_SPAN_TYPES = {"agent": "agent", "llm": "llm", "tool": "tool", "retrieval": "retriever",
               "handoff": "base", "other": "base"}

# DeepEval test-case param name -> how to check the record has it.
_FIELD_CHECKS = {
    "input": lambda r: bool(r.input),
    "actual_output": lambda r: r.output is not None and r.output != "",
    "expected_output": lambda r: r.expected_output is not None,
    "context": lambda r: bool(r.context),
    "retrieval_context": lambda r: bool(r.retrieval_context),
    "tools_called": lambda r: r.tools_called is not None,
    "expected_tools": lambda r: r.expected_tools is not None,
}


# ---- metric introspection ------------------------------------------------------


def is_conversational(metric: Any) -> bool:
    return isinstance(metric, BaseConversationalMetric)


def requires_trace(metric: Any) -> bool:
    return bool(getattr(metric, "requires_trace", False))


def required_fields(metric: Any) -> list[str]:
    """Test-case fields a single-turn metric reads (DeepEval param names)."""
    params = getattr(metric, "evaluation_params", None)  # G-Eval: chosen by the user
    if not isinstance(params, (list, tuple)):
        params = getattr(metric, "_required_params", None)
    if not isinstance(params, (list, tuple)):  # unannotated custom metric
        params = []
    names = [getattr(p, "value", str(p)) for p in params]
    return [n for n in names if n in _FIELD_CHECKS]


def applies_to(metric: Any, record: Record) -> bool:
    """Whether a metric can score this kind of record at all."""
    if is_conversational(metric):
        return record.kind == "conversation"
    if requires_trace(metric):
        return record.kind == "agent"
    return True


def missing_fields(metric: Any, record: Record) -> list[str]:
    if is_conversational(metric):
        return [] if record.messages else ["messages"]
    return [f for f in required_fields(metric) if not _FIELD_CHECKS[f](record)]


# ---- conversions ------------------------------------------------------------------


def to_test_case(record: Record, spans: list[Span] | None = None) -> LLMTestCase:
    case = LLMTestCase(
        name=record.id,  # lets results map back to the record
        input=record.input,
        actual_output=record.output or "",
        expected_output=record.expected_output,
        context=record.context,
        retrieval_context=record.retrieval_context,
        tools_called=[_tool(t) for t in record.tools_called] if record.tools_called is not None else None,
        expected_tools=[DeepEvalToolCall(name=n) for n in record.expected_tools]
        if record.expected_tools is not None else None,
        metadata={"model": record.model, "provider": record.provider, "kind": record.kind,
                  "system": record.system, "dataset_item_id": record.dataset_item_id},
    )
    if record.kind == "agent" and spans:
        attach_trace(case, spans)
    return case


def to_conversational_test_case(record: Record) -> ConversationalTestCase:
    turns = []
    for m in record.messages or []:
        if m.role not in ("user", "assistant"):
            continue  # system goes to chatbot_role; tool results are folded into tool calls
        turns.append(Turn(role=m.role, content=m.content or "",
                          tools_called=[_tool(t) for t in m.tool_calls] if m.tool_calls else None))
    if record.output:
        turns.append(Turn(role="assistant", content=record.output,
                          tools_called=[_tool(t) for t in record.tools_called] if record.tools_called else None))
    return ConversationalTestCase(
        name=record.id,
        turns=turns,
        chatbot_role=record.system,
        expected_outcome=record.expected_output,
        context=record.context,
        metadata={"model": record.model, "provider": record.provider, "kind": record.kind},
    )


def attach_trace(case: LLMTestCase, spans: list[Span]) -> bool:
    """Give DeepEval's trace-based metrics our span tree.

    DeepEval reads a private ``_trace_dict`` attribute (set by its own @observe
    tracing). We fill it from our spans. Pinned by a test against the installed
    DeepEval; if the attribute disappears, trace metrics fall back to
    input/output/tools_called (TaskCompletion) or are skipped by the runner.
    """
    if "_trace_dict" not in getattr(type(case), "__private_attributes__", {}):
        return False
    case._trace_dict = spans_to_trace_dict(spans)
    return True


def spans_to_trace_dict(spans: list[Span]) -> dict[str, Any]:
    """Nested ``{name, type, input, output, children: [...]}`` like DeepEval's own traces."""
    children: dict[str | None, list[Span]] = defaultdict(list)
    ids = {s.id for s in spans}
    for s in spans:
        parent = s.parent_id if s.parent_id in ids else None
        children[parent].append(s)

    def node(s: Span) -> dict[str, Any]:
        d: dict[str, Any] = {"name": s.name, "type": _SPAN_TYPES.get(s.kind, "base")}
        for key in ("input", "output", "model", "error"):
            value = getattr(s, key)
            if value is not None:
                d[key] = value
        d["children"] = [node(c) for c in sorted(children.get(s.id, []), key=lambda c: c.started_at)]
        return d

    roots = sorted(children.get(None, []), key=lambda s: s.started_at)
    if len(roots) == 1:
        return node(roots[0])
    return {"name": "trace", "type": "base", "children": [node(r) for r in roots]}


def _tool(t: Any) -> DeepEvalToolCall:
    return DeepEvalToolCall(name=t.name, input_parameters=t.arguments or None, output=t.output)
