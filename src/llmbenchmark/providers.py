"""Pull model/output/token info out of provider response objects.

Duck-typed on purpose: llmbenchmark never imports `anthropic` or `openai`, so the
core SDK works no matter which provider packages (or versions) the notebook
has installed. Plain dicts (e.g. ``response.model_dump()``) also work.
"""

from __future__ import annotations

import json
from typing import Any


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def extract(response: Any) -> dict[str, Any]:
    """Return a dict of CallRecord fields found on ``response`` (may be empty)."""
    if response is None:
        return {}
    kind = _get(response, "type")
    obj = _get(response, "object")

    if kind == "message":  # Anthropic Messages API
        return _anthropic(response)
    if obj == "chat.completion":  # OpenAI Chat Completions
        return _openai_chat(response)
    if obj == "response":  # OpenAI Responses API
        return _openai_responses(response)

    # Fallback heuristics for wrappers/mocks that lack the type marker.
    if _get(response, "choices") is not None:
        return _openai_chat(response)
    if _get(response, "content") is not None and _get(_get(response, "usage"), "input_tokens") is not None:
        return _anthropic(response)
    return {}


def _anthropic(r: Any) -> dict[str, Any]:
    # content is a list of blocks (text, thinking, tool_use, ...): keep only text.
    text = "".join(
        _get(b, "text") or "" for b in (_get(r, "content") or []) if _get(b, "type") == "text"
    )
    tools = [
        {"name": _get(b, "name"), "arguments": _get(b, "input") or {}, "id": _get(b, "id")}
        for b in (_get(r, "content") or [])
        if _get(b, "type") == "tool_use"
    ]
    usage = _get(r, "usage")
    return _clean(
        provider="anthropic",
        model=_get(r, "model"),
        output=text,
        tools_called=tools or None,
        input_tokens=_get(usage, "input_tokens"),
        output_tokens=_get(usage, "output_tokens"),
        metadata={"stop_reason": _get(r, "stop_reason"), "response_id": _get(r, "id")},
    )


def _openai_chat(r: Any) -> dict[str, Any]:
    choices = _get(r, "choices") or []
    first = choices[0] if choices else None
    message = _get(first, "message")
    tools = [
        {"name": _get(_get(tc, "function"), "name"),
         "arguments": _json_args(_get(_get(tc, "function"), "arguments")),
         "id": _get(tc, "id")}
        for tc in (_get(message, "tool_calls") or [])
    ]
    usage = _get(r, "usage")
    return _clean(
        provider="openai",
        model=_get(r, "model"),
        output=_get(message, "content"),
        tools_called=tools or None,
        input_tokens=_get(usage, "prompt_tokens"),
        output_tokens=_get(usage, "completion_tokens"),
        metadata={"finish_reason": _get(first, "finish_reason"), "response_id": _get(r, "id")},
    )


def _openai_responses(r: Any) -> dict[str, Any]:
    text = _get(r, "output_text")
    if text is None:  # dict form has no convenience property
        parts = []
        for item in _get(r, "output") or []:
            for c in _get(item, "content") or []:
                if _get(c, "type") == "output_text":
                    parts.append(_get(c, "text") or "")
        text = "".join(parts)
    usage = _get(r, "usage")
    return _clean(
        provider="openai",
        model=_get(r, "model"),
        output=text,
        input_tokens=_get(usage, "input_tokens"),
        output_tokens=_get(usage, "output_tokens"),
        metadata={"status": _get(r, "status"), "response_id": _get(r, "id")},
    )


def _json_args(raw: Any) -> dict[str, Any]:
    """OpenAI sends tool arguments as a JSON string."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {"raw": raw}
    return value if isinstance(value, dict) else {"value": value}


def _clean(**fields: Any) -> dict[str, Any]:
    fields["metadata"] = {k: v for k, v in fields["metadata"].items() if v is not None}
    return {k: v for k, v in fields.items() if v is not None}
