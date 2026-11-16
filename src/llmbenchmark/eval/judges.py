"""Judge models for DeepEval's LLM-as-a-judge metrics.

DeepEval's default judge is OpenAI. ``ClaudeJudge`` plugs Claude in through
DeepEval's custom-model interface (``DeepEvalBaseLLM``). DeepEval calls
``generate(prompt, schema=SomePydanticModel)`` and expects an instance of that
schema back; Claude's structured outputs (``messages.parse``) guarantee valid
JSON for it.
"""

from __future__ import annotations

import json
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

# A judge is the "cheaper second model" in an eval, so Sonnet is the default.
DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"


class JudgeRefusal(RuntimeError):
    """The judge model declined to answer (stop_reason == "refusal")."""


class JudgeIncomplete(RuntimeError):
    """The judge's answer was cut off or had no parsable verdict."""


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        *,
        client: Any = None,
        async_client: Any = None,
        max_tokens: int = 16000,
        fallbacks: bool = True,
    ) -> None:
        """
        Args:
            model: Claude model id used for judging.
            client / async_client: optional preconfigured ``anthropic.Anthropic``
                / ``anthropic.AsyncAnthropic`` (proxies, custom base_url, tests).
            fallbacks: opt into server-side refusal fallbacks, so a rare safety
                decline is retried on another model instead of failing the
                metric. Claude API only; set False on Bedrock/Vertex/Foundry.
        """
        self.model_id = model
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self._client = client
        self._async_client = async_client
        super().__init__(model)

    # ---- DeepEvalBaseLLM interface -----------------------------------------

    def load_model(self):
        # DeepEval calls this from __init__; stay lazy so building a judge
        # never needs credentials. The client is created on first use.
        return self._client

    def _load_sync(self):
        if self._client is None:
            import anthropic  # reads ANTHROPIC_API_KEY / `ant auth login` profile

            self._client = anthropic.Anthropic()
        return self._client

    def get_model_name(self) -> str:
        return self.model_id

    def supports_temperature(self) -> bool:
        return False  # current Claude models reject sampling parameters

    def supports_structured_outputs(self) -> bool:
        return True

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        client = self._load_sync()
        if schema is None:
            response = client.messages.create(**self._request(prompt))
            return _text(_checked(response))
        try:
            response = client.messages.parse(**self._request(prompt), output_format=schema)
            return _parsed(response)
        except Exception as exc:
            if not _is_schema_rejection(exc):
                raise
        # Some DeepEval schemas use JSON-schema features structured outputs
        # doesn't support; fall back to asking for JSON and validating it.
        response = client.messages.create(**self._request(_json_prompt(prompt, schema)))
        return _parse_json(_text(_checked(response)), schema)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        client = self._load_async()
        if schema is None:
            response = await client.messages.create(**self._request(prompt))
            return _text(_checked(response))
        try:
            response = await client.messages.parse(**self._request(prompt), output_format=schema)
            return _parsed(response)
        except Exception as exc:
            if not _is_schema_rejection(exc):
                raise
        response = await client.messages.create(**self._request(_json_prompt(prompt, schema)))
        return _parse_json(_text(_checked(response)), schema)

    # ---- internals ---------------------------------------------------------

    def _load_async(self):
        if self._async_client is None:
            import anthropic

            self._async_client = anthropic.AsyncAnthropic()
        return self._async_client

    def _request(self, prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.fallbacks:
            kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kwargs["extra_body"] = {"fallbacks": "default"}
        return kwargs


def judge(model: str | DeepEvalBaseLLM | None = None, **kwargs: Any) -> DeepEvalBaseLLM | str:
    """Return something you can pass as ``model=`` to any DeepEval metric.

    - ``None`` or a ``claude-*`` id  -> ClaudeJudge
    - any other string (``"gpt-4.1"``) -> returned as-is; DeepEval's built-in
      OpenAI model handles it (needs OPENAI_API_KEY)
    - an existing DeepEvalBaseLLM     -> returned as-is
    """
    if model is None:
        return ClaudeJudge(**kwargs)
    if isinstance(model, str) and model.startswith("claude"):
        return ClaudeJudge(model, **kwargs)
    return model


def _checked(response: Any) -> Any:
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        raise JudgeRefusal(f"judge model refused (category={category})")
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise JudgeIncomplete("judge hit max_tokens before finishing; raise ClaudeJudge(max_tokens=...)")
    return response


def _parsed(response: Any) -> Any:
    parsed = _checked(response).parsed_output
    if parsed is None:
        raise JudgeIncomplete(
            f"judge returned no parsable verdict (stop_reason={getattr(response, 'stop_reason', None)})"
        )
    return parsed


def _text(response: Any) -> str:
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")


def _is_schema_rejection(exc: Exception) -> bool:
    # A 400 from the API about the output schema. Checked by name so this
    # module doesn't need anthropic's exception classes at import time.
    return type(exc).__name__ == "BadRequestError" and "schema" in str(exc).lower()


def _json_prompt(prompt: str, schema: type[BaseModel]) -> str:
    return (
        f"{prompt}\n\nRespond with only a JSON object matching this JSON schema, "
        f"and no other text:\n{json.dumps(schema.model_json_schema())}"
    )


def _parse_json(text: str, schema: type[BaseModel]) -> BaseModel:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"judge did not return JSON: {text[:200]!r}")
    return schema.model_validate_json(text[start : end + 1])
