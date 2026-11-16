"""Wire format shared by the SDK and the backend.

These Pydantic models are the contract between the notebook and the server.
Bump SCHEMA_VERSION when you make a breaking change.

Vocabulary (see docs/CONCEPTS.md):
- Run: one experiment (model + params + prompt version over a dataset version).
- Record: one evaluated example: a single prompt, a conversation, or an agent task.
- Span: one step of an agent's execution (llm call, tool call, ...); spans of
  one task form a trace tree attached to an "agent" Record.
- Evaluation: one immutable scoring pass over a run. Re-scoring creates a new one.
- Score / RunMetric: per-record verdicts / run-level numbers of an Evaluation.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

# Record ids end up in URL paths and in fallback file names, so they are
# restricted to a safe alphabet (UUID4s and sha256 hex digests fit).
ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
_ID_RE = re.compile(ID_PATTERN)
Id = Annotated[str, Field(pattern=ID_PATTERN)]

# Free-form JSON values (metadata, span input/output, tool arguments, ...) are
# cut at this nesting depth by the SDK. The server rejects request bodies
# nested deeper than SERVER_MAX_JSON_DEPTH (they can't be serialized back).
MAX_JSON_DEPTH = 32
SERVER_MAX_JSON_DEPTH = 64
TRUNCATED = "<nested too deeply: truncated by llmbenchmark>"

# Most records the server accepts in one batch request. The SDK splits
# larger batches so it never sends more than this.
MAX_BATCH_RECORDS = 1000

RunStatus = Literal["running", "completed", "failed"]
RunMode = Literal["managed", "external"]
RecordKind = Literal["single", "conversation", "agent"]
Role = Literal["system", "user", "assistant", "tool"]
SpanKind = Literal["agent", "llm", "tool", "retrieval", "handoff", "other"]
EvalSource = Literal["llmbenchmark", "external"]
EvalStatus = Literal["open", "finalized"]


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def is_valid_id(value: object) -> bool:
    return isinstance(value, str) and _ID_RE.fullmatch(value) is not None


def limit_depth(value: Any, max_depth: int = MAX_JSON_DEPTH, _depth: int = 0) -> Any:
    """A copy of dicts/lists/tuples with anything deeper than ``max_depth`` replaced by TRUNCATED.

    Recursion is bounded by max_depth, so this is safe on absurdly deep input
    (unlike json.dumps, which raises RecursionError).
    """
    if isinstance(value, dict):
        if _depth >= max_depth:
            return TRUNCATED
        return {k: limit_depth(v, max_depth, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if _depth >= max_depth:
            return TRUNCATED
        return [limit_depth(v, max_depth, _depth + 1) for v in value]
    return value


def json_depth(value: Any) -> int:
    """Nesting depth of a JSON-like value (iterative: no recursion limit)."""
    deepest, stack = 0, [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, (list, tuple)):
            children = current
        else:
            continue
        deepest = max(deepest, depth + 1)
        stack.extend((child, depth + 1) for child in children)
    return deepest


# ---- building blocks -------------------------------------------------------


class ToolCall(BaseModel):
    """A tool the model/agent invoked (maps to DeepEval's ToolCall)."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    id: str | None = None


class Message(BaseModel):
    role: Role
    content: str = ""
    name: str | None = None
    tool_calls: list[ToolCall] | None = None


class PromptRef(BaseModel):
    """Which prompt a run used. `hash` is the content fingerprint."""

    name: str
    version: str | None = None
    hash: Id


class DatasetRef(BaseModel):
    name: str
    version: str | None = None
    hash: Id
    item_count: int | None = None


class Lineage(BaseModel):
    """Everything needed to reproduce a run, captured automatically."""

    sdk_version: str | None = None
    schema_version: int = SCHEMA_VERSION
    python: str | None = None
    packages: dict[str, str] = Field(default_factory=dict)
    git_commit: str | None = None
    git_dirty: bool | None = None
    # Required-but-missing fields in managed mode (e.g. ["model", "prompt"]).
    incomplete: list[str] = Field(default_factory=list)


# ---- runs ------------------------------------------------------------------


class RunCreate(BaseModel):
    id: Id = Field(default_factory=new_id)
    project: str
    name: str | None = None
    mode: RunMode = "managed"
    model: str | None = None
    provider: str | None = None
    generation_params: dict[str, Any] = Field(default_factory=dict)
    prompt: PromptRef | None = None
    dataset: DatasetRef | None = None
    tags: list[str] = Field(default_factory=list)
    status: RunStatus = "running"
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    lineage: Lineage = Field(default_factory=Lineage)


class RunInfo(RunCreate):
    """A run as the server returns it (adds server-assigned fields)."""

    team: str | None = None
    created_by: str | None = None


class RunUpdate(BaseModel):
    """Close a run. The manifest lists every record/span the client logged, so
    the server can accept late deliveries of exactly those (retries/replays)
    and nothing else. The server stamps ended_at with its own clock."""

    status: RunStatus
    record_ids: list[Id] = Field(default_factory=list)
    span_ids: list[Id] = Field(default_factory=list)


# ---- records and spans -----------------------------------------------------


class Record(BaseModel):
    """One evaluated example: a single prompt, a conversation, or an agent task."""

    id: Id = Field(default_factory=new_id)
    run_id: Id
    kind: RecordKind = "single"
    dataset_item_id: Id | None = None
    sample_index: int = 0  # >0 when the same item is sampled several times
    system: str | None = None
    input: str  # the user prompt / task, always set (for display and simple metrics)
    messages: list[Message] | None = None  # full conversation sent to the model
    output: str | None = None
    expected_output: str | None = None
    expected_tools: list[str] | None = None
    tools_called: list[ToolCall] | None = None
    context: list[str] | None = None  # ground truth (DeepEval HallucinationMetric)
    retrieval_context: list[str] | None = None  # RAG chunks (Faithfulness, ...)
    trace_id: Id | None = None
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class Span(BaseModel):
    """One step of an agent trace. Field names follow OpenTelemetry GenAI ideas."""

    id: Id = Field(default_factory=new_id)
    run_id: Id
    trace_id: Id
    record_id: Id
    parent_id: Id | None = None
    kind: SpanKind = "other"
    name: str
    input: Any = None
    output: Any = None
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    error: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


# ---- prompts and datasets --------------------------------------------------


class PromptVersion(BaseModel):
    """A prompt's full content, stored once per (team, hash)."""

    hash: Id
    name: str
    version: str | None = None
    system: str | None = None
    template: str | None = None


class DatasetItem(BaseModel):
    """A test case before any model answered it (DeepEval calls this a golden)."""

    id: Id
    input: str
    system: str | None = None
    messages: list[Message] | None = None
    expected_output: str | None = None
    expected_tools: list[str] | None = None
    context: list[str] | None = None
    retrieval_context: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetCreate(BaseModel):
    hash: Id
    name: str
    version: str | None = None
    item_count: int = Field(ge=0)


class DatasetInfo(DatasetCreate):
    # True once the server has every item and recomputed the hash from them.
    verified: bool = False


# ---- evaluations -----------------------------------------------------------


class MetricSpec(BaseModel):
    """How a metric was configured: part of the evaluation's config hash."""

    name: str
    threshold: float | None = None
    criteria: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluationCreate(BaseModel):
    id: Id = Field(default_factory=new_id)
    run_id: Id
    source: EvalSource = "external"
    evaluator: str  # e.g. "llmbenchmark" or "team-x-rubric"
    evaluator_version: str | None = None
    judge_model: str | None = None
    metrics: list[MetricSpec] = Field(default_factory=list)
    config_hash: Id
    supersedes: Id | None = None  # the evaluation this one corrects/replaces
    created_at: datetime = Field(default_factory=utcnow)


class EvaluationFinalize(BaseModel):
    """Freeze an evaluation. Manifest of everything the client scored (see RunUpdate)."""

    score_ids: list[Id] = Field(default_factory=list)
    metric_ids: list[Id] = Field(default_factory=list)


class EvaluationInfo(EvaluationCreate):
    status: EvalStatus = "open"
    finalized_at: datetime | None = None
    superseded_by: str | None = None
    team: str | None = None
    created_by: str | None = None


class Score(BaseModel):
    """One metric's verdict on one record."""

    id: Id = Field(default_factory=new_id)
    evaluation_id: Id
    record_id: Id
    metric: str
    score: float | None = None
    passed: bool | None = None
    threshold: float | None = None
    reason: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class RunMetric(BaseModel):
    """A run-level number reported by the evaluator (e.g. accuracy 0.87)."""

    id: Id = Field(default_factory=new_id)
    evaluation_id: Id
    name: str
    value: float
    n: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class BaselineSet(BaseModel):
    project: str
    run_id: Id
    evaluation_id: Id


# ---- batches ---------------------------------------------------------------


class RecordBatch(BaseModel):
    records: list[Record]


class SpanBatch(BaseModel):
    spans: list[Span]


class ScoreBatch(BaseModel):
    scores: list[Score]


class MetricBatch(BaseModel):
    metrics: list[RunMetric]


class ItemBatch(BaseModel):
    items: list[DatasetItem]
