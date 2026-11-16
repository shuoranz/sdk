"""Database tables for the reference backend (SQLModel = SQLAlchemy + Pydantic).

SQLite by default; set DATABASE_URL (e.g. postgresql+psycopg://...) for Postgres.
Every table has a ``team`` column; the API filters every query by it.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Column, event
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, SQLModel, create_engine


def _json(default: Any = None, name: str | None = None):
    column = Column(name, JSON) if name else Column(JSON)
    if callable(default):
        return Field(default_factory=default, sa_column=column)
    return Field(default=default, sa_column=column)


class RunRow(SQLModel, table=True):
    __tablename__ = "runs"
    id: str = Field(primary_key=True)
    team: str = Field(index=True)
    project: str = Field(index=True)
    name: str | None = None
    mode: str = "managed"
    model: str | None = Field(default=None, index=True)
    provider: str | None = None
    generation_params: dict[str, Any] = _json(dict)
    prompt: dict[str, Any] | None = _json()
    prompt_hash: str | None = Field(default=None, index=True)
    dataset: dict[str, Any] | None = _json()
    dataset_hash: str | None = Field(default=None, index=True)
    tags: list[str] = _json(list)
    status: str = "running"
    started_at: datetime
    ended_at: datetime | None = None
    lineage: dict[str, Any] = _json(dict)
    created_by: str | None = None
    # Set when the run closes: {"records": [ids], "spans": [ids]} the client logged.
    manifest: dict[str, Any] = _json(dict)


class RecordRow(SQLModel, table=True):
    __tablename__ = "records"
    id: str = Field(primary_key=True)
    team: str = Field(index=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    kind: str = "single"
    dataset_item_id: str | None = Field(default=None, index=True)
    sample_index: int = 0
    system: str | None = None
    input: str
    messages: list[dict[str, Any]] | None = _json()
    output: str | None = None
    expected_output: str | None = None
    expected_tools: list[str] | None = _json()
    tools_called: list[dict[str, Any]] | None = _json()
    context: list[str] | None = _json()
    retrieval_context: list[str] | None = _json()
    trace_id: str | None = Field(default=None, index=True)
    provider: str | None = None
    model: str | None = Field(default=None, index=True)
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    error: str | None = None
    # "metadata" is reserved by SQLAlchemy, so the column attribute is renamed.
    meta: dict[str, Any] = _json(dict, name="metadata")
    tags: list[str] = _json(list)
    created_at: datetime


class SpanRow(SQLModel, table=True):
    __tablename__ = "spans"
    id: str = Field(primary_key=True)
    team: str = Field(index=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    trace_id: str = Field(index=True)
    record_id: str = Field(foreign_key="records.id", index=True)
    parent_id: str | None = None
    kind: str = "other"
    name: str
    input: Any = _json()
    output: Any = _json()
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    started_at: datetime
    ended_at: datetime | None = None
    error: str | None = None
    attributes: dict[str, Any] = _json(dict)


class PromptRow(SQLModel, table=True):
    __tablename__ = "prompts"
    team: str = Field(primary_key=True)
    hash: str = Field(primary_key=True)
    name: str = Field(index=True)
    version: str | None = None
    system: str | None = None
    template: str | None = None
    created_at: datetime


class DatasetRow(SQLModel, table=True):
    __tablename__ = "datasets"
    team: str = Field(primary_key=True)
    hash: str = Field(primary_key=True)
    name: str = Field(index=True)
    version: str | None = None
    item_count: int
    verified: bool = False  # all items received and the hash recomputed from them
    created_at: datetime


class DatasetItemRow(SQLModel, table=True):
    __tablename__ = "dataset_items"
    team: str = Field(primary_key=True)
    dataset_hash: str = Field(primary_key=True)
    id: str = Field(primary_key=True)
    position: int = 0
    data: dict[str, Any] = _json(dict)


class EvaluationRow(SQLModel, table=True):
    __tablename__ = "evaluations"
    id: str = Field(primary_key=True)
    team: str = Field(index=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    source: str
    evaluator: str = Field(index=True)
    evaluator_version: str | None = None
    judge_model: str | None = None
    metrics: list[dict[str, Any]] = _json(list)
    config_hash: str = Field(index=True)
    supersedes: str | None = Field(default=None, foreign_key="evaluations.id")
    status: str = "open"
    created_at: datetime
    finalized_at: datetime | None = None
    created_by: str | None = None
    # Set when finalized: {"scores": [ids], "metrics": [ids]} the client produced.
    manifest: dict[str, Any] = _json(dict)


class ScoreRow(SQLModel, table=True):
    __tablename__ = "scores"
    id: str = Field(primary_key=True)
    team: str = Field(index=True)
    evaluation_id: str = Field(foreign_key="evaluations.id", index=True)
    record_id: str = Field(foreign_key="records.id", index=True)
    metric: str = Field(index=True)
    score: float | None = None
    passed: bool | None = None
    threshold: float | None = None
    reason: str | None = None
    error: str | None = None
    created_at: datetime


class RunMetricRow(SQLModel, table=True):
    __tablename__ = "run_metrics"
    id: str = Field(primary_key=True)
    team: str = Field(index=True)
    evaluation_id: str = Field(foreign_key="evaluations.id", index=True)
    name: str
    value: float
    n: int | None = None
    details: dict[str, Any] = _json(dict)
    created_at: datetime


class BaselineRow(SQLModel, table=True):
    __tablename__ = "baselines"
    team: str = Field(primary_key=True)
    project: str = Field(primary_key=True)
    run_id: str = Field(foreign_key="runs.id")
    evaluation_id: str = Field(foreign_key="evaluations.id")
    set_by: str | None = None
    set_at: datetime


class AuditRow(SQLModel, table=True):
    """Append-only log of state changes (who did what, when)."""

    __tablename__ = "audit_log"
    id: int | None = Field(default=None, primary_key=True)
    team: str = Field(index=True)
    actor: str
    action: str
    target_type: str
    target_id: str = Field(index=True)
    details: dict[str, Any] = _json(dict)
    at: datetime


def make_engine(url: str | None = None):
    url = url or os.environ.get("DATABASE_URL", "sqlite:///llmbenchmark.db")
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool  # one shared in-memory DB (tests)
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        # SQLite ignores foreign keys unless asked, per connection.
        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_conn, _record):  # pragma: no cover - trivial
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    SQLModel.metadata.create_all(engine)
    return engine
