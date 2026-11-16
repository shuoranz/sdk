"""Reference backend for llmbenchmark.

Run it:   LLMBENCHMARK_SERVER_KEYS_FILE=keys.json uv run uvicorn server.app:create_app --factory
          (or LLMBENCHMARK_SERVER_API_KEYS=dev-key for one team, or
           LLMBENCHMARK_SERVER_ALLOW_ANONYMOUS=1 for local development)
Docs UI:  http://localhost:8000/docs

Rules enforced here (see AGENTS.md):
- Auth: every /v1 request needs a key; the key decides the team and role.
  Every query is filtered by team; other teams' data is a 404.
- Idempotent writes: records are keyed by client-generated ids; resending is
  safe. An id owned by another run/team is a 409, never an overwrite.
- Immutability: a finished run and a finalized evaluation are frozen. Late
  deliveries (retries/replays of data created before the run ended) are
  accepted; changes are a 409.
- Audit: state changes are appended to audit_log with the key's name.
"""

# No `from __future__ import annotations` here: FastAPI resolves string
# annotations in module globals, but the route dependencies (Reader, Writer,
# Db) are closures defined inside create_app().
import hashlib
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, select

from llmbenchmark import __version__
from llmbenchmark.schemas import (
    ID_PATTERN,
    MAX_BATCH_RECORDS,
    SERVER_MAX_JSON_DEPTH,
    BaselineSet,
    DatasetCreate,
    DatasetInfo,
    DatasetItem,
    EvaluationCreate,
    EvaluationFinalize,
    EvaluationInfo,
    ItemBatch,
    MetricBatch,
    PromptVersion,
    Record,
    RecordBatch,
    RunCreate,
    RunInfo,
    RunMetric,
    RunUpdate,
    Score,
    ScoreBatch,
    Span,
    SpanBatch,
    json_depth,
    utcnow,
)
from llmbenchmark.versioning import content_hash

from .auth import ANONYMOUS, Principal, load_key_store
from .db import (
    AuditRow,
    BaselineRow,
    DatasetItemRow,
    DatasetRow,
    EvaluationRow,
    PromptRow,
    RecordRow,
    RunMetricRow,
    RunRow,
    ScoreRow,
    SpanRow,
    make_engine,
)

logger = logging.getLogger("llmbenchmark.server")

DEFAULT_MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_PAGE_SIZE = 1000

PathId = Annotated[str, Path(pattern=ID_PATTERN)]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class _BodySizeLimit:
    """Reject request bodies over ``max_bytes`` with 413 (pure ASGI middleware)."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            response = JSONResponse({"detail": f"request body too large (max {self.max_bytes} bytes)"}, 413)
            return await response(scope, receive, send)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:  # chunked body without Content-Length
                    raise HTTPException(status_code=413, detail="request body too large")
            return message

        await self.app(scope, limited_receive, send)


# ---- row <-> model conversion ----------------------------------------------

_RENAMES_IN = {"metadata": "meta"}
_RENAMES_OUT = {"meta": "metadata"}


def _utc(value: Any) -> Any:
    """SQLite returns naive datetimes; everything we store is UTC."""
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _check_depth(data: Any) -> None:
    # Accepting deeper JSON would store rows that can't be serialized back
    # (every later read of the run would fail with a 500).
    if json_depth(data) > SERVER_MAX_JSON_DEPTH:
        raise HTTPException(422, f"JSON nested deeper than {SERVER_MAX_JSON_DEPTH} levels")


def _to_row(model: BaseModel, row_cls: type[SQLModel], **extra: Any) -> SQLModel:
    data = {_RENAMES_IN.get(k, k): v for k, v in model.model_dump().items()}
    _check_depth(data)
    allowed = set(row_cls.model_fields)
    return row_cls(**{k: v for k, v in {**data, **extra}.items() if k in allowed})


def _from_row(row: SQLModel, model_cls: type[BaseModel], **extra: Any) -> Any:
    data = {_RENAMES_OUT.get(k, k): _utc(v) for k, v in row.model_dump().items()}
    allowed = set(model_cls.model_fields)
    return model_cls.model_validate({k: v for k, v in {**data, **extra}.items() if k in allowed})


def _same(a: BaseModel, b: BaseModel) -> bool:
    return a.model_dump(mode="json") == b.model_dump(mode="json")


def create_app(
    database_url: str | None = None,
    api_keys: set[str] | None = None,
    *,
    keys: dict[str, dict[str, str]] | None = None,
    keys_file: str | None = None,
    allow_anonymous: bool | None = None,
    max_body_bytes: int | None = None,
) -> FastAPI:
    """Build the app.

    Keys come from ``api_keys`` (team "default"), ``keys`` ({raw_key: {team, role, name}}),
    and/or a keys file. With no arguments, the env vars LLMBENCHMARK_SERVER_API_KEYS
    and LLMBENCHMARK_SERVER_KEYS_FILE are used.
    """
    engine = make_engine(database_url)
    if api_keys is None and keys is None and keys_file is None:
        raw = os.environ.get("LLMBENCHMARK_SERVER_API_KEYS", "")
        api_keys = {k.strip() for k in raw.split(",") if k.strip()}
        keys_file = os.environ.get("LLMBENCHMARK_SERVER_KEYS_FILE") or None
    store = load_key_store(api_keys=api_keys or (), keys=keys, keys_file=keys_file)
    if allow_anonymous is None:
        allow_anonymous = _env_flag("LLMBENCHMARK_SERVER_ALLOW_ANONYMOUS")
    if not len(store):
        if not allow_anonymous:
            raise RuntimeError(
                "No API keys configured. Set LLMBENCHMARK_SERVER_KEYS_FILE (per-team keys) or "
                "LLMBENCHMARK_SERVER_API_KEYS=key1,key2, or set LLMBENCHMARK_SERVER_ALLOW_ANONYMOUS=1 "
                "to run without auth (local development only)."
            )
        logger.warning("llmbenchmark server: authentication is DISABLED (anonymous mode)")
    if max_body_bytes is None:
        max_body_bytes = int(os.environ.get("LLMBENCHMARK_SERVER_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES))

    app = FastAPI(title="llmbenchmark backend", version=__version__)
    app.add_middleware(_BodySizeLimit, max_bytes=max_body_bytes)

    @app.exception_handler(IntegrityError)
    async def _integrity_error(request: Request, exc: IntegrityError) -> JSONResponse:
        # A constraint violation is the client's problem, not a retryable 500.
        return JSONResponse({"detail": "conflicts with existing data"}, status_code=409)

    # ---- dependencies -----------------------------------------------------

    def session():
        with Session(engine) as s:
            yield s

    def principal(authorization: str | None = Header(default=None)) -> Principal:
        if not len(store):
            return ANONYMOUS
        scheme, _, token = (authorization or "").partition(" ")
        found = store.authenticate(token.strip()) if scheme.lower() == "bearer" else None
        if found is None:
            raise HTTPException(401, "invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})
        return found

    def writer(p: Principal = Depends(principal)) -> Principal:
        if not p.can_write:
            raise HTTPException(403, "this API key is read-only")
        return p

    Reader = Annotated[Principal, Depends(principal)]
    Writer = Annotated[Principal, Depends(writer)]
    Db = Annotated[Session, Depends(session)]
    Limit = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)]
    Offset = Annotated[int, Query(ge=0)]

    # ---- helpers ------------------------------------------------------------

    def get_owned(s: Session, row_cls: type[SQLModel], key: Any, p: Principal, what: str):
        row = s.get(row_cls, key)
        if row is None or row.team != p.team:
            raise HTTPException(404, f"{what} {key if isinstance(key, str) else key[-1]} not found")
        return row

    def audit(s: Session, p: Principal, action: str, target_type: str, target_id: str, **details: Any) -> None:
        s.add(AuditRow(team=p.team, actor=p.name, action=action, target_type=target_type,
                       target_id=target_id, details=details, at=utcnow()))

    def check_batch(n: int) -> None:
        if n > MAX_BATCH_RECORDS:
            raise HTTPException(413, f"batch too large: {n} records (max {MAX_BATCH_RECORDS})")

    def upsert_children(
        s: Session,
        row_cls: type[SQLModel],
        model_cls: type[BaseModel],
        items: list[BaseModel],
        *,
        p: Principal,
        owner_field: str,
        owner_id: str,
        closed: bool,
        late_ids: Iterable[str] = (),
        extra: Callable[[BaseModel], dict[str, Any]] = lambda _: {},
    ) -> None:
        """Insert/update child rows of a run or evaluation, enforcing ownership and immutability.

        - an id that already belongs to another owner/team -> 409
        - owner closed (run finished / evaluation finalized): an existing row
          may only be resent identically; a new row is accepted only if its id
          is in the manifest the client sent when closing (a late delivery of
          a retry or replay). No client timestamp is trusted. Otherwise 409.
        """
        late = set(late_ids)
        ids = [i.id for i in items]  # type: ignore[attr-defined]
        existing = {r.id: r for r in s.exec(select(row_cls).where(row_cls.id.in_(ids)))} if ids else {}
        for item in items:
            row = existing.get(item.id)  # type: ignore[attr-defined]
            if row is not None and (row.team != p.team or getattr(row, owner_field) != owner_id):
                raise HTTPException(409, f"id {item.id} already belongs to another run or evaluation")  # type: ignore[attr-defined]
            if closed:
                if row is not None:
                    if not _same(_from_row(row, model_cls), item):
                        raise HTTPException(409, f"{item.id} is immutable: its run/evaluation is closed")  # type: ignore[attr-defined]
                    continue
                if item.id not in late:  # type: ignore[attr-defined]
                    raise HTTPException(409, "cannot add to a closed run/evaluation; start a new one")
            s.merge(_to_row(item, row_cls, team=p.team, **extra(item)))

    def evaluation_info(s: Session, row: EvaluationRow) -> EvaluationInfo:
        newer = s.exec(select(EvaluationRow.id).where(EvaluationRow.supersedes == row.id)).first()
        return _from_row(row, EvaluationInfo, superseded_by=newer)

    def run_info(row: RunRow) -> RunInfo:
        return _from_row(row, RunInfo)

    # ---- health -------------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # ---- runs ---------------------------------------------------------------

    @app.post("/v1/runs", status_code=201)
    def create_run(run: RunCreate, response: Response, p: Writer, s: Db) -> RunInfo:
        existing = s.get(RunRow, run.id)
        if existing is not None:
            # Insert-if-absent: a retry or replay must not reset a finished run
            # or move it to another team/project.
            if existing.team != p.team or existing.project != run.project:
                raise HTTPException(409, "run id already exists elsewhere")
            response.status_code = 200
            return run_info(existing)
        # A run always starts open: a client can't create it already "finished"
        # (which would skip the close manifest and the audit entry).
        row = _to_row(run, RunRow, team=p.team, created_by=p.name, status="running", ended_at=None,
                      prompt_hash=run.prompt.hash if run.prompt else None,
                      dataset_hash=run.dataset.hash if run.dataset else None)
        s.add(row)
        audit(s, p, "run.create", "run", run.id, project=run.project)
        s.commit()
        s.refresh(row)
        return run_info(row)

    @app.patch("/v1/runs/{run_id}")
    def update_run(run_id: PathId, update: RunUpdate, p: Writer, s: Db) -> RunInfo:
        row = get_owned(s, RunRow, run_id, p, "run")
        if row.status != "running":
            if row.status == update.status:
                return run_info(row)  # replayed finish: no-op
            raise HTTPException(409, f"run is already {row.status}; finished runs are immutable")
        if update.status == "running":
            return run_info(row)
        row.status = update.status
        row.ended_at = utcnow()  # server clock, never the client's
        row.manifest = {"records": update.record_ids, "spans": update.span_ids}
        s.add(row)
        audit(s, p, f"run.{update.status}", "run", run_id,
              records=len(update.record_ids), spans=len(update.span_ids))
        s.commit()
        s.refresh(row)
        return run_info(row)

    @app.get("/v1/runs")
    def list_runs(
        p: Reader, s: Db, project: str | None = None, model: str | None = None,
        mode: str | None = None, limit: Limit = 100, offset: Offset = 0,
    ) -> list[RunInfo]:
        stmt = select(RunRow).where(RunRow.team == p.team)
        if project:
            stmt = stmt.where(RunRow.project == project)
        if model:
            stmt = stmt.where(RunRow.model == model)
        if mode:
            stmt = stmt.where(RunRow.mode == mode)
        stmt = stmt.order_by(RunRow.started_at.desc(), RunRow.id).offset(offset).limit(limit)
        return [run_info(r) for r in s.exec(stmt)]

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: PathId, p: Reader, s: Db) -> RunInfo:
        return run_info(get_owned(s, RunRow, run_id, p, "run"))

    # ---- records and spans --------------------------------------------------

    @app.post("/v1/runs/{run_id}/records", status_code=201)
    def add_records(run_id: PathId, batch: RecordBatch, p: Writer, s: Db) -> dict[str, int]:
        check_batch(len(batch.records))
        run = get_owned(s, RunRow, run_id, p, "run")
        if any(r.run_id != run_id for r in batch.records):
            raise HTTPException(422, "record.run_id does not match URL")
        upsert_children(s, RecordRow, Record, batch.records, p=p, owner_field="run_id", owner_id=run_id,
                        closed=run.status != "running", late_ids=(run.manifest or {}).get("records", []))
        s.commit()
        return {"received": len(batch.records)}

    @app.get("/v1/runs/{run_id}/records")
    def list_records(run_id: PathId, p: Reader, s: Db, limit: Limit = MAX_PAGE_SIZE,
                     offset: Offset = 0) -> list[Record]:
        get_owned(s, RunRow, run_id, p, "run")
        stmt = (select(RecordRow).where(RecordRow.run_id == run_id)
                .order_by(RecordRow.created_at, RecordRow.id).offset(offset).limit(limit))
        return [_from_row(r, Record) for r in s.exec(stmt)]

    @app.post("/v1/runs/{run_id}/spans", status_code=201)
    def add_spans(run_id: PathId, batch: SpanBatch, p: Writer, s: Db) -> dict[str, int]:
        check_batch(len(batch.spans))
        run = get_owned(s, RunRow, run_id, p, "run")
        if any(sp.run_id != run_id for sp in batch.spans):
            raise HTTPException(422, "span.run_id does not match URL")
        record_ids = {sp.record_id for sp in batch.spans}
        known = set(s.exec(select(RecordRow.id).where(RecordRow.run_id == run_id, RecordRow.id.in_(record_ids))))
        if record_ids - known:
            raise HTTPException(422, f"record(s) not found in this run: {sorted(record_ids - known)[:5]}")
        upsert_children(s, SpanRow, Span, batch.spans, p=p, owner_field="run_id", owner_id=run_id,
                        closed=run.status != "running", late_ids=(run.manifest or {}).get("spans", []))
        s.commit()
        return {"received": len(batch.spans)}

    @app.get("/v1/runs/{run_id}/spans")
    def list_spans(run_id: PathId, p: Reader, s: Db, trace_id: str | None = None,
                   limit: Limit = MAX_PAGE_SIZE, offset: Offset = 0) -> list[Span]:
        get_owned(s, RunRow, run_id, p, "run")
        stmt = select(SpanRow).where(SpanRow.run_id == run_id)
        if trace_id:
            stmt = stmt.where(SpanRow.trace_id == trace_id)
        stmt = stmt.order_by(SpanRow.started_at, SpanRow.id).offset(offset).limit(limit)
        return [_from_row(r, Span) for r in s.exec(stmt)]

    # ---- prompts and datasets (content-addressed, per team) -----------------

    @app.post("/v1/prompts", status_code=201)
    def add_prompt(prompt: PromptVersion, response: Response, p: Writer, s: Db) -> PromptVersion:
        if content_hash({"system": prompt.system, "template": prompt.template}) != prompt.hash:
            raise HTTPException(422, "prompt hash does not match its content")
        existing = s.get(PromptRow, (p.team, prompt.hash))
        if existing is not None:
            response.status_code = 200
            return _from_row(existing, PromptVersion)
        s.add(_to_row(prompt, PromptRow, team=p.team, created_at=utcnow()))
        s.commit()
        return prompt

    @app.get("/v1/prompts/{prompt_hash}")
    def get_prompt(prompt_hash: PathId, p: Reader, s: Db) -> PromptVersion:
        return _from_row(get_owned(s, PromptRow, (p.team, prompt_hash), p, "prompt"), PromptVersion)

    @app.post("/v1/datasets", status_code=201)
    def add_dataset(dataset: DatasetCreate, response: Response, p: Writer, s: Db) -> DatasetInfo:
        existing = s.get(DatasetRow, (p.team, dataset.hash))
        if existing is not None:
            response.status_code = 200
            return _from_row(existing, DatasetInfo)
        empty = dataset.item_count == 0
        if empty and dataset.hash != content_hash([]):
            raise HTTPException(422, "dataset hash does not match its (empty) content")
        row = _to_row(dataset, DatasetRow, team=p.team, created_at=utcnow(), verified=empty)
        s.add(row)
        s.commit()
        s.refresh(row)
        return _from_row(row, DatasetInfo)

    @app.post("/v1/datasets/{dataset_hash}/items", status_code=201)
    def add_dataset_items(dataset_hash: PathId, batch: ItemBatch, p: Writer, s: Db) -> dict[str, int]:
        check_batch(len(batch.items))
        dataset = get_owned(s, DatasetRow, (p.team, dataset_hash), p, "dataset")
        count = len(s.exec(select(DatasetItemRow.id).where(
            DatasetItemRow.team == p.team, DatasetItemRow.dataset_hash == dataset_hash)).all())
        for item in batch.items:
            existing = s.get(DatasetItemRow, (p.team, dataset_hash, item.id))
            data = item.model_dump(mode="json")
            if existing is not None:
                if existing.data != data:
                    raise HTTPException(409, f"dataset item {item.id} differs from the stored version")
                continue
            if count >= dataset.item_count:
                raise HTTPException(409, "dataset already has all its items")
            _check_depth(data)
            s.add(DatasetItemRow(team=p.team, dataset_hash=dataset_hash, id=item.id, position=count, data=data))
            count += 1
        if count == dataset.item_count and not dataset.verified:
            # Content-addressed for real: recompute the hash from the stored items.
            s.flush()
            stored = s.exec(select(DatasetItemRow.data).where(
                DatasetItemRow.team == p.team, DatasetItemRow.dataset_hash == dataset_hash)
                .order_by(DatasetItemRow.position)).all()
            if content_hash(list(stored)) != dataset_hash:
                s.rollback()
                raise HTTPException(422, "dataset items do not match the dataset hash")
            dataset.verified = True
            s.add(dataset)
        s.commit()
        return {"received": len(batch.items)}

    @app.get("/v1/datasets/{dataset_hash}")
    def get_dataset(dataset_hash: PathId, p: Reader, s: Db) -> DatasetInfo:
        return _from_row(get_owned(s, DatasetRow, (p.team, dataset_hash), p, "dataset"), DatasetInfo)

    @app.get("/v1/datasets/{dataset_hash}/items")
    def list_dataset_items(dataset_hash: PathId, p: Reader, s: Db, limit: Limit = MAX_PAGE_SIZE,
                           offset: Offset = 0) -> list[DatasetItem]:
        get_owned(s, DatasetRow, (p.team, dataset_hash), p, "dataset")
        stmt = (select(DatasetItemRow).where(DatasetItemRow.team == p.team,
                                             DatasetItemRow.dataset_hash == dataset_hash)
                .order_by(DatasetItemRow.position).offset(offset).limit(limit))
        return [DatasetItem.model_validate(r.data) for r in s.exec(stmt)]

    # ---- evaluations ----------------------------------------------------------

    @app.post("/v1/runs/{run_id}/evaluations", status_code=201)
    def create_evaluation(run_id: PathId, ev: EvaluationCreate, response: Response,
                          p: Writer, s: Db) -> EvaluationInfo:
        get_owned(s, RunRow, run_id, p, "run")
        if ev.run_id != run_id:
            raise HTTPException(422, "evaluation.run_id does not match URL")
        existing = s.get(EvaluationRow, ev.id)
        if existing is not None:
            if existing.team != p.team or existing.run_id != run_id:
                raise HTTPException(409, "evaluation id already exists elsewhere")
            response.status_code = 200
            return evaluation_info(s, existing)
        if ev.supersedes is not None:
            old = s.get(EvaluationRow, ev.supersedes)
            if old is None or old.team != p.team or old.run_id != run_id:
                raise HTTPException(422, "supersedes must be an evaluation of the same run")
        row = _to_row(ev, EvaluationRow, team=p.team, created_by=p.name, status="open")
        s.add(row)
        audit(s, p, "evaluation.create", "evaluation", ev.id, run_id=run_id, evaluator=ev.evaluator,
              supersedes=ev.supersedes, config_hash=ev.config_hash)
        s.commit()
        s.refresh(row)
        return evaluation_info(s, row)

    @app.get("/v1/runs/{run_id}/evaluations")
    def list_evaluations(run_id: PathId, p: Reader, s: Db) -> list[EvaluationInfo]:
        get_owned(s, RunRow, run_id, p, "run")
        rows = s.exec(select(EvaluationRow).where(EvaluationRow.run_id == run_id)
                      .order_by(EvaluationRow.created_at, EvaluationRow.id))
        return [evaluation_info(s, r) for r in rows]

    @app.get("/v1/evaluations/{evaluation_id}")
    def get_evaluation(evaluation_id: PathId, p: Reader, s: Db) -> EvaluationInfo:
        return evaluation_info(s, get_owned(s, EvaluationRow, evaluation_id, p, "evaluation"))

    @app.post("/v1/evaluations/{evaluation_id}/scores", status_code=201)
    def add_scores(evaluation_id: PathId, batch: ScoreBatch, p: Writer, s: Db) -> dict[str, int]:
        check_batch(len(batch.scores))
        ev = get_owned(s, EvaluationRow, evaluation_id, p, "evaluation")
        if any(sc.evaluation_id != evaluation_id for sc in batch.scores):
            raise HTTPException(422, "score.evaluation_id does not match URL")
        record_ids = {sc.record_id for sc in batch.scores}
        known = set(s.exec(select(RecordRow.id).where(RecordRow.run_id == ev.run_id,
                                                      RecordRow.id.in_(record_ids))))
        if record_ids - known:
            raise HTTPException(422, f"record(s) not found in this run: {sorted(record_ids - known)[:5]}")
        upsert_children(s, ScoreRow, Score, batch.scores, p=p, owner_field="evaluation_id",
                        owner_id=evaluation_id, closed=ev.status == "finalized",
                        late_ids=(ev.manifest or {}).get("scores", []))
        s.commit()
        return {"received": len(batch.scores)}

    @app.get("/v1/evaluations/{evaluation_id}/scores")
    def list_scores(evaluation_id: PathId, p: Reader, s: Db, limit: Limit = MAX_PAGE_SIZE,
                    offset: Offset = 0) -> list[Score]:
        get_owned(s, EvaluationRow, evaluation_id, p, "evaluation")
        stmt = (select(ScoreRow).where(ScoreRow.evaluation_id == evaluation_id)
                .order_by(ScoreRow.created_at, ScoreRow.id).offset(offset).limit(limit))
        return [_from_row(r, Score) for r in s.exec(stmt)]

    @app.post("/v1/evaluations/{evaluation_id}/metrics", status_code=201)
    def add_metrics(evaluation_id: PathId, batch: MetricBatch, p: Writer, s: Db) -> dict[str, int]:
        check_batch(len(batch.metrics))
        ev = get_owned(s, EvaluationRow, evaluation_id, p, "evaluation")
        if any(m.evaluation_id != evaluation_id for m in batch.metrics):
            raise HTTPException(422, "metric.evaluation_id does not match URL")
        upsert_children(s, RunMetricRow, RunMetric, batch.metrics, p=p, owner_field="evaluation_id",
                        owner_id=evaluation_id, closed=ev.status == "finalized",
                        late_ids=(ev.manifest or {}).get("metrics", []))
        s.commit()
        return {"received": len(batch.metrics)}

    @app.get("/v1/evaluations/{evaluation_id}/metrics")
    def list_metrics(evaluation_id: PathId, p: Reader, s: Db) -> list[RunMetric]:
        get_owned(s, EvaluationRow, evaluation_id, p, "evaluation")
        rows = s.exec(select(RunMetricRow).where(RunMetricRow.evaluation_id == evaluation_id)
                      .order_by(RunMetricRow.created_at, RunMetricRow.id))
        return [_from_row(r, RunMetric) for r in rows]

    @app.post("/v1/evaluations/{evaluation_id}/finalize")
    def finalize_evaluation(evaluation_id: PathId, p: Writer, s: Db,
                            body: EvaluationFinalize | None = None) -> EvaluationInfo:
        ev = get_owned(s, EvaluationRow, evaluation_id, p, "evaluation")
        if ev.status != "finalized":
            body = body or EvaluationFinalize()
            ev.status = "finalized"
            ev.finalized_at = utcnow()  # server clock, never the client's
            ev.manifest = {"scores": body.score_ids, "metrics": body.metric_ids}
            s.add(ev)
            audit(s, p, "evaluation.finalize", "evaluation", evaluation_id,
                  scores=len(body.score_ids), metrics=len(body.metric_ids))
            s.commit()
            s.refresh(ev)
        return evaluation_info(s, ev)

    @app.get("/v1/evaluations/{evaluation_id}/summary")
    def evaluation_summary(evaluation_id: PathId, p: Reader, s: Db) -> dict[str, Any]:
        ev = get_owned(s, EvaluationRow, evaluation_id, p, "evaluation")
        scores = s.exec(select(ScoreRow).where(ScoreRow.evaluation_id == evaluation_id)).all()
        reported = s.exec(select(RunMetricRow).where(RunMetricRow.evaluation_id == evaluation_id)).all()
        return {
            "evaluation": evaluation_info(s, ev).model_dump(mode="json"),
            "metrics": _summarize(scores),
            "reported": [_from_row(m, RunMetric).model_dump(mode="json") for m in reported],
        }

    # ---- baselines and comparison ---------------------------------------------

    @app.put("/v1/baselines")
    def set_baseline(body: BaselineSet, p: Writer, s: Db) -> dict[str, Any]:
        run = get_owned(s, RunRow, body.run_id, p, "run")
        ev = get_owned(s, EvaluationRow, body.evaluation_id, p, "evaluation")
        if ev.run_id != run.id or run.project != body.project:
            raise HTTPException(422, "the evaluation must belong to the run, and the run to the project")
        if ev.status != "finalized":
            raise HTTPException(409, "only a finalized evaluation can be a baseline")
        s.merge(BaselineRow(team=p.team, project=body.project, run_id=run.id, evaluation_id=ev.id,
                            set_by=p.name, set_at=utcnow()))
        audit(s, p, "baseline.set", "project", body.project, run_id=run.id, evaluation_id=ev.id)
        s.commit()
        return _baseline_out(s.get(BaselineRow, (p.team, body.project)))

    @app.get("/v1/baselines")
    def get_baseline(project: str, p: Reader, s: Db) -> dict[str, Any]:
        row = s.get(BaselineRow, (p.team, project))
        if row is None:
            raise HTTPException(404, f"no baseline set for project {project!r}")
        return _baseline_out(row)

    @app.get("/v1/compare")
    def compare(p: Reader, s: Db, head: Annotated[str, Query(pattern=ID_PATTERN)],
                base: Annotated[str | None, Query(pattern=ID_PATTERN)] = None) -> dict[str, Any]:
        head_ev = get_owned(s, EvaluationRow, head, p, "evaluation")
        head_run = s.get(RunRow, head_ev.run_id)
        if base is None:
            baseline = s.get(BaselineRow, (p.team, head_run.project))
            if baseline is None:
                raise HTTPException(404, f"no baseline set for project {head_run.project!r}; pass base=")
            base = baseline.evaluation_id
        base_ev = get_owned(s, EvaluationRow, base, p, "evaluation")
        base_run = s.get(RunRow, base_ev.run_id)
        return _compare(s, base_ev, base_run, head_ev, head_run)

    @app.get("/v1/audit")
    def list_audit(p: Reader, s: Db, target_id: str | None = None, limit: Limit = 100,
                   offset: Offset = 0) -> list[dict[str, Any]]:
        stmt = select(AuditRow).where(AuditRow.team == p.team)
        if target_id:
            stmt = stmt.where(AuditRow.target_id == target_id)
        stmt = stmt.order_by(AuditRow.id.desc()).offset(offset).limit(limit)
        return [{**r.model_dump(exclude={"team"}), "at": _utc(r.at).isoformat()} for r in s.exec(stmt)]

    return app


# ---- comparison logic (pure functions over rows) -------------------------------


def _baseline_out(row: BaselineRow) -> dict[str, Any]:
    return {"project": row.project, "run_id": row.run_id, "evaluation_id": row.evaluation_id,
            "set_by": row.set_by, "set_at": _utc(row.set_at).isoformat()}


def _summarize(scores: Iterable[ScoreRow]) -> list[dict[str, Any]]:
    by_metric: dict[str, list[ScoreRow]] = defaultdict(list)
    for sc in scores:
        by_metric[sc.metric].append(sc)
    out = []
    for metric, rows in sorted(by_metric.items()):
        values = [r.score for r in rows if r.score is not None]
        verdicts = [r.passed for r in rows if r.passed is not None]
        out.append({
            "metric": metric,
            "mean_score": sum(values) / len(values) if values else None,
            "pass_rate": sum(verdicts) / len(verdicts) if verdicts else None,
            "n": len(rows),
            "errors": sum(1 for r in rows if r.error),
        })
    return out


def _record_key(r: RecordRow) -> str:
    if r.dataset_item_id:
        return f"item:{r.dataset_item_id}#{r.sample_index}"
    digest = hashlib.sha256(f"{r.system or ''}\x00{r.input}".encode()).hexdigest()[:16]
    return f"input:{digest}#{r.sample_index}"


def _side(s: Session, ev: EvaluationRow) -> tuple[dict[tuple[str, str], ScoreRow], dict[str, RecordRow], int]:
    """Scores and records of one side, keyed so both sides can be matched.

    Records sharing a key (same dataset item or input, same sample_index) are
    disambiguated in logging order (``key~2``, ``key~3``), not collapsed.
    Returns (scores by (key, metric), records by key, number of duplicates).
    """
    rows = s.exec(select(RecordRow).where(RecordRow.run_id == ev.run_id)
                  .order_by(RecordRow.created_at, RecordRow.id)).all()
    key_of: dict[str, str] = {}
    by_key: dict[str, RecordRow] = {}
    seen: dict[str, int] = defaultdict(int)
    duplicates = 0
    for r in rows:
        base = _record_key(r)
        seen[base] += 1
        key = base if seen[base] == 1 else f"{base}~{seen[base]}"
        duplicates += seen[base] > 1
        key_of[r.id] = key
        by_key[key] = r
    scores: dict[tuple[str, str], ScoreRow] = {}
    for sc in s.exec(select(ScoreRow).where(ScoreRow.evaluation_id == ev.id).order_by(ScoreRow.created_at)):
        if sc.record_id in key_of:
            scores[(key_of[sc.record_id], sc.metric)] = sc  # latest score per record and metric
    return scores, by_key, duplicates


def _compare(s: Session, base_ev: EvaluationRow, base_run: RunRow,
             head_ev: EvaluationRow, head_run: RunRow) -> dict[str, Any]:
    warnings: list[str] = []
    if base_run.dataset_hash is None or head_run.dataset_hash is None:
        warnings.append("at least one run has no dataset version; items are matched by input text")
    elif base_run.dataset_hash != head_run.dataset_hash:
        warnings.append("the runs used different dataset versions: not a like-for-like comparison")
    if base_ev.config_hash != head_ev.config_hash:
        warnings.append("the evaluations used different scoring configs (metrics, thresholds, judge or "
                        "evaluator); score changes may come from the scoring, not the model")
    for label, ev in (("base", base_ev), ("head", head_ev)):
        if ev.status != "finalized":
            warnings.append(f"the {label} evaluation is still open; its scores may be incomplete")

    differences = {}
    for field in ("model", "provider", "generation_params", "prompt_hash", "dataset_hash", "mode"):
        a, b = getattr(base_run, field), getattr(head_run, field)
        if a != b:
            differences[field] = {"base": a, "head": b}
    for field in ("evaluator", "evaluator_version", "judge_model", "config_hash"):
        a, b = getattr(base_ev, field), getattr(head_ev, field)
        if a != b:
            differences[field] = {"base": a, "head": b}

    base_scores, base_records, base_dups = _side(s, base_ev)
    head_scores, head_records, head_dups = _side(s, head_ev)
    if base_dups or head_dups:
        warnings.append(
            f"{base_dups + head_dups} record(s) share a dataset item / input and sample_index with another "
            "record in the same run; they were matched in logging order. Give them distinct dataset item "
            "ids or sample_index values for a reliable comparison."
        )
    items = []
    per_metric: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "base": [], "head": [], "base_pass": [], "head_pass": [], "regressions": 0, "improvements": 0,
        "errors": 0, "matched": 0})
    for key, metric in sorted(set(base_scores) | set(head_scores)):
        b, h = base_scores.get((key, metric)), head_scores.get((key, metric))
        stats = per_metric[metric]
        if b is not None:
            stats["base"].append(b.score)
            stats["base_pass"].append(b.passed)
        if h is not None:
            stats["head"].append(h.score)
            stats["head_pass"].append(h.passed)
        if b is None:
            status = "added"
        elif h is None:
            status = "removed"
        else:
            stats["matched"] += 1
            if b.error or h.error:
                # A judge/scoring error is not a model change; don't hide it as
                # "unchanged" or count it as a regression.
                status = "errored"
                stats["errors"] += 1
            elif b.passed is True and h.passed is False:
                status = "regressed"
            elif b.passed is False and h.passed is True:
                status = "improved"
            elif b.score is not None and h.score is not None and h.score != b.score:
                status = "worse" if h.score < b.score else "better"
            else:
                status = "unchanged"
            if status == "regressed":
                stats["regressions"] += 1
            elif status == "improved":
                stats["improvements"] += 1
        record = head_records.get(key) or base_records.get(key)
        items.append({
            "key": key,
            "metric": metric,
            "input": (record.input[:200] if record else None),
            "base_score": b.score if b else None,
            "head_score": h.score if h else None,
            "delta": (h.score - b.score) if b and h and b.score is not None and h.score is not None else None,
            "base_passed": b.passed if b else None,
            "head_passed": h.passed if h else None,
            "status": status,
            "base_record_id": b.record_id if b else None,
            "head_record_id": h.record_id if h else None,
        })

    def mean(values: list[Any]) -> float | None:
        present = [float(v) for v in values if v is not None]
        return sum(present) / len(present) if present else None

    metrics = []
    for metric, st in sorted(per_metric.items()):
        base_mean, head_mean = mean(st["base"]), mean(st["head"])
        metrics.append({
            "metric": metric,
            "base_mean": base_mean,
            "head_mean": head_mean,
            "delta": head_mean - base_mean if base_mean is not None and head_mean is not None else None,
            "base_pass_rate": mean(st["base_pass"]),
            "head_pass_rate": mean(st["head_pass"]),
            "matched": st["matched"],
            "regressions": st["regressions"],
            "improvements": st["improvements"],
            "errors": st["errors"],
        })

    def side(ev: EvaluationRow, run: RunRow) -> dict[str, Any]:
        return {"evaluation_id": ev.id, "run_id": run.id, "run_name": run.name, "model": run.model,
                "evaluator": ev.evaluator, "config_hash": ev.config_hash, "status": ev.status}

    return {
        "base": side(base_ev, base_run),
        "head": side(head_ev, head_run),
        "warnings": warnings,
        "differences": differences,
        "items": items,
        "metrics": metrics,
    }
