"""HTTP transport to the llmbenchmark backend.

Design rule: telemetry must never break the notebook. Every write method
here swallows network/server failures (and any other error while sending),
retries the retryable ones with exponential backoff, and finally appends the
payload to a local JSONL file that can be replayed later with
``llmbenchmark.replay()``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from .config import Config
from .schemas import (
    ID_PATTERN,
    MAX_BATCH_RECORDS,
    DatasetCreate,
    DatasetItem,
    EvaluationCreate,
    EvaluationFinalize,
    ItemBatch,
    MetricBatch,
    PromptVersion,
    Record,
    RecordBatch,
    RunCreate,
    RunMetric,
    RunUpdate,
    Score,
    ScoreBatch,
    Span,
    SpanBatch,
)

logger = logging.getLogger("llmbenchmark")

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

# The only requests the SDK ever writes to a fallback file. replay() refuses
# anything else, so a tampered file can't send the API key to another host
# (an absolute URL in `path` would override httpx's base_url) or hit other
# endpoints.
_ID = ID_PATTERN.strip("^$")
_REPLAYABLE = [
    ("POST", re.compile(r"^/v1/runs$")),
    ("PATCH", re.compile(rf"^/v1/runs/{_ID}$")),
    ("POST", re.compile(rf"^/v1/runs/{_ID}/(records|spans|evaluations)$")),
    ("POST", re.compile(rf"^/v1/evaluations/{_ID}/(scores|metrics|finalize)$")),
    ("POST", re.compile(r"^/v1/prompts$")),
    ("POST", re.compile(r"^/v1/datasets$")),
    ("POST", re.compile(rf"^/v1/datasets/{_ID}/items$")),
]


class APIError(Exception):
    """An explicit read or admin call failed (the backend answered >= 400).

    Raised instead of the HTTP library's own exception, so callers don't
    depend on which client (httpx, a test client, ...) is underneath.
    """

    def __init__(self, status_code: int, detail: Any, method: str, path: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{method} {path} failed with HTTP {status_code}: {detail}")


def _check(response: Any, method: str, path: str) -> Any:
    if response.status_code < 400:
        return response.json()
    try:
        detail = response.json().get("detail")
    except Exception:
        detail = response.text[:200]
    raise APIError(response.status_code, detail, method, path)


class _Outcome:
    """Result of one request with retries."""

    __slots__ = ("ok", "retryable", "error")

    def __init__(self, ok: bool, retryable: bool = False, error: str = "") -> None:
        self.ok, self.retryable, self.error = ok, retryable, error


class Transport:
    def __init__(
        self,
        config: Config,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        headers = {"User-Agent": f"llmbenchmark-python/{_sdk_version()}"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
            if _plain_http_remote(config.endpoint):
                logger.warning("llmbenchmark: sending the API key over plain http to %s; use https",
                               config.endpoint)
        self._client = client or httpx.Client(
            base_url=config.endpoint, timeout=config.timeout, headers=headers
        )
        if client is not None:
            self._client.headers.update(headers)
        self._sleep = sleep

    # ---- writes (never raise; fall back to the JSONL file) ---------------
    # Every write is keyed by its run id, so all of a run's requests land in
    # one fallback file in order and replay in that order.

    def create_run(self, run: RunCreate) -> bool:
        return self._send("POST", "/v1/runs", run, run.id)

    def update_run(self, run_id: str, update: RunUpdate) -> bool:
        return self._send("PATCH", f"/v1/runs/{_q(run_id)}", update, run_id)

    def send_records(self, run_id: str, records: list[Record]) -> bool:
        return self._send_chunks(f"/v1/runs/{_q(run_id)}/records", records, RecordBatch, "records", run_id)

    def send_spans(self, run_id: str, spans: list[Span]) -> bool:
        return self._send_chunks(f"/v1/runs/{_q(run_id)}/spans", spans, SpanBatch, "spans", run_id)

    def send_prompt(self, prompt: PromptVersion, run_id: str) -> bool:
        return self._send("POST", "/v1/prompts", prompt, run_id)

    def send_dataset(self, dataset: DatasetCreate, items: list[DatasetItem], run_id: str) -> bool:
        ok = self._send("POST", "/v1/datasets", dataset, run_id)
        return self._send_chunks(f"/v1/datasets/{_q(dataset.hash)}/items", items, ItemBatch, "items", run_id) and ok

    def create_evaluation(self, evaluation: EvaluationCreate) -> bool:
        return self._send("POST", f"/v1/runs/{_q(evaluation.run_id)}/evaluations", evaluation, evaluation.run_id)

    def send_scores(self, run_id: str, evaluation_id: str, scores: list[Score]) -> bool:
        path = f"/v1/evaluations/{_q(evaluation_id)}/scores"
        return self._send_chunks(path, scores, ScoreBatch, "scores", run_id)

    def send_metrics(self, run_id: str, evaluation_id: str, metrics: list[RunMetric]) -> bool:
        path = f"/v1/evaluations/{_q(evaluation_id)}/metrics"
        return self._send_chunks(path, metrics, MetricBatch, "metrics", run_id)

    def finalize_evaluation(self, run_id: str, evaluation_id: str,
                            manifest: EvaluationFinalize | None = None) -> bool:
        path = f"/v1/evaluations/{_q(evaluation_id)}/finalize"
        return self._send("POST", path, manifest or EvaluationFinalize(), run_id)

    # ---- explicit reads / admin calls (raise on failure) ------------------

    def get(self, path: str, **params: Any) -> Any:
        response = self._client.get(path, params={k: v for k, v in params.items() if v is not None})
        return _check(response, "GET", path)

    def get_all(self, path: str, page_size: int = 1000, **params: Any) -> list[Any]:
        """Follow limit/offset pages until a short page."""
        out: list[Any] = []
        offset = 0
        while True:
            page = self.get(path, limit=page_size, offset=offset, **params)
            out.extend(page)
            if len(page) < page_size:
                return out
            offset += page_size

    def put(self, path: str, body: BaseModel) -> Any:
        response = self._client.put(path, json=_dump(body))
        return _check(response, "PUT", path)

    def get_records(self, run_id: str, page_size: int = 1000) -> list[Record]:
        rows = self.get_all(f"/v1/runs/{_q(run_id)}/records", page_size=page_size)
        return [Record.model_validate(r) for r in rows]

    def close(self) -> None:
        self._client.close()

    # ---- internals ------------------------------------------------------

    def _send_chunks(self, path: str, items: list, batch_cls, field: str, run_id: str) -> bool:
        # A list, not a generator: every chunk is attempted (or saved) even
        # after one fails.
        results = [self._send("POST", path, batch_cls(**{field: chunk}), run_id) for chunk in _chunks(items)]
        return all(results)

    def _send(self, method: str, path: str, payload: BaseModel, run_id: str) -> bool:
        if not self.config.enabled:
            return True
        try:
            body = _dump(payload)
        except Exception as exc:  # pragma: no cover - _dump already falls back
            logger.error("llmbenchmark: could not serialize %s %s: %s", method, path, exc)
            return False
        outcome = self._request_with_retries(method, path, body)
        if outcome.ok:
            return True
        self._write_fallback(method, path, body, run_id)
        logger.warning(
            "llmbenchmark: %s %s failed (%s); saved to %s",
            method, path, outcome.error, self._fallback_path(run_id),
        )
        return False

    def _request_with_retries(self, method: str, path: str, body: Any) -> _Outcome:
        outcome = _Outcome(False, error="unknown error")
        for attempt in range(self.config.max_retries + 1):
            if attempt:
                self._sleep(min(0.5 * 2 ** (attempt - 1), 8.0))
            try:
                response = self._client.request(method, path, json=body)
            except httpx.TransportError as exc:  # DNS, refused, timeout, ...
                outcome = _Outcome(False, True, f"{type(exc).__name__}: {exc}")
                continue
            except (ValueError, TypeError) as exc:
                # The payload itself can't be encoded: retrying will never help,
                # so replay moves it to .rejected instead of looping forever.
                return _Outcome(False, False, f"{type(exc).__name__}: {exc}")
            except Exception as exc:  # e.g. client closed: never raise into the notebook
                return _Outcome(False, True, f"{type(exc).__name__}: {exc}")
            if response.status_code < 400:
                return _Outcome(True)
            retryable = response.status_code in _RETRYABLE_STATUS
            outcome = _Outcome(False, retryable, f"HTTP {response.status_code}: {response.text[:200]}")
            if not retryable:
                break  # 4xx client errors won't fix themselves
        return outcome

    def _fallback_path(self, run_id: str) -> Path:
        return self.config.fallback_dir / f"{_safe_filename(run_id)}.jsonl"

    def _write_fallback(self, method: str, path: str, body: Any, run_id: str) -> None:
        try:
            target = self._fallback_path(run_id)
            # Payloads hold full prompts and outputs: keep them private to the user.
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _tighten_dir(target.parent)
            _append_lines(target, [json.dumps({"method": method, "path": path, "body": body})])
        except Exception as exc:  # even the fallback must not raise
            logger.error("llmbenchmark: could not write fallback file: %s", exc)

    def replay(self, file: str | Path) -> tuple[int, int]:
        """Re-send requests saved in a fallback JSONL file.

        Returns (sent, failed). Lines are sent strictly in order: at the first
        failure that can heal (network error, 5xx) replay stops sending, and
        that line and every later one stay in the file for the next replay.
        (Later lines often depend on earlier ones: spans and scores need their
        records.) Lines the server rejects for good (4xx) and lines that are
        malformed or not an llmbenchmark request are moved to ``<file>.rejected``.
        Safe to repeat: the server upserts by record id.
        """
        file = Path(file)
        work = file.with_name(file.name + ".replaying")
        rejected_file = file.with_name(file.name + ".rejected")
        # Someone who can write to the fallback dir must not be able to make
        # replay read or append to other files (e.g. a .rejected symlink to
        # ~/.bashrc). Refuse symlinks and files owned by another user.
        for path in (file, work, rejected_file):
            _check_private_file(path)
        if not file.exists() and not work.exists():
            return 0, 0  # nothing was saved: every request got through
        if not self.config.enabled:
            logger.warning("llmbenchmark: disabled (LLMBENCHMARK_ENABLED=false); not replaying %s", file)
            lines = [ln for ln in _read_lines(file) if ln.strip()]
            return 0, len(lines)

        # Move the file aside first so a run that is still appending to it
        # writes to a fresh file instead of having its lines overwritten.
        if work.exists():  # an earlier replay was interrupted: merge, don't overwrite
            if file.exists():
                pending = _read_lines(file)
                if pending:
                    _append_lines(work, pending)
                file.unlink()
        else:
            file.replace(work)
        keep: list[str] = []
        rejected: list[str] = []
        sent = 0
        blocked = False
        # A fallback file is named after its run; every line must belong to it,
        # so a planted line can't act (with your key) on some other run.
        run_id = file.name.split(".jsonl")[0]
        evaluations_here: set[str] = set()
        owners: dict[str, str | None] = {}
        for number, line in enumerate(_read_lines(work), start=1):
            if not line.strip():
                continue
            if blocked:  # an earlier line must go first; keep the order
                keep.append(line)
                continue
            item = _parse_replay_line(line)
            belongs = item is not None and self._belongs_to_run(item, run_id, evaluations_here, owners)
            if belongs is None:  # couldn't check (offline): try again next time, in order
                keep.append(line)
                blocked = True
                continue
            if not belongs:
                # Don't echo the line: it may hold prompts, or be a planted file's content.
                logger.warning("llmbenchmark: skipping invalid line %d of %s", number, file.name)
                rejected.append(line)
                continue
            outcome = self._request_with_retries(item["method"], item["path"], item["body"])
            if outcome.ok:
                sent += 1
            elif outcome.retryable:
                keep.append(line)
                blocked = True
            else:
                logger.warning("llmbenchmark: server rejected replayed request (%s)", outcome.error)
                rejected.append(line)
        if keep:
            _append_lines(file, keep)
        if rejected:
            _append_lines(rejected_file, rejected)
        work.unlink()
        return sent, len(keep) + len(rejected)

    def _belongs_to_run(self, item: dict[str, Any], run_id: str, evaluations_here: set[str],
                        owners: dict[str, str | None]) -> bool | None:
        """True/False whether a replay line acts on ``run_id``; None if it can't be checked now."""
        method, path, body = item["method"], item["path"], item["body"]
        parts = path.strip("/").split("/")  # ["v1", "runs", "<id>", ...]
        if parts[1] in ("prompts", "datasets"):
            return True  # content-addressed and per-team; not tied to a run
        if parts[1] == "runs":
            target = parts[2] if len(parts) > 2 else body.get("id")
            if target != run_id:
                return False
            if parts[-1] == "evaluations":
                evaluations_here.add(str(body.get("id")))
            return True
        evaluation_id = parts[2]  # /v1/evaluations/<id>/...
        if evaluation_id in evaluations_here:
            return True
        if evaluation_id not in owners:  # created in an earlier session: ask the server
            try:
                owners[evaluation_id] = self.get(f"/v1/evaluations/{_q(evaluation_id)}").get("run_id")
            except APIError:
                owners[evaluation_id] = None
            except Exception:
                return None
        return owners[evaluation_id] == run_id


def _parse_replay_line(line: str) -> dict[str, Any] | None:
    try:
        item = json.loads(line)
    except ValueError:
        return None
    if not isinstance(item, dict) or not isinstance(item.get("body"), dict):
        return None
    method, path = item.get("method"), item.get("path")
    if not isinstance(method, str) or not isinstance(path, str):
        return None
    if not any(method == m and rx.fullmatch(path) for m, rx in _REPLAYABLE):
        return None
    return item


_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)  # POSIX: refuse to open through a symlink


def _append_lines(target: Path, lines: list[str]) -> None:
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND | _O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _read_lines(path: Path) -> list[str]:
    fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as fh:
        return fh.read().splitlines()


def _check_private_file(path: Path) -> None:
    """Raise if ``path`` is a symlink or owned by someone else (replay is explicit, so it may raise)."""
    if path.is_symlink():
        raise ValueError(f"refusing to replay: {path} is a symlink")
    if path.exists() and hasattr(os, "getuid") and path.stat().st_uid != os.getuid():
        raise ValueError(f"refusing to replay: {path} is owned by another user")


def _tighten_dir(path: Path) -> None:
    """Make an existing fallback dir private too (mkdir's mode only applies when creating)."""
    try:
        if hasattr(os, "getuid") and path.stat().st_uid == os.getuid():
            os.chmod(path, 0o700)
    except OSError:
        pass


def _dump(payload: BaseModel) -> dict[str, Any]:
    """JSON-ready dict; values pydantic can't serialize become strings, and
    NaN/inf become null (JSON has no such numbers; httpx refuses to send them)."""
    try:
        data = payload.model_dump(mode="json")
    except Exception:
        data = json.loads(json.dumps(payload.model_dump(), default=str))
    return _finite(data)


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _chunks(items: list) -> list[list]:
    return [items[i : i + MAX_BATCH_RECORDS] for i in range(0, len(items), MAX_BATCH_RECORDS)]


def _plain_http_remote(endpoint: str) -> bool:
    from urllib.parse import urlsplit

    parts = urlsplit(endpoint)
    return parts.scheme == "http" and parts.hostname not in ("localhost", "127.0.0.1", "::1", "testserver")


def _q(run_id: str) -> str:
    return quote(run_id, safe="")


def _safe_filename(run_id: str) -> str:
    # Never let an id like "../../x" leave fallback_dir.
    return re.sub(r"[^A-Za-z0-9_-]", "_", run_id)[:128] or "_"


def _sdk_version() -> str:
    from . import __version__

    return __version__
