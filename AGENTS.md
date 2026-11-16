# AGENTS.md — `llmbenchmark` replication spec

This file is written for coding agents (Claude Code, Codex, Cursor, Copilot, ...).
It describes the `llmbenchmark` project precisely enough to rebuild it from an
empty directory, and it is the working guide for changing the existing code.

- **§1–3**: what the project is, its stack, and its layout
- **§4–9**: module-by-module spec (SDK core, eval)
- **§10**: backend and REST API
- **§11–12**: tests, notebooks, docs
- **§13**: build order with verification checkpoints
- **§14**: guarantees and the tests that enforce them; known limitations
- **§15**: conventions and gotchas

---

## 1. What the project is

`llmbenchmark` is a Python SDK, used from Jupyter notebooks, for benchmarking LLMs and agents.

1. It **records** single prompts (with or without a system prompt), multi-turn conversations, and **agent traces** (every LLM call, tool call, and sub-step as a span tree).
2. It **ships them in batches over HTTP** to a team-isolated backend (SQLite or Postgres).
3. It **evaluates** them in one of two ways:
   - **managed**: `llmbenchmark.eval` runs DeepEval metrics with a Claude judge;
   - **bring your own (BYO)**: teams upload per-example scores and run-level numbers from their own methods.
4. It **versions results**:
   - prompts and datasets are content-hashed;
   - runs pin lineage;
   - every scoring pass is an immutable Evaluation (re-scoring creates a new one);
   - runs can be compared item by item against a baseline.

```
Prompt@hash ─┐                 ┌─ Evaluation v1 (evaluator, metrics, judge, config_hash) ─ Scores, RunMetrics
Dataset@hash ┼─► Run ─► Records┤
model+params ┘        └─ Spans └─ Evaluation v2 (supersedes v1)                         ─ Scores, RunMetrics
                  (agent traces)

notebook ─► llmbenchmark ─(batched HTTP, retries, JSONL fallback)─► backend (FastAPI, per-team) ─► DB
```

| Part | Path | Shipped in wheel? | Deps |
|---|---|---|---|
| Core SDK | `src/llmbenchmark/` | yes | `httpx`, `pydantic` only |
| Eval extra | `src/llmbenchmark/eval/` | yes (needs `[eval]` to import) | `deepeval`, `anthropic`, `pandas`, `ipywidgets` |
| Reference backend | `server/` | **no** | `fastapi`, `uvicorn`, `sqlmodel` (`[server]`) |

### Glossary

| Term | Meaning |
|---|---|
| Run | One experiment: one model + generation params + prompt version over one dataset version. `mode` is `managed` or `external` (BYO) |
| Record | One example. `kind` is `single`, `conversation`, or `agent` |
| Trace / Span | One agent task / one step in it (`agent`, `llm`, `tool`, `retrieval`, `handoff`, `other`), forming a tree |
| Prompt, Dataset | Versioned inputs identified by the sha256 of their content |
| Evaluation | One immutable scoring pass over a run: evaluator name+version, metric specs, judge model, `config_hash`, optional `supersedes` |
| Score / RunMetric | A per-record verdict / a run-level number, both belonging to an Evaluation |
| Baseline | Per (team, project): the evaluation new runs are compared against |

### Design invariants (preserve them)

1. **Telemetry never breaks the notebook.** Nothing raised while *recording or sending* may reach user code: network errors, server errors, a closed client, unserializable values, odd output types, or span-building failures. On failure: retry with backoff, then append the payload to a local JSONL file, log a warning, and return `False`. Explicit reads (`compare`, `get_baseline`, `get_records`) raise `llmbenchmark.APIError`. Explicit misuse (bad config, a malformed Dataset, scoring a finalized evaluation) raises.
2. **Idempotent, non-destructive writes.** Every record has a client-generated id, and the server upserts by id. An id that belongs to another run, evaluation, team, or project is a 409, never an overwrite.
3. **Immutability.** A finished run and a finalized evaluation are frozen. Closing sends a **manifest** of every id the client logged; afterwards the server accepts identical resends and late deliveries of manifest ids (retries/replays) and refuses everything else (409). **No client timestamp or client-set status is ever trusted**: close times are server time, and a run is always created `running`. Re-scoring = a new Evaluation. Logging to a finished run raises (it could never be stored).
4. **Tiny core.** The core imports neither `anthropic`, `openai`, `deepeval`, nor `pandas` (pandas only lazily in `to_dataframe`/`summary`). Provider responses are duck-typed.
5. **Small public API.** Only `llmbenchmark.__all__` and `llmbenchmark.eval.__all__` are public.
6. **Notebook-safe.** No `asyncio.run()` on the notebook's thread. DeepEval runs in a worker thread when a loop is already running.
7. **Lazy credentials.** Building a judge never needs an API key.
8. **Secure by default.**
   - The server refuses to start without keys unless anonymous mode is explicitly set.
   - Keys are stored as sha256 hashes; each key belongs to one team and an explicit role (no default role; duplicate keys refused at startup).
   - Every query is team-filtered, and another team's data is a 404.
   - Secrets never appear in reprs or logs; the SDK warns when an API key would travel over plain http to a non-local host.
   - Untrusted local state can't escalate: replay refuses symlinked/foreign-owned files and lines that don't belong to the file's run; lineage capture disables repo-configured git hooks.
   - Request bodies are bounded in size, batch length and JSON nesting depth.
   - Local files holding prompts are `0600`/`0700`.

---

## 2. Tech stack (reference environment, verified working)

| Component | Version verified | Declared constraint |
|---|---|---|
| Python | 3.14.7 | `>=3.10` |
| httpx | 0.28.1 | `>=0.27` |
| pydantic | 2.14.3 | `>=2.6` |
| deepeval | 4.3.3 | `>=3.0` |
| anthropic | 2.7.0 | `>=0.70` |
| pandas | 3.1.1 | `>=2.0` |
| fastapi | 0.162.2 (Starlette's `TestClient` subclasses **`httpx2.Client`**, not `httpx.Client`) | `>=0.110` |
| uvicorn | 0.54.0 | `>=0.29` |
| sqlmodel | 0.0.48 (SQLAlchemy 2.1.8) | `>=0.0.21` |
| pytest | 9.2.0 | `>=8` |
| respx | 0.23.1 | `>=0.21` |
| openai (dev only) | 3.48.0 | `>=1.40` |

Build backend: **hatchling**. Package manager: pip+venv or **uv**. Model ids: `claude-opus-5-5` (under test), `claude-sonnet-5-5` (default judge), `gpt-5`, `gpt-4.1`.

---

## 3. Repository layout

```
.
├── AGENTS.md  README.md  pyproject.toml  .gitignore
├── docs/CONCEPTS.md           # Part 1 SDK design, Part 2 DeepEval, Part 3 data model/versioning/teams
├── examples/
│   ├── quickstart.ipynb       # 19 cells: Claude vs GPT, versioned prompt/dataset, judge eval, compare (needs keys)
│   └── tutorial.ipynb         # 50 cells, steps 0–13; runs WITHOUT keys (mock mode); see §12
├── src/llmbenchmark/
│   ├── __init__.py            # public API, module singletons, atexit flush          (~150 lines)
│   ├── config.py              # Config + env loading/validation                       (~95)
│   ├── schemas.py             # wire models (SCHEMA_VERSION = 1)                       (~365)
│   ├── versioning.py          # content_hash(), capture_lineage()                      (~75)
│   ├── assets.py              # Prompt, Dataset                                        (~225)
│   ├── transport.py           # HTTP, retries, fallback, replay, APIError              (~445)
│   ├── run.py                 # Run: log/track/trace/evaluation/log_results            (~660)
│   ├── tracing.py             # span(), observe(), propagate(), contextvars            (~245)
│   ├── evaluation.py          # Evaluation (versioned scoring pass), summarize()       (~330)
│   ├── compare.py             # compare(), Comparison, set/get_baseline                (~105)
│   ├── providers.py           # duck-typed extraction incl. tool calls                 (~120)
│   └── eval/
│       ├── __init__.py        # deepeval import guard + re-exports
│       ├── adapters.py        # Record/Span → LLMTestCase / ConversationalTestCase / trace dict
│       ├── judges.py          # ClaudeJudge, judge()
│       ├── metrics.py         # ExactMatch, ExpectedTools, correctness()
│       └── runner.py          # evaluate_run(), EvalReport
├── server/
│   ├── __init__.py            # empty
│   ├── auth.py                # Principal, KeyStore, load_key_store()
│   ├── keys.py                # CLI: python -m server.keys create|list
│   ├── db.py                  # SQLModel tables (every table has `team`), make_engine()
│   └── app.py                 # create_app(): auth, limits, routes, comparison logic   (~810)
└── tests/                     # 156 tests, ~6 s, no network
    conftest.py · test_config (8) · test_providers (4) · test_transport (28) · test_run (27)
    test_tracing (10) · test_evaluation (18) · test_compare (8) · test_server (32) · test_eval (21)
```

### 3.1 `pyproject.toml`

Settings:
- `[build-system]`: `requires = ["hatchling>=1.25"]`, `build-backend = "hatchling.build"`
- `[project]`: `name = "llmbenchmark"`, `dynamic = ["version"]`, `readme = "README.md"`, `requires-python = ">=3.10"`, `authors = [{ name = "llmbenchmark contributors" }]` (never a personal name: the repo is public)
- `description = "Record LLM calls, conversations and agent traces, evaluate them (your way or with DeepEval), and keep versioned, comparable results."`
- `dependencies = ["httpx>=0.27", "pydantic>=2.6"]`
- Extras: `eval = ["deepeval>=3.0", "anthropic>=0.70", "pandas>=2.0", "ipywidgets>=8"]`, `server = ["fastapi>=0.110", "uvicorn>=0.29", "sqlmodel>=0.0.21"]`
- `[dependency-groups] dev = ["llmbenchmark[eval,server]", "pytest>=8", "respx>=0.21", "openai>=1.40", "ipykernel>=6"]`
- `[tool.hatch.version] path = "src/llmbenchmark/__init__.py"`; `[tool.hatch.build.targets.wheel] packages = ["src/llmbenchmark"]`
- `[tool.pytest.ini_options] testpaths = ["tests"]`, `pythonpath = ["."]`, `asyncio_default_fixture_loop_scope = "function"`, `filterwarnings = ["ignore::DeprecationWarning:deepeval.*"]`

Rules: lower bounds only, never `==` pins. The version is single-sourced from `__version__ = "0.1.0"`.

### 3.2 `.gitignore`

`.venv/ __pycache__/ *.pyc .pytest_cache/ dist/ build/ *.egg-info/ .ipynb_checkpoints/ .llmbenchmark/ llmbenchmark.db .deepeval/ .env`, plus keys files you create locally (never commit a keys file).

---

## 4. `config.py`

`Config` dataclass fields:

| Field | Default |
|---|---|
| `endpoint` | `"http://localhost:8000"` |
| `api_key` | `field(default=None, repr=False)` |
| `project` | `"default"` |
| `batch_size` | `50` |
| `timeout` | `10.0` |
| `max_retries` | `3` |
| `fallback_dir` | `Path(".llmbenchmark")` |
| `enabled` | `True` |
| `capture_git` | `True` |

`load_config(**overrides)` works in this order:
1. Reject unknown names with `TypeError`.
2. Read `LLMBENCHMARK_<NAME>` through an **explicit converter table**. `_to_bool` accepts `1/true/yes/on` and `0/false/no/off/""`; anything else raises `ValueError`. A conversion failure becomes `ValueError(f"{VAR}={raw!r} is invalid: …")`.
3. Apply non-None overrides.
4. Coerce `fallback_dir` to a `Path` and strip a trailing `/` from `endpoint`.
5. Validate `batch_size >= 1`, `max_retries >= 0`, and `timeout > 0`.

---

## 5. `schemas.py` (the wire contract; the server imports it)

Constants:
- `SCHEMA_VERSION = 1` (bump on breaking wire changes)
- `ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"` and `Id = Annotated[str, Field(pattern=ID_PATTERN)]`
- `MAX_BATCH_RECORDS = 1000`
- `MAX_JSON_DEPTH = 32` (the SDK cuts free-form JSON values deeper than this, replacing the rest with `TRUNCATED = "<nested too deeply: truncated by llmbenchmark>"`) and `SERVER_MAX_JSON_DEPTH = 64` (the server refuses deeper bodies)
- Literals: `RunStatus` (running/completed/failed), `RunMode` (managed/external), `RecordKind` (single/conversation/agent), `Role` (system/user/assistant/tool), `SpanKind` (agent/llm/tool/retrieval/handoff/other), `EvalSource` (llmbenchmark/external), `EvalStatus` (open/finalized)

Helpers: `new_id()` (uuid4 str), `utcnow()`, `is_valid_id()`, `limit_depth(value, max_depth)` (recursion bounded by the limit, so safe on absurdly deep input, unlike `json.dumps`), `json_depth(value)` (iterative).

| Model | Fields |
|---|---|
| `ToolCall` | `name`, `arguments: dict = {}`, `output: Any = None`, `id?` |
| `Message` | `role: Role`, `content: str = ""`, `name?`, `tool_calls?: list[ToolCall]` |
| `PromptRef` | `name`, `version?`, `hash: Id` |
| `DatasetRef` | `name`, `version?`, `hash: Id`, `item_count?` |
| `Lineage` | `sdk_version?`, `schema_version = 1`, `python?`, `packages: dict[str,str]`, `git_commit?`, `git_dirty?`, `incomplete: list[str]` |
| `RunCreate` | `id: Id`, `project`, `name?`, `mode = "managed"`, `model?`, `provider?`, `generation_params: dict`, `prompt?: PromptRef`, `dataset?: DatasetRef`, `tags: list[str]`, `status = "running"`, `started_at`, `ended_at?`, `lineage: Lineage` |
| `RunInfo(RunCreate)` | + `team?`, `created_by?` (server-assigned; never trusted from clients) |
| `RunUpdate` | `status`, `record_ids: list[Id]`, `span_ids: list[Id]` (the close **manifest**; no timestamp: the server stamps `ended_at`) |
| `Record` | `id`, `run_id`, `kind = "single"`, `dataset_item_id?: Id`, `sample_index = 0`, `system?`, `input: str` (always set), `messages?: list[Message]`, `output?`, `expected_output?`, `expected_tools?: list[str]`, `tools_called?: list[ToolCall]` (`[]` = called none; `None` = unknown), `context?`, `retrieval_context?`, `trace_id?`, `provider?`, `model?`, `input_tokens?`, `output_tokens?`, `latency_ms?`, `error?`, `metadata: dict`, `tags: list`, `created_at` |
| `Span` | `id`, `run_id`, `trace_id`, `record_id`, `parent_id?`, `kind = "other"`, `name`, `input: Any`, `output: Any`, `provider?`, `model?`, `input_tokens?`, `output_tokens?`, `started_at`, `ended_at?`, `error?`, `attributes: dict` |
| `PromptVersion` | `hash`, `name`, `version?`, `system?`, `template?` |
| `DatasetItem` | `id: Id`, `input`, `system?`, `messages?`, `expected_output?`, `expected_tools?`, `context?`, `retrieval_context?`, `metadata: dict` |
| `DatasetCreate` | `hash`, `name`, `version?`, `item_count: int = Field(ge=0)` |
| `DatasetInfo(DatasetCreate)` | + `verified: bool` (all items received and the hash recomputed from them) |
| `MetricSpec` | `name`, `threshold?`, `criteria?`, `params: dict` |
| `EvaluationCreate` | `id`, `run_id`, `source = "external"`, `evaluator`, `evaluator_version?`, `judge_model?`, `metrics: list[MetricSpec]`, `config_hash: Id`, `supersedes?: Id`, `created_at` |
| `EvaluationFinalize` | `score_ids: list[Id]`, `metric_ids: list[Id]` (manifest; the server stamps `finalized_at`) |
| `EvaluationInfo(EvaluationCreate)` | + `status`, `finalized_at?`, `superseded_by?`, `team?`, `created_by?` |
| `Score` | `id`, `evaluation_id`, `record_id`, `metric`, `score?: float`, `passed?`, `threshold?`, `reason?`, `error?`, `created_at` |
| `RunMetric` | `id`, `evaluation_id`, `name`, `value: float`, `n?`, `details: dict`, `created_at` |
| `BaselineSet` | `project`, `run_id`, `evaluation_id` |

Batches: `RecordBatch{records}`, `SpanBatch{spans}`, `ScoreBatch{scores}`, `MetricBatch{metrics}`, `ItemBatch{items}`.

Keep size limits **out of** shared models: the server enforces them. Otherwise the SDK would raise in the notebook.

---

## 6. `versioning.py` and `assets.py`

`versioning.py`:
- `content_hash(obj)`: the sha256 hex of `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)`.
- `capture_lineage(*, sdk_version, capture_git)`:
  - **never raises**;
  - fills `python` and `packages` (llmbenchmark, anthropic, openai, deepeval, httpx, pydantic, via `importlib.metadata`);
  - when `capture_git` is set, fills `git_commit` (`git rev-parse HEAD`) and `git_dirty` (non-empty `git status --porcelain`), using `subprocess.run(timeout=3)`. Any failure → None.

`assets.py` (explicit user code, so invalid input **raises**):

**`Prompt(name, *, system=None, template=None, version=None)`**
- `hash = content_hash({"system", "template"})`. Name and version are labels; they don't affect the hash.
- `variables` = the `str.format` placeholders in the template.
- `render(item | mapping | str | None, **vars)`:
  - the values are the item's `metadata`, plus `input`, plus explicit vars;
  - with no template, it returns `input`;
  - missing variables raise `KeyError`.
- `messages(item, *, include_system=False, **vars)` returns `[{"role": "user", "content": render(...)}]`. The system message is prepended only with `include_system` (OpenAI); Anthropic takes `system=` separately.
- `ref()` → `PromptRef`; `to_wire()` → `PromptVersion`.

**`Dataset(name, items, *, version=None)`**
- Items are dicts or `DatasetItem`s. A dict needs `input`.
- A missing id becomes `content_hash({input, system, messages})[:32]`. An invalid id raises `ValueError`, and so do duplicate ids.
- Unknown dict keys move into `metadata`.
- `hash = content_hash([item.model_dump(mode="json") ...])`.
- `from_records(rows | DataFrame, name, *, version, input_col="input", expected_col="expected_output", id_col="id", system_col=None)`.
- Iterable and indexable; `__len__`; `ref()`; `to_wire()` → `DatasetCreate`.

---

## 7. `transport.py`

`Transport(config, client=None, sleep=time.sleep)`:
- Headers: `User-Agent: llmbenchmark-python/<__version__>`, plus `Authorization: Bearer <api_key>` when a key is set. If that key would go over `http://` to a host other than localhost/127.0.0.1/::1/testserver, log a warning.
- Without an injected client it builds `httpx.Client(base_url=config.endpoint, timeout=config.timeout, headers=headers)`. An injected client (tests) gets `client.headers.update(headers)`. Only `request()`, `get()`, `put()` and `close()` are used, so any httpx-compatible client works.
- `sleep` is injectable so tests don't wait.
- Every id placed in a URL is `urllib.parse.quote(id, safe="")`-escaped.
- `_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}`.

**Writes** never raise. Each returns `bool` and is keyed by `run_id`, so all of a run's requests land in one fallback file **in order**:

| Method | Request |
|---|---|
| `create_run(RunCreate)` | `POST /v1/runs` |
| `update_run(run_id, RunUpdate)` | `PATCH /v1/runs/{id}` |
| `send_records(run_id, records)` | `POST /v1/runs/{id}/records` |
| `send_spans(run_id, spans)` | `POST /v1/runs/{id}/spans` |
| `send_prompt(PromptVersion, run_id)` | `POST /v1/prompts` |
| `send_dataset(DatasetCreate, items, run_id)` | `POST /v1/datasets`, then `POST /v1/datasets/{hash}/items` |
| `create_evaluation(EvaluationCreate)` | `POST /v1/runs/{run_id}/evaluations` |
| `send_scores(run_id, eval_id, scores)` | `POST /v1/evaluations/{id}/scores` |
| `send_metrics(run_id, eval_id, metrics)` | `POST /v1/evaluations/{id}/metrics` |
| `finalize_evaluation(run_id, eval_id, manifest)` | `POST /v1/evaluations/{id}/finalize`, body `EvaluationFinalize` |

`_send_chunks` splits a list into `MAX_BATCH_RECORDS` chunks and sends **every** chunk (a list comprehension, then `all()`). An empty list → `True`, no request.

**Explicit reads/admin** raise `APIError(status_code, detail, method, path)` on any status ≥ 400, via `_check(response)`. Never let `httpx`/`httpx2` exceptions leak.
- `get(path, **params)`
- `get_all(path, page_size=1000, **params)` (follows `limit`/`offset` until a short page)
- `put(path, body)`
- `get_records(run_id)`

**`_send(method, path, payload, run_id) -> bool`**
1. If `config.enabled` is False, return `True` without a request.
2. `body = _dump(payload)`: `model_dump(mode="json")`, falling back to `json.loads(json.dumps(model_dump(), default=str))` when pydantic can't serialize a value; then `_finite()` turns every NaN/±inf float into `null` (JSON has no such numbers, and httpx refuses to encode them).
3. `_request_with_retries(...)`. On success, return `True`.
4. Otherwise `_write_fallback(...)`, then `logger.warning("llmbenchmark: %s %s failed (%s); saved to %s", ...)`, and return `False`.

**`_request_with_retries(method, path, body) -> _Outcome(ok, retryable, error)`**
- Up to `max_retries + 1` attempts. Before each retry, sleep `min(0.5 * 2**(attempt-1), 8.0)` (0.5, 1, 2, 4, 8, 8 …).
- `httpx.TransportError` (DNS, refused, timeout) → retryable; try again.
- `ValueError`/`TypeError` (the payload can't be encoded) → a **non-retryable** failure, so replay moves it to `.rejected` instead of retrying it forever.
- **Any other exception** (e.g. a closed client) → return a retryable failure immediately. Never raise.
- Status < 400 → ok. A status in `_RETRYABLE_STATUS` → retry. Any other 4xx → stop (it won't fix itself).

**`_write_fallback(method, path, body, run_id)`**
- Path: `config.fallback_dir / f"{_safe_filename(run_id)}.jsonl"`. `_safe_filename` replaces every char outside `[A-Za-z0-9_-]` with `_` and truncates to 128, so `"../../x"` can't leave the dir.
- `mkdir(mode=0o700, parents=True, exist_ok=True)`, then `_tighten_dir` (chmod `0700` if we own it: mkdir's mode only applies when creating), then `_append_lines` (`os.open(path, O_WRONLY|O_CREAT|O_APPEND|O_NOFOLLOW, 0o600)`).
- Appends one line: `{"method", "path", "body"}`. Any exception is logged with `logger.error` and swallowed.

**`replay(file) -> (sent, failed)`**
1. `_check_private_file` on `file`, `<file>.replaying` and `<file>.rejected`: a **symlink or a file owned by another user → `ValueError`** (replay is explicit, so it may raise). All reads/appends use `O_NOFOLLOW`.
2. If neither `file` nor `.replaying` exists → `(0, 0)`. If `enabled` is False: log a warning and return `(0, <non-blank line count>)` without sending or touching the file.
3. Move the file to `<file>.replaying` (`Path.replace`), so a run still appending writes to a fresh file. If a `.replaying` file already exists (an interrupted replay), append the file's lines to it (if the file exists) and delete the file; never overwrite.
4. Go through the lines **strictly in order**:
   - after the first line that fails in a way that can heal, every later line is kept unsent (spans and scores depend on earlier records);
   - a line must parse (JSON object, dict `body`, `(method, path)` on the allow-list below) **and belong to the file's run** (`run_id` = the file name before `.jsonl`): `/v1/runs` lines must target that run id; `/v1/evaluations/<id>/…` lines must refer to an evaluation created earlier in the same file, or one the server reports (`GET /v1/evaluations/<id>`) as belonging to the run; prompts and datasets are allowed. If that GET can't be made (offline), the line is kept and replay stops. Otherwise a non-matching line is **rejected** and never sent;
   - invalid lines are logged by number only (never their content).
5. Send accepted lines with `_request_with_retries` (no new fallback write). Success → `sent += 1`. A retryable failure keeps the line (and blocks the rest). A non-retryable one is rejected.
6. Append the kept lines back to `file`, and the rejected ones to `<file>.rejected` (both `0600`). Delete `.replaying`. Return `(sent, kept + rejected)`.

**Replay allow-list** (`<id>` = `[A-Za-z0-9_-]{1,128}`). Anything else is never sent:

```
POST  ^/v1/runs$                                 PATCH ^/v1/runs/<id>$
POST  ^/v1/runs/<id>/(records|spans|evaluations)$
POST  ^/v1/evaluations/<id>/(scores|metrics|finalize)$
POST  ^/v1/prompts$   POST ^/v1/datasets$   POST ^/v1/datasets/<id>/items$
```

---

## 8. Recording: `run.py`, `tracing.py`, `evaluation.py`, `compare.py`, `providers.py`, `__init__.py`

### 8.1 `Run(transport, config, *, name, project, mode="managed", model, provider, generation_params, prompt: Prompt, dataset: Dataset, tags)`

**Constructor**
- Type-checks `prompt` and `dataset`.
- Builds `info: RunCreate`: `generation_params` go through `_jsonable`, `prompt`/`dataset` become refs, and `lineage = capture_lineage(...)`.
- **Managed mode**: missing `model`/`generation_params`/`prompt` → `lineage.incomplete = [...]`, plus a `logger.warning` naming them and suggesting `mode='external'`. It never raises.
- State: `records`, `spans: dict[trace_id, list[Span]]`, `evaluations`, `_buffer`, `_lock`, `_send_lock` (RLock), `_finished`.

**Lifecycle**
- `start()` sends the prompt, then the dataset with its items, then `create_run`.
- `finish(status)` is idempotent: `flush()`, then `flush_evaluations()`, then `update_run(RunUpdate(status, record_ids=[every logged record], span_ids=[every span]))`. `info.ended_at` is set locally for display only.
- `flush_evaluations()` sends buffered scores of evaluations never finalized (warning; they stay open). Also called at exit.
- `__exit__` finishes with `failed` when an exception escapes, otherwise `completed`.
- `flush()` holds `_send_lock` from swapping the buffer (under `_lock`) until the batch is sent or saved. Anything that depends on earlier records (a trace's spans, an evaluation's scores) flushes first and so waits for in-flight records; this also keeps the fallback file in order. Without it, parallel traces send spans before their records.

**`log(*, input, output, response, system, messages, item, expected_output, expected_tools, tools_called, context, retrieval_context, model, provider, input_tokens, output_tokens, latency_ms, error, metadata, tags, sample_index=0, kind=None, trace_id=None, id=None) -> Record`**

Raises `RuntimeError` on a finished run (the record could never be stored).

| Step | Behavior |
|---|---|
| Item | A dict item goes through `assets._to_item`. The item fills `input`, `system`, `messages`, `expected_output`, `expected_tools`, `context`, `retrieval_context`, and `dataset_item_id`. Explicit arguments win |
| Messages | `_messages` normalizes them: dict or `Message`; content-block lists are joined (`text`/`input_text`/`output_text`); role aliases `developer→system`, `human→user`, `ai`/`model→assistant`; unknown roles → `user`; OpenAI `tool_calls` are parsed |
| System and input | `system` defaults to the joined system messages; a content-block list (Anthropic format) is joined to text. `input` defaults to the last user message. With no input at all → `ValueError` |
| Coercion | tokens → `_int_or_none`; `latency_ms` → `_float_or_none`; `sample_index` → int (default 0); `tags`/`expected_tools` → `_str_list` (drops `None`); `model`/`provider` → str; NaN / `pd.NA` / `NaT` (`assets.is_missing`) → missing in text fields |
| Response | `providers.extract(response)`. Explicit `model`/`provider`/`output`/tokens win. `tools_called` comes from the explicit argument, else extraction |
| Metadata | `metadata = _jsonable({**extracted.metadata, **metadata})`; `output` → `_opt_text` |
| Model and provider | Fall back to the run's `model`/`provider` |
| Kind | `agent` if `trace_id`, else `conversation` if any assistant message, else `single` |
| Buffer | The record is appended under the lock; the buffer is flushed when full |

**`track(...)`** is a context manager yielding `TrackedCall(response, output, tools_called, metadata, record)`.
- Same inputs as `log`, plus `raise_errors=True`.
- It catches `BaseException` to record it, and re-raises unless `raise_errors=False`. A non-`Exception` (`KeyboardInterrupt`) is always re-raised.
- The record is built with `_safe_log`: on a finished run it logs an error and records nothing; if `log` fails, it retries keeping `output`/`model`/`provider`/tokens, then a minimal record, each with `error="…could not record this call…"`, so the user's exception is never masked.

**`trace(*, input, system, item, expected_output, expected_tools, name="agent", metadata, tags, sample_index, raise_errors=True)`** is a context manager yielding `TraceHandle(trace_id, record_id, output, metadata, record, spans)`.
- Sets the `_current_trace` and `_current_span` contextvars, and opens a root `span(name, kind="agent")`.
- On exit:
  - closes the trace state (`state.close()`) and collects the spans; a span finishing later is dropped with a warning;
  - `tools_called` comes from the tool spans in start order (`arguments` = the span's input if it's a dict, else `{"input": …}`) and is **`[]` when there are none**;
  - `model` is the first LLM span's model, and tokens are summed over LLM spans;
  - `_safe_log(kind="agent", id=record_id, trace_id=…)`;
  - then **`flush()` before `send_spans`**, because spans reference the record.

**`evaluation(evaluator, *, version, metrics, judge_model, supersedes)`** → a started `Evaluation` (`source="external"`, `before_send=self.flush`). Allowed on finished runs (evaluating afterwards is the normal flow). `supersedes` goes through `resolve_supersedes` (§8.3).

**`log_results(rows, *, evaluator, scores, version, input_col, output_col, expected_col, system_col, item_id_col, reason_cols, metrics, aggregates)`**
- Bulk BYO upload from a list of dicts or a DataFrame.
- `scores` maps metric name → column, or is a list of column names.
- Raises `RuntimeError` on a finished run (call it inside the run's `with` block).
- Rows go through `assets._rows` (empty DataFrame cells → `None`). Logs each row and scores it; `aggregates` become `ev.metric(..., n=<number of rows>)`.
- Returns the **finalized** Evaluation.

Helpers: `_to_text`, `_opt_text`, `_opt_texts`, `_jsonable`, `_jsonable_value`, `_content_text`, `_messages`, `_tool_calls` (`None` stays `None`; `[]` stays `[]`; handles the OpenAI `{"function": {...}}` shape), and `_sum`.

### 8.2 `tracing.py`

- `SpanHandle`: `id`, `name`, `kind`, `input`, `output`, `response` (for LLM spans), `model`, `provider`, `input_tokens`, `output_tokens`, `attributes`.
- `span(name, *, kind="other", input=None, attributes=None)`:
  - outside a trace it is a **no-op** (debug log);
  - it records the parent from `_current_span`, sets itself as current, and records `error` and re-raises on exception;
  - in `finally` it builds the `Span` via `_build_span`. `providers.extract(response)` fills `model`, `tokens`, and `output` when not set; values go through `_jsonable_value`. A failure is logged, never raised.
- `observe(name=None, *, kind="other")` is a decorator for sync and async functions and for sync/async **generators** (the span covers the whole iteration; output `{"yielded": n}`). The span input is the bound arguments by name (minus `self`); the output is the return value.
- `_TraceState.add()` returns False once `close()` was called (the trace ended); `span` then logs a warning that the span was not recorded.
- `propagate(fn)` captures the caller's context. Each call runs in `context.copy().run(...)`, so concurrent threads never share one context.
- asyncio tasks inherit contexts automatically. Plain threads start empty, so un-propagated spans are no-ops.

### 8.3 `evaluation.py`: `Evaluation(transport, run_id, *, evaluator, version, metrics, judge_model, supersedes, source="external", before_send, batch_size=50, upload=True)`

**Metric specs**
- `metrics` may be a `{name: {threshold, criteria, ...}}` mapping, or a list of `MetricSpec`, dicts, or names.
- **Unknown keys go into `params`**; they are never dropped.

**Config hash**: `config_hash = content_hash({evaluator, evaluator_version, judge_model, metrics})`.

**Lifecycle**
- `start()` runs `before_send` (flushing the run's records), then `create_evaluation`.
- As a context manager: a clean exit → `finalize()`. An exception → `flush()`, then the evaluation is **left open** with a warning.
- `finalize()` flushes, then `finalize_evaluation(..., EvaluationFinalize(score_ids=[…], metric_ids=[…]))`; it is idempotent.
- `flush()` runs `before_send` first, so scores never reference unsent records.

**`score(record | id, metric, score=None, *, passed, threshold, reason, error)`**

| Input | Effect |
|---|---|
| Finalized evaluation | `RuntimeError` |
| Boolean score (Python or numpy) | Stored as 1.0/0.0, and sets `passed` |
| Non-numeric, NaN or ±inf score | `score=None` and `error="score is not a finite number: …"` (`_to_float` returns only finite floats) |
| Threshold | Coerced with `_to_float` (e.g. `"0.5"` → 0.5); if not given, taken from the metric spec (spec thresholds must be finite, else `ValueError`) |
| No `passed` given | `passed = score >= threshold` when both are known |

Scores are buffered and flushed at `batch_size`.

**`metric(name, value, *, n, details)`** records a run-level number. A non-finite or non-numeric value → `ValueError`. `details` go through `_jsonable`.

**`resolve_supersedes(value, run_id, transport)`**: accepts an Evaluation, an EvalReport (`.evaluation_id` + `.run_id`) or an id. The superseded evaluation must belong to `run_id` (a bare id is looked up with `GET /v1/evaluations/<id>`; 404 → `ValueError`; offline → warning, the server still checks). A mismatch raises `ValueError` locally instead of failing on the server, which would send the whole new evaluation to the fallback file.

**State**: `uploaded` = AND of every send result. `summary()` = `summarize(scores)`: per metric, `mean_score`, `pass_rate` (over non-null), `n`, `errors`.

### 8.4 `compare.py`

These are explicit reads and raise `APIError`:
- `compare(base=None, head)`: `GET /v1/compare?head=&base=`. Without `base`, the project's baseline is used. Returns `Comparison(base, head, warnings, differences, items, metrics)` with `regressions`, `improvements`, `summary()` (indexed by metric), `to_dataframe()`.
- `set_baseline(evaluation, *, project=None)`: looks up the evaluation's run (and its project), then `PUT /v1/baselines`.
- `get_baseline(project=None)` → a dict, or `None` on 404.
- Accepted ids: a string, or any object with `.evaluation_id` or `.id`, validated with `is_valid_id`.

### 8.5 `providers.py`

Duck-typed on purpose: the core never imports `anthropic` or `openai`. `_get(obj, name)` returns `None` for None, `obj.get(name)` for dicts, else `getattr(obj, name, None)`.

`extract(response) -> dict` dispatches on:
1. `type == "message"` → Anthropic;
2. `object == "chat.completion"` → OpenAI Chat;
3. `object == "response"` → OpenAI Responses;
4. heuristics: has `choices` → OpenAI Chat; has `content` and `usage.input_tokens` → Anthropic;
5. otherwise `{}`.

| Provider | output | input_tokens | output_tokens | tools_called | metadata |
|---|---|---|---|---|---|
| anthropic | join `.text` of `type=="text"` blocks (skip thinking/tool_use) | `usage.input_tokens` | `usage.output_tokens` | `tool_use` blocks → `{name, arguments: input, id}` | `stop_reason`, `response_id` |
| openai chat | `choices[0].message.content` | `usage.prompt_tokens` | `usage.completion_tokens` | `message.tool_calls[].function` → `{name, arguments: _json_args(arguments), id}` (JSON string → dict; unparsable → `{"raw": …}`) | `finish_reason`, `response_id` |
| openai responses | `output_text`, else join `output[*].content[*].text` where `type=="output_text"` | `usage.input_tokens` | `usage.output_tokens` | — | `status`, `response_id` |

All set `provider` and `model`. `_clean()` drops None values (top level and inside metadata); `tools_called` is omitted when empty.

### 8.6 `__init__.py`

`__version__ = "0.1.0"`.

`__all__`:
- Setup: `init`, `get_config`, `replay`.
- Recording: `start_run`, `Run`, `TrackedCall`, `TraceHandle`, `span`, `observe`, `propagate`.
- Versioned inputs: `Prompt`, `Dataset`, `DatasetItem`.
- Results: `Evaluation`, `compare`, `Comparison`, `set_baseline`, `get_baseline`, `APIError`.
- Types: `Record`, `Message`, `ToolCall`, `Span`, `Score`, `RunMetric`.

Functions:
- `init()` builds a new transport and closes the old one **only** if no run in `_open_runs` (finished or not) uses it: finished runs can still be evaluated.
- `start_run(name=None, *, mode="managed", model, provider, generation_params, prompt, dataset, project, tags)`.
- `atexit` flushes open runs and their evaluations' buffered scores.

---

## 9. Eval package (`llmbenchmark.eval`)

`__all__`: `evaluate_run`, `EvalReport`, `judge`, `ClaudeJudge`, `JudgeRefusal`, `JudgeIncomplete`, `DEFAULT_JUDGE_MODEL`, `ExactMatch`, `ExpectedTools`, `correctness`, `to_test_case`, `to_conversational_test_case`, `spans_to_trace_dict`.

### 9.1 `judges.py`

`DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"`.

- `ClaudeJudge(DeepEvalBaseLLM).__init__(model=DEFAULT_JUDGE_MODEL, *, client=None, async_client=None, max_tokens=16000, fallbacks=True)` stores its fields **before** `super().__init__(model)`, because DeepEval's `__init__` calls `load_model()`. `load_model()` returns `self._client` (possibly None).
- It is lazy: `anthropic.Anthropic()` / `AsyncAnthropic()` are imported and created on first use. `get_model_name()` returns the model id; `supports_temperature()` → `False`; `supports_structured_outputs()` → `True`.
- It never sends `temperature`.
- `generate(prompt, schema=None)` / `a_generate`: without a schema, `messages.create` and join the text blocks. With a schema, `messages.parse(**request, output_format=schema)`. If that raises an exception whose class name is `BadRequestError` and whose message mentions "schema", retry with `messages.create` and the schema's JSON appended to the prompt ("Respond with only a JSON object matching this JSON schema…"), then parse from the first `{` to the last `}` with `model_validate_json`. Re-raise everything else.
- Opt-in server-side fallback: header `anthropic-beta: server-side-fallback-2026-07-01` plus `extra_body={"fallbacks": "default"}`.
- `_checked` raises `JudgeRefusal` (refusal) and `JudgeIncomplete` (max_tokens); `_parsed` raises `JudgeIncomplete` when `parsed_output` is None.
- `judge(model)` routes: None or `claude*` → `ClaudeJudge`; anything else is passed through.

### 9.2 `metrics.py`

DeepEval 4.x renamed `LLMTestCaseParams` to `SingleTurnParams`. Import whichever exists:

```python
try: from deepeval.test_case import SingleTurnParams as _Params
except ImportError: from deepeval.test_case import LLMTestCaseParams as _Params
```

Custom metrics subclass `BaseMetric`, call `super().__init__()`, set `threshold`, `evaluation_model=None`, `include_reason=True`, `async_mode=True`, `strict_mode=False`, implement `measure` (sets `score`, `success = score >= threshold`, `reason`), `a_measure` (delegates), `is_successful()`, and a `__name__` **property** (DeepEval uses it as the metric name).

| Metric | Behavior |
|---|---|
| `ExactMatch(threshold=1.0, case_sensitive=False)` | `_required_params = [ACTUAL_OUTPUT, EXPECTED_OUTPUT]`; strips both outputs and lowercases them unless `case_sensitive`; score 1.0/0.0; reason `"exact match"` or `"expected '…', got '…'"`; `__name__ = "Exact Match"` |
| `ExpectedTools(threshold=1.0)` | `_required_params = [TOOLS_CALLED, EXPECTED_TOOLS]`; score = \|expected ∩ called names\| / \|expected\| (1.0 if nothing is expected); reason `"never called: …"`; `__name__ = "Expected Tools"`. Free and deterministic, so it works in mock mode. DeepEval's `ToolCorrectnessMetric` needs a judge model even to construct |
| `correctness(model, threshold=0.7)` | `GEval(name="Correctness", criteria="Determine whether the actual output is factually consistent with the expected output. Extra correct detail is fine; any contradiction or missing key fact from the expected output is not.", evaluation_params=[INPUT, ACTUAL_OUTPUT, EXPECTED_OUTPUT])` |

Both custom metrics call `super().__init__()`.

### 9.3 `adapters.py`

- **`required_fields(metric)`**: `metric.evaluation_params` (G-Eval) if it's a list, else `_required_params` if a list (DeepEval's base class has a *typing object* there), else `[]`. Values are mapped to DeepEval param names (`input`, `actual_output`, `expected_output`, `context`, `retrieval_context`, `tools_called`, `expected_tools`).
- **`applies_to(metric, record)`**:
  - conversational metrics (`isinstance(m, BaseConversationalMetric)`) apply only to `conversation` records;
  - metrics with `requires_trace` (an instance attribute set by TaskCompletion, StepEfficiency, PlanAdherence) apply only to `agent` records;
  - everything else applies to all kinds.
- **`missing_fields(metric, record)`**: for conversational metrics, `["messages"]` if there are none; otherwise the required fields that fail checks such as `output` non-empty or `tools_called is not None`.
- **`to_test_case(record, spans)`**: an `LLMTestCase` with `name=record.id`, `actual_output=output or ""`, tool calls (`input_parameters=arguments`), `expected_tools` as `ToolCall(name=…)`, and metadata. For agent records with spans, it calls `attach_trace`.
- **`to_conversational_test_case(record)`**: turns come from user/assistant messages (tool calls included), plus a final assistant turn from `output`. `chatbot_role=system`, `expected_outcome=expected_output`.
- **`attach_trace(case, spans)`**: sets DeepEval's **private** `case._trace_dict` when `"_trace_dict" in type(case).__private_attributes__` (pinned by `test_deepeval_still_reads_private_trace_dict`).
- **`spans_to_trace_dict(spans)`**: nested `{name, type, input?, output?, model?, error?, children:[…]}`, ordered by `started_at`.
  - Type map: agent→`agent`, llm→`llm`, tool→`tool`, retrieval→`retriever`, handoff/other→`base`.
  - Multiple roots are wrapped in `{"name": "trace", "type": "base"}`.

### 9.4 `runner.py`: `evaluate_run(run | run_id, metrics, *, records=None, upload=True, strict=False, supersedes=None, max_concurrent=10, print_results=False) -> EvalReport`

1. **Resolve.** For a `Run`, use `run.records`, `run.spans`, and `flush=run.flush`. For an id: validate it, lazily init the global SDK, then `get_records`, and fetch spans via `get_all("/v1/runs/{id}/spans")` if there are agent records.
2. **Pick eligible records.** A record is eligible if it has no `error` and has `output` (or is a conversation with messages). The rest go to `skipped_record_ids`.
3. **Plan every (record, metric) pair.**
   - Not applicable → appended to `skipped` with `reason="not applicable to <kind> records"`.
   - Missing fields → `incomplete`.
   - Otherwise planned.
4. **Strict mode.** With `strict=True` and any incomplete pair, raise `ValueError("… Exact Match needs expected_output …")` **before creating anything**.
5. **Create the Evaluation.**
   - `evaluator="llmbenchmark"`, `version=__version__`, `source="llmbenchmark"`, `before_send=flush`, `upload=upload`, `supersedes=resolve_supersedes(...)`.
   - `metrics=[_spec(m)]`: name, threshold, criteria, and `params` (`class`, `fields` or `conversational`, `evaluation_steps`, `strict_mode`).
   - `judge_model` = the sorted unique string `evaluation_model`s, joined with `","`.
   - Each incomplete pair becomes a `Score(error="missing required field(s): …")`.
6. **Group and run.** Records are grouped by (conversational?, tuple of applicable metric indices), with one `_run_deepeval` call per group. Results map back by `test_result.name`, falling back to the index. Each metric result becomes `ev.score(...)`, with `passed=None` when the metric errored (DeepEval reports `success=False` for errors; an error is not a verdict, so it stays out of pass rates, like missing fields).
7. **Finalize and report.** `ev.finalize()`, then return `EvalReport(run_id, evaluation_id, scores, skipped_record_ids, skipped, uploaded, evaluation)` with `.summary()` and `.to_dataframe()`.

- `_run_deepeval(**kwargs)`: if `asyncio.get_running_loop()` succeeds (Jupyter), run `deepeval.evaluate` in `ThreadPoolExecutor(max_workers=1, thread_name_prefix="llmbenchmark-eval")`; otherwise call it directly. Why: DeepEval's nest_asyncio patching of a running loop breaks on Python 3.12+ ("Timeout should be used inside a task").
- `_config(cls, **kwargs)` keeps only kwargs that are fields of that DeepEval config dataclass (`AsyncConfig(max_concurrent)`, `DisplayConfig(print_results, show_indicator=True, inspect_after_run=False)`, `ErrorConfig(ignore_errors=True)` from `deepeval.evaluate.configs`), so the code works across DeepEval versions.

---

## 10. Backend `server/`

### 10.1 `auth.py` and `keys.py`

- `Principal(name, team, role: "reader"|"writer")` with a `can_write` property. `ANONYMOUS = Principal("anonymous", "default", "writer")`.
- `KeyStore(entries: [(sha256 hex, Principal)])`. A hash appearing twice → `ValueError` (which team/role would win would be an accident). `authenticate(token)` hashes the token and compares against **all** entries with `hmac.compare_digest` (no early exit).
- `load_key_store(*, api_keys=(), keys=None, keys_file=None)` merges three sources:
  - `api_keys`: writers in team `"default"`, named `key-<hash[:8]>`;
  - `keys`: `{raw_key: {team, role, name}}`, for tests and embedding;
  - `keys_file`: JSON `{"keys": [{name, team, role, key_sha256}]}`. Every hash is validated (64 lowercase hex chars).
  - Every `keys`/`keys_file` entry needs a team and an **explicit** role `reader` or `writer` (no default: a forgotten role must not become a writer).
- `python -m server.keys create --file F --team T --name N [--role reader|writer]` generates `"lmb_" + secrets.token_urlsafe(32)` and prints it **once**. It stores only the hash, writes the file and `chmod`s it `0600` (also when it already existed), and rejects a duplicate (team, name). `list` prints team, name and role.

### 10.2 `db.py`

Every table has `team`. JSON columns use `sa_column=Column(JSON)`, and `Record.metadata` is stored as attribute `meta`, column `"metadata"`.

| Table | Key | Notable columns |
|---|---|---|
| `runs` | `id` | `team`, `project`, `mode`, `model`, `generation_params`, `prompt` (JSON) + `prompt_hash`, `dataset` (JSON) + `dataset_hash`, `tags`, `status`, `started_at`, `ended_at` (server clock), `lineage`, `created_by`, `manifest` (JSON `{"records": [...], "spans": [...]}`, set at close) |
| `records` | `id` | FK `run_id`; all `Record` fields |
| `spans` | `id` | FK `run_id`, FK `record_id`; all `Span` fields |
| `prompts` | (`team`, `hash`) | `name`, `version`, `system`, `template`, `created_at` |
| `datasets` | (`team`, `hash`) | `name`, `version`, `item_count`, `verified`, `created_at` |
| `dataset_items` | (`team`, `dataset_hash`, `id`) | `position`, `data` (JSON) |
| `evaluations` | `id` | FK `run_id`; `source`, `evaluator`, `evaluator_version`, `judge_model`, `metrics`, `config_hash`, FK `supersedes`, `status`, `created_at`, `finalized_at` (server clock), `created_by`, `manifest` (JSON `{"scores": [...], "metrics": [...]}`) |
| `scores` | `id` | FK `evaluation_id`, FK `record_id` |
| `run_metrics` | `id` | FK `evaluation_id` |
| `baselines` | (`team`, `project`) | `run_id`, `evaluation_id`, `set_by`, `set_at` |
| `audit_log` | autoincrement `id` | `team`, `actor`, `action`, `target_type`, `target_id`, `details`, `at` |

`make_engine(url=None)`:
- `url = url or $DATABASE_URL or "sqlite:///llmbenchmark.db"`.
- SQLite: `connect_args={"check_same_thread": False}`; for `sqlite://` and `sqlite:///:memory:`, also `poolclass=StaticPool` (one shared in-memory DB for tests). Register `@event.listens_for(engine, "connect")` that runs **`PRAGMA foreign_keys=ON`** (SQLite ignores FKs otherwise).
- `SQLModel.metadata.create_all(engine)`. There are no migrations.

### 10.3 `app.py`: `create_app(database_url=None, api_keys=None, *, keys=None, keys_file=None, allow_anonymous=None, max_body_bytes=None)`

**No `from __future__ import annotations` in this file.** FastAPI resolves string annotations in module globals, but the route dependency aliases (`Reader`, `Writer`, `Db`, `Limit`, `Offset`) are closures inside `create_app`. With the future import, they silently become query parameters (422 "query p missing").

**Configuration**
- With no key arguments at all, read `LLMBENCHMARK_SERVER_API_KEYS` and `LLMBENCHMARK_SERVER_KEYS_FILE`.
- An empty store with anonymous mode off (`LLMBENCHMARK_SERVER_ALLOW_ANONYMOUS`) → `RuntimeError` naming all three env vars.
- Anonymous mode → every request is `ANONYMOUS` (with a warning).
- `_BodySizeLimit` (pure ASGI middleware; `max_bytes` default 16 MiB or `LLMBENCHMARK_SERVER_MAX_BODY_BYTES`): a `Content-Length` over the limit gets an immediate 413 JSON response. Otherwise it wraps `receive`, counts body bytes (catching chunked uploads), and raises `HTTPException(413)` once the count passes the limit.
- `@app.exception_handler(IntegrityError)` → 409 `"conflicts with existing data"` (never a 500, which the SDK would retry).

**Dependencies**
- `principal`: Bearer scheme parsing, then `store.authenticate` → `Principal`, or 401 with `WWW-Authenticate: Bearer`.
- `writer`: 403 for readers.
- Path ids use `Path(pattern=ID_PATTERN)`.

**Helpers**
- `get_owned(s, row_cls, key, p, what)` → 404 unless `row.team == p.team`.
- `audit(...)` appends an `audit_log` row.
- `check_batch(n)` → 413 when `n` exceeds `MAX_BATCH_RECORDS`.
- `_check_depth(data)` → 422 when `json_depth(data) > SERVER_MAX_JSON_DEPTH`; called by `_to_row` (every write) and for dataset items. Deeper JSON would be stored but could never be serialized back (every read of the run → 500).
- `upsert_children(...)` applies the ownership and immutability rule shared by records, spans, scores and metrics:
  - an existing id owned by another team or owner → 409;
  - if the owner is closed (run not `running`; evaluation `finalized`): an existing row must be **identical** (compare `model_dump(mode="json")` after `_from_row`, which re-attaches UTC to SQLite's naive datetimes), else 409. A new row is accepted only if its id is in the owner's close **manifest** (a late delivery), else 409. Client timestamps play no part;
  - otherwise the row is merged.
- `_to_row` and `_from_row` convert between pydantic models and rows (with the `metadata`/`meta` rename, keeping only known fields, and applying `_utc`).

**Routes** (all `/v1/*` routes are team-scoped; mutating routes need a writer key)

| Route | Behavior |
|---|---|
| `POST /runs` | Insert-if-absent. Existing in the same team and project → 200 with the stored run; else 409. Always stored with `status="running"`, `ended_at=None` (whatever the client sent). Stamps `team` and `created_by`; audit `run.create` |
| `PATCH /runs/{id}` | `running` → `completed`/`failed` sets `ended_at` = **server time** and stores the manifest; audit `run.<status>` (with counts). Already terminal: same status → 200 no-op, other → 409 |
| `GET /runs?project&model&mode&limit=100&offset` | |
| `GET /runs/{id}` | |
| `POST /runs/{id}/records` | 413, 404, 422 (`run_id` mismatch), then `upsert_children` |
| `GET /runs/{id}/records` | Paged |
| `POST /runs/{id}/spans` | Records must exist in the run (422), then `upsert_children` (manifest `spans`) |
| `GET /runs/{id}/spans?trace_id=` | |
| `POST /prompts` | The server **recomputes the hash** from (`system`, `template`) → 422 on mismatch. Insert-if-absent per (team, hash) |
| `GET /prompts/{hash}` | |
| `POST /datasets` | Insert-if-absent; `item_count >= 0` (422). An empty dataset must have `hash == content_hash([])` (422) and is `verified` at once |
| `POST /datasets/{hash}/items` | An existing item must be identical (409). Refused once `item_count` is reached (409). When the last item arrives, the server **recomputes** `content_hash([item data in position order])`: mismatch → rollback and 422; match → `verified=True` |
| `GET /datasets/{hash}[/items]` | |
| `POST /runs/{id}/evaluations` | 422 on a `run_id` mismatch. Insert-if-absent (same team and run → 200, else 409). `supersedes` must be an evaluation of the same run (422). Audit `evaluation.create` |
| `GET /runs/{id}/evaluations` | Each with a computed `superseded_by` |
| `GET /evaluations/{id}` | |
| `POST /evaluations/{id}/scores` | Records must belong to the evaluation's run (422), then `upsert_children` (manifest `scores`) |
| `GET /evaluations/{id}/scores` | |
| `POST /evaluations/{id}/metrics` | |
| `GET /evaluations/{id}/metrics` | |
| `POST /evaluations/{id}/finalize` | Optional body `EvaluationFinalize` (manifest); sets `finalized_at` = **server time**; idempotent; audit (with counts) |
| `GET /evaluations/{id}/summary` | `{evaluation, metrics: [{metric, mean_score, pass_rate, n, errors}], reported: [RunMetric]}` |
| `PUT /baselines` | The evaluation must belong to the run, and the run to the project (422). The evaluation must be finalized (409). Upsert; audit `baseline.set` |
| `GET /baselines?project=` | 404 when none |
| `GET /compare?head=&base=` | Without `base`, the head project's baseline (404 when none). See below |
| `GET /audit?target_id&limit&offset` | Newest first |

**Comparison** (`_compare`)
- Record key: `item:<dataset_item_id>#<sample_index>`, else `input:<sha256(system\0input)[:16]>#<sample_index>`. Records sharing a key within one run are **not collapsed**: in logging order they become `key`, `key~2`, `key~3` (plus a warning).
- Per (key, metric) (the latest score per record and metric), the status is:
  - `added` / `removed`;
  - `errored` (either side has `error`: a scoring failure is not a model change);
  - `regressed` (passed→failed) / `improved`;
  - `worse` / `better` (score moved);
  - `unchanged`.
- Per metric: `base_mean`, `head_mean`, `delta`, both pass rates, `matched`, `regressions`, `improvements`, `errors`.
- Warnings:
  - a missing dataset version;
  - **different dataset hashes**;
  - **different `config_hash`**;
  - an open evaluation on either side;
  - records sharing a key (matched in logging order).
- `differences`: run fields (`model`, `provider`, `generation_params`, `prompt_hash`, `dataset_hash`, `mode`) and evaluation fields (`evaluator`, `evaluator_version`, `judge_model`, `config_hash`) that differ.

---

## 11. Tests

All tests are offline.

**Fixtures** (`conftest.py`)
- `config`: `batch_size=3`, `max_retries=2`, `capture_git=False`.
- `server_client`: `api_keys={"test-key"}`, team `default`.
- `teams_client`: `TEAM_KEYS`: `key-a` (team-a writer "alice"), `key-a-reader`, `key-b` (team-b writer "bob").
- `live_transport`: a `Transport` around the `TestClient`.
- `httpx_transport`: like `live_transport` but through a **real `httpx.Client`** (`MockTransport` forwarding to the TestClient). TestClient is httpx2-based and encodes JSON differently (it accepts NaN/inf), so production-only bugs need this fixture.
- `mock_transport`: respx.
- `global_sdk`: resets the module singletons.
- `live_sdk`: points the module singletons at `live_transport`, for `compare()` and `evaluate_run(id)`.
- Constants: `AUTH`, `bearer(key)`.

| File | Covers |
|---|---|
| `test_transport.py` | Retries, fallback, replay, the allow-list, chunking. Includes **`test_offline_session_replays_completely`**: a whole offline session (prompt, dataset, run, records, trace spans, evaluation, scores, metrics, finalize, finish = 11 lines) replays into a live backend with 0 failures. This guards the allow-list and ordering |
| `test_run.py` | Lifecycle, batching, errors, the input shapes (system/no-system, conversation with content blocks, dataset item, tool-call extraction from Anthropic and OpenAI), prompt/dataset storage, hash stability, Dataset validation, Prompt rendering, the managed-mode warning, lineage |
| `test_tracing.py` | The span tree and parents, `observe`, the agent record (`tools_called`, summed tokens), asyncio branches, `propagate` across threads, no-op outside a trace, errors, span-recording failure, unserializable values |
| `test_evaluation.py` | BYO scores, aggregates and summary; `log_results`; scores flush records first; bad score values; an exception leaves the evaluation open; `config_hash`; finalized immutability (late, changed, identical); late delivery to a finished run; status freeze; supersede; cross-run supersede 422; audit |
| `test_compare.py` | Regressions and improvements, `differences`, like-for-like warnings, baselines, baseline-must-be-finalized, no-baseline 404, bad ids |
| `test_server.py` | Auth, idempotency, the score round trip, prompt hash check, dataset item immutability, **team isolation** (404 across teams, reused run id 409, readers 403, per-team prompts), the keys CLI and file, plus the §14 regressions |
| `test_eval.py` | Managed eval and upload; event loop; required-field errors and `strict`; conversation-only metrics (`TurnCount` fake); agent trace metrics (`UsedToolInTrace` fake reading `_trace_dict`) by Run and by id; the `_trace_dict` pin; the trace-dict shape; `ExpectedTools`; supersede; judge tests; regressions |

---

## 12. Notebooks and docs

- **`examples/tutorial.ipynb`** has 50 cells, steps 0–13 (see its table). Edit it with `nbformat` (v4) rather than by hand, and keep outputs empty. It starts the backend with `LLMBENCHMARK_SERVER_ALLOW_ANONYMOUS=1` via `subprocess.Popen(..., env=...)`. Mock responses are dicts shaped like real responses. The mock GPT answers "continents" → `"7"` and "largest ocean" → `"Atlantic Ocean"`. Judge cells run only with `ANTHROPIC_API_KEY`.
  - It must run headless with `nbclient` (skip the `%pip` cell, keys unset): every cell succeeds.
- **`examples/quickstart.ipynb`** has 19 cells and needs real API keys; it is not executed by the test suite.
- **README.md** covers recording, the input-shapes table, agents, both evaluation modes, version control and comparison, the offline fallback, configuration (9 env vars), the backend (keys CLI, teams, limits, integrity, audit, endpoint table), development, and release.
- **docs/CONCEPTS.md**: Part 3 covers the data model, the two user types, tracing, version control, and teams.

---

## 13. Build order (with checkpoints)

1. Scaffold `pyproject.toml`, `.gitignore`, and the `__init__.py` version, then install `.[eval,server]` and the dev tools.
2. `schemas.py`, `config.py`, `versioning.py`, `assets.py`, then `test_config`. ✔
3. `providers.py`, then `test_providers`. ✔
4. `server/auth.py`, `keys.py`, `db.py`, `app.py`, then `test_server`. ✔
5. `transport.py`, then `test_transport` (except the offline-session test). ✔
6. `tracing.py`, `evaluation.py`, `run.py`, `compare.py`, `__init__.py`, then `test_run`, `test_tracing`, `test_evaluation`, `test_compare`, and the offline-session test. ✔
7. `eval/` (metrics → judges → adapters → runner), then `test_eval`. ✔ `pytest` → all 156 tests pass.
8. Smoke test: create a keys file with two teams, start uvicorn with `LLMBENCHMARK_SERVER_KEYS_FILE`, and check that team B gets 404 on team A's run. Run the tutorial headless with no keys. Delete `llmbenchmark.db` afterwards.
9. Build the wheel (`uv build`, or `pip wheel . --no-deps`, which fetches hatchling into an isolated build env). It must contain only `llmbenchmark/`. Check that `import llmbenchmark` loads no deepeval, pandas, fastapi, anthropic, or openai (`sys.modules`).

---

## 14. Guarantees and the tests that enforce them

Each guarantee below is enforced by the listed tests; keep them passing. Security requirements are specified where they live (invariant 8 in §1, §7, §10) and tested in `test_server.py`, `test_transport.py` and `test_config.py`.

### 14.1 Guarantees

| Area | Guarantee | Test(s) |
|---|---|---|
| Recording | Re-running `init()` never breaks a run you still hold (open or finished); only transports no held run uses are closed | `test_run::test_reinit_does_not_break_an_open_run`, `test_reinit_does_not_break_a_finished_run` |
| Recording | Values JSON can't encode are stored as strings; nesting deeper than `MAX_JSON_DEPTH` is truncated | `test_run::test_unserializable_metadata_and_params_are_sanitized`, `test_transport::test_unserializable_values_are_stringified`, `test_tracing::test_unserializable_span_values_are_stringified`, `test_server::test_sdk_truncates_deep_values_instead_of_failing` |
| Recording | A record that can't be built never masks the user's exception; the fallback record keeps output, model and tokens | `test_run::test_non_string_output_is_coerced`, `test_recording_failure_never_masks_the_users_exception`, `test_recording_fallback_keeps_output_and_model` |
| Recording | Odd input types (content-block system prompts, numeric strings, `None` tags) are coerced, not lost | `test_run::test_odd_input_types_are_coerced_not_lost` |
| Recording | `KeyboardInterrupt` inside `track()` is recorded and always re-raised | `test_run::test_keyboard_interrupt_is_recorded_and_reraised` |
| Recording | Logging to a finished run raises (`log`, `log_results`) or is reported (`track`, `trace`); evaluating a finished run works | `test_run::test_logging_to_a_finished_run_fails_loudly` |
| Recording | Empty DataFrame cells (NaN, `pd.NA`, `NaT`) are missing values, never the text "nan"; `log_results` aggregates count its rows | `test_run::test_empty_dataframe_cells_are_missing_not_nan_text` |
| Tracing | An agent with no tool calls records `tools_called == []` and is scored (not "missing") | `test_eval::test_expected_tools_metric_on_agents` |
| Tracing | Parallel traces never send spans before their record | `test_tracing::test_parallel_traces_never_send_spans_before_their_record` |
| Tracing | A span recording failure never breaks the agent; spans finishing after their trace are reported; `@observe` covers generator iteration | `test_tracing::test_span_recording_failure_never_breaks_the_agent`, `test_spans_finishing_after_the_trace_are_reported`, `test_observe_on_generators_covers_the_iteration` |
| Sending | Every chunk of a batch is attempted (or saved) even after one fails; `uploaded` reports failures | `test_transport::test_every_chunk_is_attempted_after_a_failure`, `test_eval::test_upload_failure_is_reported` |
| Sending | NaN/±inf never block an upload; a payload that can't be encoded is rejected, not retried forever | `test_evaluation::test_non_finite_numbers_never_block_uploads`, `test_unencodable_payload_is_rejected_not_retried_forever` |
| Sending | Scores are sent after the records they reference; scores of never-finalized evaluations are sent at finish and exit | `test_eval::test_evaluate_run_flushes_buffered_records_before_upload`, `test_evaluation::test_scores_reference_records_flushed_first`, `test_unfinalized_evaluation_scores_are_sent_on_finish_and_exit` |
| Replay | Strict order: replay stops at the first failure that can heal; malformed lines don't abort it; lines appended meanwhile are kept; interrupted and missing files are handled | `test_transport::test_replay_stops_at_the_first_temporary_failure`, `test_replay_survives_malformed_lines`, `test_replay_keeps_retryable_and_rejects_permanent_failures`, `test_replay_does_not_lose_lines_appended_meanwhile`, `test_replay_after_an_interrupted_replay`, `test_replay_of_a_missing_file_is_a_no_op` |
| Replay | A complete offline session replays into a live backend with no failures | `test_transport::test_offline_session_replays_completely` |
| Server | Scores and spans must reference records of the same run; SQLite enforces foreign keys | `test_server::test_score_for_unknown_record_is_rejected`, `test_sqlite_enforces_foreign_keys` |
| Server | Re-creating a run never resets or moves it; runs are always created open | `test_server::test_recreating_a_run_does_not_reset_it`, `test_evaluation::test_runs_cannot_be_created_already_finished` |
| Server | Finished runs and finalized evaluations are frozen: only identical resends and manifest ids are accepted; client timestamps are ignored | `test_evaluation::test_finalized_evaluation_rejects_new_scores`, `test_late_delivery_to_a_finished_run_is_accepted`, `test_closed_runs_and_evaluations_ignore_client_timestamps`, `test_finished_run_status_cannot_change` |
| Server | Datasets are verified against their hash | `test_server::test_datasets_are_verified_against_their_hash` |
| Evaluation | Metric errors are recorded with `passed=None` and don't lower pass rates | `test_eval::test_metric_errors_are_not_failures` |
| Evaluation | `supersedes` must name an evaluation of the same run (checked locally) | `test_eval::test_supersedes_from_another_run_is_caught_locally` |
| Evaluation | Scores accept numpy bools and numeric-string thresholds | `test_evaluation::test_numpy_like_and_string_inputs_to_score` |
| Evaluation | The judge raises `JudgeIncomplete` when cut off or without a verdict | `test_eval::test_claude_judge_raises_when_cut_off` |
| Compare | Errors show as `errored` (never "unchanged" or a regression); records sharing a key are matched in order, not collapsed | `test_compare::test_errors_are_not_hidden_as_unchanged`, `test_duplicate_items_are_matched_in_order_not_collapsed` |
| Explicit reads | Failed reads raise `APIError`, never the HTTP library's exception | `test_compare::test_baseline_must_be_finalized`, `test_compare_without_baseline_is_a_clear_error` |
| Config | Bad env values name the variable; out-of-range values are refused | `test_config::test_invalid_env_value_names_the_variable`, `test_invalid_bool_env_value_rejected`, `test_out_of_range_values_rejected` |

### 14.2 Known limitations

- No migrations (`create_all`): adopt Alembic before changing tables on a live database.
- No per-project roles within a team. No key rotation or expiry endpoint: edit the keys file and restart.
- Ids are global primary keys: using another team's id is a 409 (not 404), which reveals that the id exists, and lets a team squat ids another team would use. Low risk with random uuid4 ids; scoping keys by `(team, id)` would close it. Avoid predictable user-chosen ids (`log(id=...)`).
- `/v1/compare`, `/summary` and evaluation lists load whole runs into memory (no caps); `/docs` and `/openapi.json` are unauthenticated.
- DeepEval telemetry is opted out only in notebooks and tests.
- `_trace_dict` is a private DeepEval API (pinned by a test).
- Not implemented: repeated sampling with variance or pass@k (`sample_index` exists), cost tracking, a PII redaction hook, human review labels, OTel export, a CI regression gate.
- No LICENSE, CI, or lint/type configuration.

---

## 15. Conventions and gotchas

- **Commands**: `.venv/bin/pytest`. Backend: `LLMBENCHMARK_SERVER_KEYS_FILE=keys.json .venv/bin/uvicorn server.app:create_app --factory` (or `LLMBENCHMARK_SERVER_ALLOW_ANONYMOUS=1` for local development).
- Every SDK module uses `from __future__ import annotations` and a *why* docstring. **Exception: `server/app.py`** (see §10.3).
- Any new code path that records or sends data must uphold invariant #1: add a test that injects a failure and asserts nothing raises.
- Every new write endpoint the SDK uses must be **added to the replay allow-list** and covered by `test_offline_session_replays_completely`.
- Every new server table needs `team`, and every query must filter by `p.team` (use `get_owned`).
- Public API changes require updating `__all__`, the README, CONCEPTS, this file, and the version.
- Never import `anthropic`, `openai`, `deepeval`, or `pandas` at module level in the core.
- Never send `temperature`/`top_p` to current Claude models.
- `TestClient` is `httpx2`-based in current Starlette. The transport only duck-types its client, so never `isinstance`-check it, and never catch `httpx.*Error` for explicit reads (use `APIError`).
- DeepEval drift: go through `_config()`, the params shim, and `required_fields()` (the base class's `_required_params` is a typing object).
- Notebooks: commit them with outputs cleared, and never display secrets.
