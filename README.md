# llmbenchmark

Record LLM calls, conversations and agent traces from notebooks, store them on a shared backend, and evaluate them, either with your own methods or with [DeepEval](https://github.com/confident-ai/deepeval) and a Claude judge. Results are versioned: every score knows exactly which prompt, dataset, model settings and scoring config produced it, and runs can be compared item by item.

```
notebook ─► your Claude / OpenAI calls, conversations, agents
   └─ llmbenchmark ─(batched HTTP, retries, offline fallback)─► backend (FastAPI) ─► SQLite / Postgres
        ├─ your own scores ─────────────────────────────► Evaluation ─┘     (per team, audited)
        └─ llmbenchmark.eval ─ DeepEval + Claude judge ─► Evaluation ─┘
```

New to SDKs or evals? Read [docs/CONCEPTS.md](docs/CONCEPTS.md) (Part 3 covers the data model and versioning).

**Learn by running:** [examples/tutorial.ipynb](examples/tutorial.ipynb) walks through every feature. It runs without API keys (mock responses) and switches to real Claude/OpenAI calls when `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are set. [examples/quickstart.ipynb](examples/quickstart.ipynb) is the short version.

## Install

```python
# In a notebook, from the private index:
%pip install --index-url https://<artifactory>/api/pypi/<repo>/simple "llmbenchmark[eval]==0.1.*"
```

| Extra | Adds |
|---|---|
| *(none)* | recording, tracing, uploading your own scores, comparing (`httpx`, `pydantic` only) |
| `[eval]` | DeepEval integration with a Claude judge |
| `[server]` | the reference backend |

## Record

```python
import llmbenchmark, anthropic

llmbenchmark.init(endpoint="http://localhost:8000", project="qa-bench")  # or LLMBENCHMARK_* env vars
client = anthropic.Anthropic()

prompt  = llmbenchmark.Prompt("qa", system="Answer in one short sentence.", template="{input}", version="v1")
dataset = llmbenchmark.Dataset("qa", [{"id": "q1", "input": "Capital of France?", "expected_output": "Paris"}],
                               version="2026-11")

with llmbenchmark.start_run("claude-baseline", model="claude-opus-5-5",
                            generation_params={"max_tokens": 1024}, prompt=prompt, dataset=dataset) as run:
    for item in dataset:
        with run.track(item=item) as t:              # input, system, expected output come from the item
            t.response = client.messages.create(
                model="claude-opus-5-5", max_tokens=1024,
                system=prompt.system, messages=prompt.messages(item),
            )
        # model, output, token counts and tool calls are read from the response; latency is timed.
```

Input shapes:

| Shape | How |
|---|---|
| Single prompt | `run.track(input="...")`, optionally `system="..."` |
| Conversation | `run.track(messages=[{"role": "user", ...}, {"role": "assistant", ...}, ...])` |
| Agent (full trace) | `with run.trace(input=task) as tr:` plus `llmbenchmark.span(...)` / `@llmbenchmark.observe(kind="tool")` inside your agent |
| No response object | `run.log(input=..., output=..., model=..., input_tokens=...)` |

- **Recognized responses:** Anthropic `Message`, OpenAI `ChatCompletion` and Responses API objects (or their `.model_dump()` dicts), including tool calls.
- **Errors:** exceptions inside `track()`/`trace()` are recorded and re-raised. Pass `raise_errors=False` to record them and keep looping.
- **Log inside the run.** Once a run has finished (its `with` block ended), `run.log()` and `run.log_results()` raise: a finished run's records are frozen. Evaluating a finished run is fine.
- **Agents across threads:** submit `llmbenchmark.propagate(fn)` to a thread pool so its spans join the current trace.

### Agents

```python
@llmbenchmark.observe(kind="tool")
def search(query): ...

with run.trace(input="Find the refund policy", expected_tools=["search"]) as tr:
    with llmbenchmark.span("plan", kind="llm") as s:
        s.response = client.messages.create(...)
    docs = search("refund policy")
    tr.output = answer
```

Each trace becomes one `kind="agent"` record. Its tool spans become `tools_called`, token counts are summed over LLM spans, and the full span tree is stored for agent metrics.

## Evaluate

### With your own method (`mode="external"`)

```python
with llmbenchmark.start_run("team-x", mode="external", model="my-model") as run:
    with run.evaluation("team-x-rubric", version="1.2", metrics={"accuracy": {"threshold": 1}}) as ev:
        for row in results:
            rec = run.log(input=row.prompt, system=row.system, output=row.answer, expected_output=row.gold)
            ev.score(rec, "accuracy", row.correct, reason=row.note)   # passed = score >= threshold
        ev.metric("accuracy", 0.87, n=len(results))                  # run-level number

# Or in bulk from a DataFrame / list of dicts (inside the run: a finished run accepts no new records):
with llmbenchmark.start_run("team-x-bulk", mode="external", model="my-model") as run:
    run.log_results(df, evaluator="team-x-rubric", scores={"accuracy": "correct"}, aggregates={"accuracy": 0.87})
```

### With llmbenchmark (`mode="managed"`, the default)

```python
from deepeval.metrics import AnswerRelevancyMetric
from llmbenchmark.eval import evaluate_run, judge, correctness, ExactMatch

j = judge()                       # Claude judge (claude-sonnet-5-5); judge("gpt-4.1") for OpenAI
report = evaluate_run(run, metrics=[AnswerRelevancyMetric(model=j), correctness(model=j)])
report.summary()                  # mean score / pass rate / errors per metric
report.to_dataframe()             # one row per (record, metric), with the judge's reasons
```

- **A metric that errors** (e.g. the judge was unavailable) is recorded with `error` set and `passed=None`, so it counts in `errors`, not against the pass rate.
- **Required fields are checked before any judge call.** A record missing a field a metric needs (e.g. `expected_output` for `correctness`) gets a score with `error` set; `strict=True` raises instead. Metrics that don't fit a record kind are skipped (multi-turn metrics only score conversations, trace metrics only score agents).
- Managed runs should declare `model`, `generation_params` and `prompt`. Missing ones are logged as a warning and recorded in the run's `lineage.incomplete`.
- To evaluate a run from an earlier session, pass its id: `evaluate_run("<run-id>", metrics=...)`.

## Version control and comparison

- **Every scoring pass is an immutable Evaluation**: evaluator name and version, metrics, thresholds, criteria, judge model, and a `config_hash` of all of them. Re-scoring creates a new evaluation; pass `supersedes=` to mark the old one as replaced. Nothing is overwritten.
- **Every run pins its inputs**: content hashes of the prompt and dataset, model and generation params, SDK/package versions and the git commit.
- **Finished runs and finalized evaluations are frozen.** Finishing sends the ids of everything logged; afterwards the server accepts late retries of exactly those and refuses anything else. Close times come from the server's clock; client timestamps are never trusted.

```python
diff = llmbenchmark.compare(base=old_report, head=new_report)   # evaluation ids, Evaluations or EvalReports
diff.summary()        # per metric: base vs head mean and pass rate, regressions, improvements
diff.regressions      # items that passed before and fail now
diff.warnings         # e.g. different dataset versions: not a like-for-like comparison

llmbenchmark.set_baseline(old_report)       # project baseline
llmbenchmark.compare(head=new_report)       # compare against it
```

Failed explicit reads (compare, baselines, fetching a run) raise `llmbenchmark.APIError`.

## When the backend is down

Recording never raises. The SDK retries, then saves payloads to `.llmbenchmark/<run_id>.jsonl` and logs a warning. Re-send them later:

```python
llmbenchmark.replay(".llmbenchmark/<run_id>.jsonl")   # -> (sent, failed)
```

Replay sends lines strictly in order. At the first failure that can heal (network error, 5xx) it stops, and that line and everything after it stay in the file for next time. Requests the server rejects for good, and lines that aren't valid llmbenchmark requests for that file's run, are moved to `<file>.rejected`. Don't rename fallback files: each is named after its run, and replay checks every line belongs to it.

Fallback files hold full prompts and outputs, so they are created readable only by you (`0600`, folder `0700`). Treat them as sensitive. Replay refuses symlinked files and files owned by another user.

## Configuration

| Env var | Default | |
|---|---|---|
| `LLMBENCHMARK_ENDPOINT` | `http://localhost:8000` | backend URL |
| `LLMBENCHMARK_API_KEY` | — | sent as `Authorization: Bearer …` |
| `LLMBENCHMARK_PROJECT` | `default` | groups runs |
| `LLMBENCHMARK_BATCH_SIZE` | `50` | records per HTTP request (≥ 1) |
| `LLMBENCHMARK_TIMEOUT` | `10` | HTTP timeout in seconds |
| `LLMBENCHMARK_MAX_RETRIES` | `3` | for network errors / 408 / 429 / 5xx |
| `LLMBENCHMARK_FALLBACK_DIR` | `.llmbenchmark` | where unsent payloads are saved |
| `LLMBENCHMARK_ENABLED` | `true` | `false` = don't send anything |
| `LLMBENCHMARK_CAPTURE_GIT` | `true` | record the git commit/dirty flag with each run |

## Backend

```bash
python -m server.keys create --file keys.json --team search --name alice            # prints a key once
python -m server.keys create --file keys.json --team search --name dashboards --role reader
LLMBENCHMARK_SERVER_KEYS_FILE=keys.json .venv/bin/uvicorn server.app:create_app --factory   # http://localhost:8000/docs

# single team:            LLMBENCHMARK_SERVER_API_KEYS=key1,key2
# local development only: LLMBENCHMARK_SERVER_ALLOW_ANONYMOUS=1
```

- **Teams:** each key belongs to one team and is a `reader` (GET only) or `writer`. Every keys-file entry must state its role, and the same key may not appear twice. Every query is filtered by team; other teams' data is a 404. Keys are stored only as sha256 hashes. The server refuses to start without keys unless anonymous mode is set. The SDK warns if your key would be sent over plain `http://` to a remote host.
- **Database:** set `DATABASE_URL` for Postgres (default: `sqlite:///llmbenchmark.db`). Tables are created at startup; there are no migrations yet.
- **Limits:** request bodies up to `LLMBENCHMARK_SERVER_MAX_BODY_BYTES` (default 16 MiB) and 1000 items per batch (the SDK splits larger batches). Larger requests get `413`. JSON nested deeper than 64 levels gets `422` (the SDK cuts metadata and span values at 32 levels).
- **Integrity:** writes are idempotent by client-generated id. An id that belongs to another run, team or project is a `409`. Scores and spans must reference records of the same run (`422`). Runs are always created open. Finished runs and finalized evaluations accept only identical resends and late deliveries of ids listed when they were closed (`409` otherwise). A dataset is marked `verified` once all its items arrive and their hash matches.
- **Audit:** state changes (run created/finished, evaluation created/finalized, baseline set) are appended to an audit log with the key's name: `GET /v1/audit`.

| Method & path | |
|---|---|
| `POST /v1/runs` · `GET /v1/runs[?project=&model=&mode=]` · `GET/PATCH /v1/runs/{id}` | runs |
| `POST/GET /v1/runs/{id}/records` · `POST/GET /v1/runs/{id}/spans[?trace_id=]` | records, agent spans |
| `POST /v1/prompts` · `GET /v1/prompts/{hash}` · `POST /v1/datasets` · `POST/GET /v1/datasets/{hash}/items` | versioned inputs |
| `POST/GET /v1/runs/{id}/evaluations` · `GET /v1/evaluations/{id}[/summary]` | evaluations |
| `POST/GET /v1/evaluations/{id}/scores` · `POST/GET .../metrics` · `POST .../finalize` | scores, run-level metrics |
| `PUT/GET /v1/baselines` · `GET /v1/compare?head=[&base=]` · `GET /v1/audit` | baselines, comparison, audit |

List endpoints take `limit` and `offset` (max 1000).

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[eval,server]" pytest respx openai ipykernel
.venv/bin/pytest
```

With [uv](https://docs.astral.sh/uv/) installed, `uv sync && uv run pytest` does the same. [AGENTS.md](AGENTS.md) is the detailed spec for coding agents.

## Release

1. Bump `__version__` in `src/llmbenchmark/__init__.py` ([SemVer](docs/CONCEPTS.md#15-versioning-semver)).
2. Build: `python -m build` (or `uv build`). This produces `dist/llmbenchmark-<version>-py3-none-any.whl` and `.tar.gz`.
3. Publish: `twine upload --repository-url https://<artifactory>/api/pypi/<repo> dist/*` (or `uv publish --publish-url …`).
