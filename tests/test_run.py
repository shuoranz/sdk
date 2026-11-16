"""End-to-end: SDK Run -> Transport -> real FastAPI backend (in-memory SQLite)."""

import logging
from types import SimpleNamespace as NS

import pytest
import respx

import llmbenchmark
from llmbenchmark.run import Run
from tests.conftest import AUTH


def fake_claude_response(text="Paris", tools=()):
    content = [NS(type="text", text=text)] + [
        NS(type="tool_use", id=f"tu_{i}", name=name, input=args) for i, (name, args) in enumerate(tools)
    ]
    return NS(type="message", id="msg_1", model="claude-opus-5-5", stop_reason="end_turn",
              content=content, usage=NS(input_tokens=5, output_tokens=1))


def records_on_server(client, run_id):
    return client.get(f"/v1/runs/{run_id}/records", headers=AUTH).json()


# ---- lifecycle and batching ---------------------------------------------------


def test_run_lifecycle_against_real_backend(live_transport, config, server_client):
    with Run(live_transport, config, name="demo", model="claude-opus-5-5",
             generation_params={"max_tokens": 100}, mode="external").start() as run:
        with run.track(input="Capital of France?", expected_output="Paris") as t:
            t.response = fake_claude_response()
        run.log(input="2+2?", output="4", model="gpt-5", provider="openai")

    stored = server_client.get(f"/v1/runs/{run.id}", headers=AUTH).json()
    assert stored["status"] == "completed"
    assert stored["model"] == "claude-opus-5-5"
    assert stored["generation_params"] == {"max_tokens": 100}
    assert stored["team"] == "default" and stored["created_by"].startswith("key-")

    first, second = records_on_server(server_client, run.id)
    assert first["model"] == "claude-opus-5-5"
    assert first["output"] == "Paris"
    assert first["input_tokens"] == 5
    assert first["latency_ms"] >= 0
    assert first["kind"] == "single"
    assert second["model"] == "gpt-5"


def test_buffer_flushes_at_batch_size(live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()  # batch_size=3 in the fixture
    for i in range(3):
        run.log(input=f"q{i}", output="a")
    assert len(records_on_server(server_client, run.id)) == 3
    run.log(input="q3", output="a")  # buffered, not yet sent
    assert len(records_on_server(server_client, run.id)) == 3
    run.finish()
    assert len(records_on_server(server_client, run.id)) == 4


def test_track_records_errors(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    with run.track(input="q", raise_errors=False) as t:
        raise TimeoutError("provider timed out")
    assert t.record.error == "TimeoutError: provider timed out"

    with pytest.raises(ValueError):
        with run.track(input="q2"):
            raise ValueError("boom")
    assert run.records[-1].error == "ValueError: boom"


def test_exception_marks_run_failed(live_transport, config, server_client):
    with pytest.raises(RuntimeError):
        with Run(live_transport, config, mode="external").start() as run:
            raise RuntimeError("notebook cell blew up")
    assert server_client.get(f"/v1/runs/{run.id}", headers=AUTH).json()["status"] == "failed"


def test_explicit_fields_override_extracted(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    rec = run.log(input="q", response=fake_claude_response(), model="my-alias")
    assert rec.model == "my-alias"
    assert rec.output == "Paris"


def test_non_string_input_is_serialized(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    rec = run.log(input={"question": "hi"}, output="hello")
    assert rec.input == '{"question": "hi"}'


# ---- input shapes -----------------------------------------------------------------


def test_single_prompt_with_and_without_system(live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()
    with run.track(input="Capital of France?", system="Answer in one word.") as t:
        t.output = "Paris"
    plain = run.log(input="Capital of Spain?", output="Madrid")
    run.finish()

    with_system, without = records_on_server(server_client, run.id)
    assert with_system["system"] == "Answer in one word." and with_system["kind"] == "single"
    assert without["system"] is None and plain.kind == "single"


def test_conversation_record(live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()
    rec = run.log(
        messages=[
            {"role": "system", "content": "You are a travel agent."},
            {"role": "user", "content": "I want to go to Paris."},
            {"role": "assistant", "content": "When?"},
            {"role": "user", "content": [{"type": "text", "text": "In May."}]},  # content blocks
        ],
        output="Great, May in Paris it is.",
    )
    assert rec.kind == "conversation"
    assert rec.system == "You are a travel agent."
    assert rec.input == "In May."  # last user turn
    run.finish()
    [stored] = records_on_server(server_client, run.id)
    assert [m["role"] for m in stored["messages"]] == ["system", "user", "assistant", "user"]


def test_dataset_item_fills_record(live_transport, config):
    ds = llmbenchmark.Dataset("qa", [{"id": "q1", "input": "2+2?", "expected_output": "4", "system": "Be terse."}])
    run = Run(live_transport, config, mode="external", dataset=ds).start()
    rec = run.log(item=ds[0], output="4")
    assert (rec.input, rec.expected_output, rec.system, rec.dataset_item_id) == ("2+2?", "4", "Be terse.", "q1")
    override = run.log(item=ds[0], output="4", expected_output="four")
    assert override.expected_output == "four"  # explicit beats item


def test_tool_calls_extracted_from_responses(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    anthropic_rec = run.log(input="weather?", response=fake_claude_response("", tools=[("get_weather", {"city": "Paris"})]))
    assert anthropic_rec.tools_called[0].name == "get_weather"
    assert anthropic_rec.tools_called[0].arguments == {"city": "Paris"}

    openai_rec = run.log(input="weather?", response={
        "object": "chat.completion", "model": "gpt-5",
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}]},
            "finish_reason": "tool_calls"}],
    })
    assert openai_rec.tools_called[0].arguments == {"city": "Rome"}


# ---- versioned inputs and lineage -------------------------------------------------


def test_prompt_and_dataset_are_stored_by_content_hash(live_transport, config, server_client):
    prompt = llmbenchmark.Prompt("qa", system="Be brief.", template="Q: {input}", version="v2")
    ds = llmbenchmark.Dataset("qa", [{"id": "q1", "input": "a"}, {"id": "q2", "input": "b"}], version="2026-11")
    run = Run(live_transport, config, model="m", generation_params={"max_tokens": 10},
              prompt=prompt, dataset=ds).start()

    stored = server_client.get(f"/v1/runs/{run.id}", headers=AUTH).json()
    assert stored["prompt"] == {"name": "qa", "version": "v2", "hash": prompt.hash}
    assert stored["dataset"]["hash"] == ds.hash and stored["dataset"]["item_count"] == 2
    assert server_client.get(f"/v1/prompts/{prompt.hash}", headers=AUTH).json()["template"] == "Q: {input}"
    items = server_client.get(f"/v1/datasets/{ds.hash}/items", headers=AUTH).json()
    assert [i["id"] for i in items] == ["q1", "q2"]


def test_content_hashes_are_stable_and_sensitive():
    a = llmbenchmark.Prompt("qa", system="S", template="T {input}", version="v1")
    b = llmbenchmark.Prompt("renamed", system="S", template="T {input}", version="v9")
    c = llmbenchmark.Prompt("qa", system="S!", template="T {input}")
    assert a.hash == b.hash  # name/version label don't change the content
    assert a.hash != c.hash
    d1 = llmbenchmark.Dataset("x", [{"id": "1", "input": "a"}])
    d2 = llmbenchmark.Dataset("x", [{"id": "1", "input": "a", "expected_output": "b"}])
    assert d1.hash != d2.hash


def test_dataset_validation():
    with pytest.raises(ValueError, match="duplicate"):
        llmbenchmark.Dataset("x", [{"id": "1", "input": "a"}, {"id": "1", "input": "b"}])
    with pytest.raises(ValueError, match="must match"):
        llmbenchmark.Dataset("x", [{"id": "bad id!", "input": "a"}])
    auto = llmbenchmark.Dataset("x", [{"input": "a"}])
    assert len(auto[0].id) == 32  # derived from content


def test_prompt_render():
    p = llmbenchmark.Prompt("qa", system="Be brief.", template="{topic}: {input}")
    item = llmbenchmark.DatasetItem(id="1", input="2+2?", metadata={"topic": "math"})
    assert p.render(item) == "math: 2+2?"
    assert p.messages(item, include_system=True)[0] == {"role": "system", "content": "Be brief."}
    with pytest.raises(KeyError, match="topic"):
        p.render(input="x")


def test_managed_mode_warns_about_missing_fields(live_transport, config, caplog):
    with caplog.at_level(logging.WARNING, logger="llmbenchmark"):
        run = Run(live_transport, config, name="incomplete", model="m")
    assert run.info.lineage.incomplete == ["generation_params", "prompt"]
    assert "missing generation_params, prompt" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="llmbenchmark"):
        external = Run(live_transport, config, mode="external")
    assert external.info.lineage.incomplete == [] and "missing" not in caplog.text


def test_lineage_is_captured(live_transport, config, tmp_path, monkeypatch):
    config.capture_git = True
    monkeypatch.chdir(tmp_path)  # not a git repo: capture must not raise
    run = Run(live_transport, config, mode="external")
    lineage = run.info.lineage
    assert lineage.sdk_version == llmbenchmark.__version__
    assert lineage.python and "pydantic" in lineage.packages
    assert lineage.git_commit is None


# ---- regression tests ---------------------------------------------------------


@respx.mock
def test_reinit_does_not_break_an_open_run(global_sdk):
    """re-running the init() cell mid-run used to raise 'client has been closed'."""
    route = respx.route(url__startswith="http://testserver").respond(201)
    llmbenchmark.init(endpoint="http://testserver", fallback_dir=global_sdk, max_retries=0, capture_git=False)
    run = llmbenchmark.start_run(name="long-benchmark", mode="external")

    llmbenchmark.init(endpoint="http://testserver", fallback_dir=global_sdk, max_retries=0, capture_git=False)
    run.log(input="q", output="a")
    run.finish()  # no exception

    assert [c.request.method for c in route.calls] == ["POST", "POST", "PATCH"]
    assert not any(global_sdk.iterdir())  # nothing fell back


def test_unserializable_metadata_and_params_are_sanitized(live_transport, config, server_client):
    """used to raise at flush time and lose the whole batch."""
    class Opaque:
        def __str__(self):
            return "opaque!"

    with Run(live_transport, config, mode="external", generation_params={"tokenizer": Opaque()}).start() as run:
        run.log(input="q", output="a", metadata={"obj": Opaque()})

    assert server_client.get(f"/v1/runs/{run.id}", headers=AUTH).json()["generation_params"] == {"tokenizer": "opaque!"}
    [stored] = records_on_server(server_client, run.id)
    assert stored["metadata"] == {"obj": "opaque!"}


def test_non_string_output_is_coerced(live_transport, config):
    """t.output = 42 used to raise a ValidationError from the finally block."""
    run = Run(live_transport, config, mode="external").start()
    with run.track(input="q") as t:
        t.output = 42
    assert t.record.output == "42"


def test_recording_failure_never_masks_the_users_exception(live_transport, config):
    """a record that can't be built must not replace the real error."""
    class BadResponse:  # reading it blows up inside the SDK's extraction
        def __getattr__(self, name):
            raise RuntimeError("provider object exploded")

    bad_response = BadResponse()
    run = Run(live_transport, config, mode="external").start()

    with pytest.raises(ValueError, match="boom"):  # not pydantic's ValidationError
        with run.track(input="q") as t:
            t.response = bad_response
            raise ValueError("boom")
    assert "ValueError: boom" in run.records[-1].error
    assert "could not record" in run.records[-1].error

    with run.track(input="q2", raise_errors=False) as t:  # and raise_errors=False holds
        t.response = bad_response
    assert "could not record" in t.record.error


def test_keyboard_interrupt_is_recorded_and_reraised(live_transport, config):
    """an interrupted call used to be recorded with no error at all."""
    run = Run(live_transport, config, mode="external").start()
    with pytest.raises(KeyboardInterrupt):
        with run.track(input="q", raise_errors=False):
            raise KeyboardInterrupt
    assert run.records[-1].error == "KeyboardInterrupt: "


def test_logging_to_a_finished_run_fails_loudly(live_transport, config, caplog):
    """records logged after finish() used to be silently lost (409 on the server)."""
    with Run(live_transport, config, mode="external").start() as run:
        run.log(input="q", output="a")
    with pytest.raises(RuntimeError, match="finished"):
        run.log(input="late", output="a")
    with pytest.raises(RuntimeError, match="inside the run"):
        run.log_results([{"input": "q", "output": "a", "correct": 1}], evaluator="x", scores=["correct"])
    with caplog.at_level(logging.ERROR, logger="llmbenchmark"):
        with run.track(input="late") as t:  # track() never raises; it reports instead
            t.output = "a"
    assert t.record is None and "already finished" in caplog.text
    ev = run.evaluation("x")  # evaluating a finished run is the normal flow
    ev.score(run.records[0], "m", 1.0)
    ev.finalize()
    assert ev.uploaded


def test_lineage_does_not_run_repo_configured_commands(tmp_path, monkeypatch):
    """`git status` used to honor core.fsmonitor from an untrusted .git/config."""
    import shutil
    import subprocess

    from llmbenchmark.versioning import _git_state

    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
    marker = tmp_path / "FSMONITOR_RAN"
    git("config", "core.fsmonitor", f"touch {marker}; false")
    monkeypatch.chdir(repo)

    commit, dirty = _git_state()
    assert commit and dirty is False
    assert not marker.exists()


@respx.mock
def test_reinit_does_not_break_a_finished_run(global_sdk):
    """re-running init() closed the transport a *finished* run still uses,
    so evaluating that run afterwards uploaded nothing."""
    respx.route(url__startswith="http://testserver").respond(201)
    llmbenchmark.init(endpoint="http://testserver", fallback_dir=global_sdk, max_retries=0, capture_git=False)
    with llmbenchmark.start_run("done", mode="external") as run:
        rec = run.log(input="q", output="a")
    llmbenchmark.init(endpoint="http://testserver", fallback_dir=global_sdk, max_retries=0, capture_git=False)
    with run.evaluation("x") as ev:
        ev.score(rec, "m", 1.0)
    assert ev.uploaded
    assert not any(global_sdk.iterdir())  # nothing fell back


def test_odd_input_types_are_coerced_not_lost(live_transport, config):
    """a content-block system prompt (Anthropic format) and other odd types used
    to make log() raise, and track() then kept only an error record."""
    run = Run(live_transport, config, mode="external").start()
    system_blocks = [{"type": "text", "text": "Be brief."}, {"type": "text", "text": " Use English."}]
    with run.track(input="q", system=system_blocks, tags=["a", 1, None]) as t:
        t.response = {"type": "message", "model": "claude", "content": [{"type": "text", "text": "answer"}],
                      "usage": {"input_tokens": "12", "output_tokens": "lots"}}
    rec = t.record
    assert rec.error is None and rec.output == "answer" and rec.system == "Be brief. Use English."
    assert (rec.input_tokens, rec.output_tokens, rec.tags) == (12, None, ["a", "1"])
    rec2 = run.log(input="q", output="a", expected_tools=["search", None], sample_index=None, latency_ms="5")
    assert (rec2.expected_tools, rec2.sample_index, rec2.latency_ms) == (["search"], 0, 5.0)


def test_recording_fallback_keeps_output_and_model(live_transport, config, monkeypatch):
    """when a record can't be built, the error record keeps output/model/tokens."""
    import llmbenchmark.run as run_module

    real = run_module._tool_calls
    monkeypatch.setattr(run_module, "_tool_calls", lambda raw: (_ for _ in ()).throw(RuntimeError("x"))
                        if raw else real(raw))
    run = Run(live_transport, config, mode="external").start()
    with run.track(input="q", model="m") as t:
        t.output = "kept"
        t.tools_called = [{"name": "tool"}]
    assert t.record.output == "kept" and t.record.model == "m" and "could not record" in t.record.error


def test_empty_dataframe_cells_are_missing_not_nan_text(live_transport, config):
    """NaN cells used to become the string "nan" (and a fake expected answer)."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([{"id": "q1", "input": "2+2?", "expected_output": "4"},
                       {"id": None, "input": "3+3?", "expected_output": float("nan")}])
    ds = llmbenchmark.Dataset.from_records(df, "d")
    assert ds[1].expected_output is None and ds[1].id != "nan"

    run = Run(live_transport, config, mode="external").start()
    rows = pd.DataFrame([{"q": "a", "out": "x", "gold": float("nan"), "id": float("nan"), "ok": 1}])
    ev = run.log_results(rows, evaluator="x", input_col="q", output_col="out", expected_col="gold",
                         item_id_col="id", scores=["ok"], aggregates={"ok": 1.0})
    rec = run.records[0]
    assert rec.expected_output is None and rec.dataset_item_id is None
    assert ev.metrics[0].n == 1  # #14: rows given, not all records in the run
