"""Agent tracing: spans form a tree per trace and attach to one agent record."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest

import llmbenchmark
from llmbenchmark import tracing
from llmbenchmark.run import Run
from tests.conftest import AUTH


def llm_response(text, tokens=(10, 2)):
    return NS(type="message", id="m", model="claude-opus-5-5", stop_reason="end_turn",
              content=[NS(type="text", text=text)], usage=NS(input_tokens=tokens[0], output_tokens=tokens[1]))


@llmbenchmark.observe(kind="tool")
def search(query, limit=3):
    return [f"doc about {query}"][:limit]


@llmbenchmark.observe(kind="tool", name="async_lookup")
async def lookup(key):
    await asyncio.sleep(0)
    return key.upper()


def by_name(spans):
    return {s.name: s for s in spans}


def test_trace_builds_span_tree_and_agent_record(live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()
    with run.trace(input="Find a doc about cats", expected_tools=["search"]) as tr:
        with llmbenchmark.span("planner", kind="agent"):
            with llmbenchmark.span("plan", kind="llm") as s:
                s.response = llm_response("search for cats")
            docs = search("cats")
        with llmbenchmark.span("answer", kind="llm") as s:
            s.response = llm_response("Here is a doc", tokens=(20, 5))
        tr.output = docs[0]

    spans = by_name(tr.spans)
    assert set(spans) == {"agent", "planner", "plan", "search", "answer"}
    root = spans["agent"]
    assert root.parent_id is None and root.kind == "agent"
    assert spans["planner"].parent_id == root.id
    assert spans["plan"].parent_id == spans["planner"].id
    assert spans["search"].parent_id == spans["planner"].id
    assert spans["search"].input == {"query": "cats"} and spans["search"].output == ["doc about cats"]
    assert spans["plan"].model == "claude-opus-5-5" and spans["plan"].output == "search for cats"

    record = tr.record
    assert record.kind == "agent" and record.trace_id == tr.trace_id
    assert record.output == "doc about cats"
    assert [t.name for t in record.tools_called] == ["search"]
    assert record.expected_tools == ["search"]
    assert (record.input_tokens, record.output_tokens) == (30, 7)  # summed over llm spans

    stored = server_client.get(f"/v1/runs/{run.id}/spans", headers=AUTH, params={"trace_id": tr.trace_id}).json()
    assert len(stored) == 5 and all(s["record_id"] == record.id for s in stored)


def test_async_tasks_keep_their_own_parents(live_transport, config):
    run = Run(live_transport, config, mode="external").start()

    async def agent():
        async def branch(name):
            with llmbenchmark.span(name, kind="agent"):
                return await lookup(name)
        return await asyncio.gather(branch("left"), branch("right"))

    with run.trace(input="parallel") as tr:
        tr.output = asyncio.run(agent())

    spans = tr.spans
    lookups = [s for s in spans if s.name == "async_lookup"]
    parents = {s.id: s.name for s in spans}
    assert sorted(parents[s.parent_id] for s in lookups) == ["left", "right"]
    assert sorted(s.output for s in lookups) == ["LEFT", "RIGHT"]


def test_threads_need_propagate(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    with run.trace(input="threads") as tr:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(llmbenchmark.propagate(search), ["a", "b"]))
            list(pool.map(search, ["not-traced"]))  # no context in a plain thread: no-op, no error
        tr.output = "done"
    tool_inputs = sorted(s.input["query"] for s in tr.spans if s.kind == "tool")
    assert tool_inputs == ["a", "b"]


def test_span_outside_trace_is_a_no_op():
    with llmbenchmark.span("lonely", kind="tool") as s:
        s.output = 1
    assert search("x") == ["doc about x"]  # decorator works without a trace


def test_span_errors_are_recorded_and_reraised(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    with run.trace(input="fails", raise_errors=False) as tr:
        with llmbenchmark.span("flaky_tool", kind="tool"):
            raise TimeoutError("tool timed out")
    spans = by_name(tr.spans)
    assert spans["flaky_tool"].error == "TimeoutError: tool timed out"
    assert spans["agent"].error == "TimeoutError: tool timed out"
    assert tr.record.error == "TimeoutError: tool timed out"

    with pytest.raises(ValueError):
        with run.trace(input="raises"):
            raise ValueError("agent crashed")
    assert run.records[-1].error == "ValueError: agent crashed"


def test_span_recording_failure_never_breaks_the_agent(live_transport, config, monkeypatch):
    """Invariant 1: a span that can't be recorded is logged, the agent keeps going."""
    def broken(*args, **kwargs):
        raise RuntimeError("serializer exploded")

    monkeypatch.setattr(tracing, "_build_span", broken)
    run = Run(live_transport, config, mode="external").start()
    with run.trace(input="q") as tr:
        with llmbenchmark.span("step"):
            pass
        tr.output = "ok"
    assert tr.record.output == "ok" and tr.spans == []


def test_unserializable_span_values_are_stringified(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    with run.trace(input="q") as tr:
        with llmbenchmark.span("step", input={"obj": object()}, attributes={"x": object()}) as s:
            s.output = object()
        tr.output = "ok"
    step = by_name(tr.spans)["step"]
    assert step.input["obj"].startswith("<object")
    assert isinstance(step.output, str)


def test_parallel_traces_never_send_spans_before_their_record(live_transport, config, server_client, monkeypatch):
    """#4: with traces on many threads, a span batch could overtake its record
    (still in flight inside another thread's flush) and be rejected."""
    import time

    real = live_transport.send_records

    def slow_send(run_id, records):  # widen the race window
        time.sleep(0.02)
        return real(run_id, records)

    monkeypatch.setattr(live_transport, "send_records", slow_send)
    run = Run(live_transport, config, mode="external").start()

    def task(i):
        with run.trace(input=f"task {i}") as tr:
            with llmbenchmark.span("tool", kind="tool"):
                pass
            tr.output = "ok"

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(task, range(24)))
    run.finish()
    spans = server_client.get(f"/v1/runs/{run.id}/spans", headers=AUTH).json()
    assert len(spans) == 48
    assert not config.fallback_dir.exists() or not list(config.fallback_dir.iterdir())


def test_spans_finishing_after_the_trace_are_reported(live_transport, config, caplog):
    """#13: a background task's span that outlived the trace vanished silently."""
    run = Run(live_transport, config, mode="external").start()

    async def main():
        with run.trace(input="q") as tr:
            async def background():
                with llmbenchmark.span("late"):
                    await asyncio.sleep(0.01)
            task = asyncio.create_task(background())
            await asyncio.sleep(0)
            tr.output = "done"
        await task
        return tr

    with caplog.at_level("WARNING", logger="llmbenchmark"):
        tr = asyncio.run(main())
    assert "late" not in by_name(tr.spans) and "finished after its trace ended" in caplog.text


def test_observe_on_generators_covers_the_iteration(live_transport, config):
    """#13: @observe on a generator recorded '<generator object>' with ~0 ms."""
    @llmbenchmark.observe(kind="tool")
    def stream(n):
        for i in range(n):
            yield i

    @llmbenchmark.observe(kind="tool")
    async def astream(n):
        for i in range(n):
            yield i

    async def consume():
        return [i async for i in astream(2)]

    run = Run(live_transport, config, mode="external").start()
    with run.trace(input="q") as tr:
        assert list(stream(3)) == [0, 1, 2]
        assert asyncio.run(consume()) == [0, 1]
        tr.output = "ok"
    spans = by_name(tr.spans)
    assert spans["stream"].output == {"yielded": 3} and spans["astream"].output == {"yielded": 2}
