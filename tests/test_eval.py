"""Eval tests. No API calls: deterministic metrics and a fake Anthropic client."""

import asyncio
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("deepeval")

from deepeval.metrics import BaseMetric  # noqa: E402
from deepeval.metrics.base_metric import BaseConversationalMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import llmbenchmark  # noqa: E402
from llmbenchmark.eval import (  # noqa: E402
    ClaudeJudge,
    ExactMatch,
    ExpectedTools,
    JudgeIncomplete,
    JudgeRefusal,
    evaluate_run,
    judge,
    spans_to_trace_dict,
    to_conversational_test_case,
)
from llmbenchmark.run import Run  # noqa: E402
from tests.conftest import AUTH  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_deepeval(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    monkeypatch.setenv("DEEPEVAL_NO_INSPECT_PROMPT", "1")


# ---- deterministic test metrics (no judge) --------------------------------------


class TurnCount(BaseConversationalMetric):
    """1.0 if the conversation has at least `n` turns."""

    def __init__(self, n=3, threshold=1.0):
        super().__init__()
        self.n, self.threshold = n, threshold
        self.evaluation_model, self.include_reason, self.async_mode, self.strict_mode = None, True, False, False

    def measure(self, test_case, *args, **kwargs):
        self.score = 1.0 if len(test_case.turns) >= self.n else 0.0
        self.success = self.score >= self.threshold
        self.reason = f"{len(test_case.turns)} turns"
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return bool(getattr(self, "success", False))

    @property
    def __name__(self):
        return "Turn Count"


class UsedToolInTrace(BaseMetric):
    """Trace metric: 1.0 if the agent's span tree contains a tool span named `tool`."""

    _required_params = []

    def __init__(self, tool, threshold=1.0):
        super().__init__()
        self.tool, self.threshold, self.requires_trace = tool, threshold, True
        self.evaluation_model, self.include_reason, self.async_mode, self.strict_mode = None, True, False, False

    def measure(self, test_case, *args, **kwargs):
        def names(node):
            yield (node.get("type"), node.get("name"))
            for child in node.get("children", []):
                yield from names(child)

        found = ("tool", self.tool) in set(names(test_case._trace_dict or {}))
        self.score = 1.0 if found else 0.0
        self.success = found
        self.reason = "tool used" if found else "tool not used"
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return bool(getattr(self, "success", False))

    @property
    def __name__(self):
        return "Used Tool"


def server_get(client, path):
    response = client.get(path, headers=AUTH)
    assert response.status_code == 200, response.text
    return response.json()


# ---- managed evaluation -----------------------------------------------------------


def test_evaluate_run_scores_and_uploads(live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()
    a = run.log(input="Capital of France?", output="Paris", expected_output="paris")
    b = run.log(input="2+2?", output="5", expected_output="4")
    failed = run.log(input="q", error="TimeoutError")
    run.finish()

    report = evaluate_run(run, metrics=[ExactMatch()])

    by_record = {s.record_id: s for s in report.scores}
    assert by_record[a.id].score == 1.0 and by_record[a.id].passed is True
    assert by_record[b.id].score == 0.0 and by_record[b.id].passed is False
    assert report.skipped_record_ids == [failed.id]
    assert report.uploaded is True
    assert report.summary().loc["Exact Match", "pass_rate"] == 0.5

    info = server_get(server_client, f"/v1/evaluations/{report.evaluation_id}")
    assert info["status"] == "finalized" and info["source"] == "llmbenchmark"
    assert info["evaluator"] == "llmbenchmark" and info["evaluator_version"] == llmbenchmark.__version__
    assert info["metrics"][0]["name"] == "Exact Match"
    assert info["metrics"][0]["params"]["fields"] == ["actual_output", "expected_output"]
    stored = server_get(server_client, f"/v1/evaluations/{report.evaluation_id}/scores")
    assert {s["record_id"] for s in stored} == {a.id, b.id}


def test_evaluate_run_inside_running_event_loop(live_transport, config):
    """Jupyter runs cells inside an asyncio loop; evaluate_run must still work."""
    run = Run(live_transport, config, mode="external").start()
    run.log(input="q", output="a", expected_output="a")

    async def notebook_cell():
        return evaluate_run(run, metrics=[ExactMatch()], upload=False)

    for _ in range(2):  # twice: the second call used to hit the nested-loop bug
        report = asyncio.run(notebook_cell())
        assert report.scores[0].score == 1.0


def test_missing_required_fields_are_recorded_not_judged(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    complete = run.log(input="q", output="a", expected_output="a")
    no_expected = run.log(input="q2", output="b")

    report = evaluate_run(run, metrics=[ExactMatch()])
    by_record = {s.record_id: s for s in report.scores}
    assert by_record[complete.id].score == 1.0
    assert by_record[no_expected.id].score is None
    assert by_record[no_expected.id].error == "missing required field(s): expected_output"
    assert report.summary().loc["Exact Match", "errors"] == 1


def test_strict_mode_refuses_before_judging(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    run.log(input="q", output="a")  # no expected_output
    with pytest.raises(ValueError, match="Exact Match needs expected_output"):
        evaluate_run(run, metrics=[ExactMatch()], strict=True)
    assert run.evaluations == []  # nothing was created


def test_conversation_metrics_score_only_conversations(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    convo = run.log(messages=[
        {"role": "system", "content": "You are a travel agent."},
        {"role": "user", "content": "Paris please"},
        {"role": "assistant", "content": "When?"},
        {"role": "user", "content": "May"},
    ], output="Booked for May.", expected_output="a booking")
    single = run.log(input="hi", output="hello", expected_output="hello")

    report = evaluate_run(run, metrics=[TurnCount(n=4), ExactMatch()])
    scores = {(s.record_id, s.metric): s for s in report.scores}
    assert scores[(convo.id, "Turn Count")].score == 1.0
    assert (single.id, "Turn Count") not in scores
    assert {"record_id": single.id, "metric": "Turn Count",
            "reason": "not applicable to single records"} in report.skipped
    assert scores[(single.id, "Exact Match")].score == 1.0
    assert scores[(convo.id, "Exact Match")].score == 0.0  # single-turn metrics also apply


def test_conversational_test_case_shape():
    rec = llmbenchmark.Record(run_id="r", input="May", kind="conversation", system="Agent.",
                              messages=[{"role": "system", "content": "Agent."}, {"role": "user", "content": "Paris"},
                                        {"role": "assistant", "content": "When?"}, {"role": "user", "content": "May"}],
                              output="Booked.")
    case = to_conversational_test_case(rec)
    assert [(t.role, t.content) for t in case.turns] == [
        ("user", "Paris"), ("assistant", "When?"), ("user", "May"), ("assistant", "Booked.")]
    assert case.chatbot_role == "Agent."


def test_agent_traces_feed_trace_metrics(live_sdk, live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()
    with run.trace(input="find cats") as good:
        with llmbenchmark.span("search", kind="tool", input={"q": "cats"}) as s:
            s.output = ["cat doc"]
        good.output = "cat doc"
    with run.trace(input="find dogs") as lazy:
        lazy.output = "I don't know"
    plain = run.log(input="hi", output="hello")
    run.finish()

    report = evaluate_run(run, metrics=[UsedToolInTrace("search")])
    scores = {s.record_id: s.score for s in report.scores}
    assert scores == {good.record.id: 1.0, lazy.record.id: 0.0}
    assert any(sk["record_id"] == plain.id for sk in report.skipped)

    # Same result evaluating by run id: records and spans come from the backend.
    by_id = evaluate_run(run.id, metrics=[UsedToolInTrace("search")], upload=False)
    assert {s.record_id: s.score for s in by_id.scores} == scores


def test_expected_tools_metric_on_agents(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    with run.trace(input="refund?", expected_tools=["search", "lookup_order"]) as half:
        with llmbenchmark.span("search", kind="tool"):
            pass
        half.output = "partial"
    with run.trace(input="refund?", expected_tools=["search"]) as none:
        none.output = "guessed"  # used no tools: tools_called == [], scored 0 (not "missing")
    report = evaluate_run(run, metrics=[ExpectedTools()], upload=False)
    scores = {s.record_id: s for s in report.scores}
    assert scores[half.record.id].score == 0.5 and scores[half.record.id].reason == "never called: lookup_order"
    assert scores[none.record.id].score == 0.0 and scores[none.record.id].error is None


def test_deepeval_still_reads_private_trace_dict():
    """Pin: our agent adapter relies on DeepEval's private LLMTestCase._trace_dict.
    If this fails after a DeepEval upgrade, update eval/adapters.attach_trace."""
    assert "_trace_dict" in LLMTestCase.__private_attributes__


def test_spans_to_trace_dict_shape():
    S = llmbenchmark.Span
    root = S(run_id="r", trace_id="t", record_id="x", kind="agent", name="agent", input="task", output="done")
    tool = S(run_id="r", trace_id="t", record_id="x", parent_id=root.id, kind="tool", name="search",
             input={"q": "a"}, output=["b"])
    retr = S(run_id="r", trace_id="t", record_id="x", parent_id=root.id, kind="retrieval", name="kb")
    tree = spans_to_trace_dict([tool, retr, root])
    assert tree["type"] == "agent" and tree["output"] == "done"
    assert [(c["type"], c["name"]) for c in tree["children"]] == [("tool", "search"), ("retriever", "kb")]


def test_re_evaluation_supersedes(live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()
    run.log(input="q", output="a", expected_output="a")
    first = evaluate_run(run, metrics=[ExactMatch()])
    second = evaluate_run(run, metrics=[ExactMatch(case_sensitive=True)], supersedes=first)
    versions = server_get(server_client, f"/v1/runs/{run.id}/evaluations")
    assert [(v["id"], v["superseded_by"]) for v in versions] == [
        (first.evaluation_id, second.evaluation_id), (second.evaluation_id, None)]


# ---- judge ---------------------------------------------------------------------


class Verdict(BaseModel):
    score: float
    reason: str


class FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.requests = []

    def parse(self, **kwargs):
        self.requests.append(kwargs)
        return NS(stop_reason=self.stop_reason, stop_details=NS(category="cyber"),
                  parsed_output=kwargs["output_format"](score=0.9, reason="ok"))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return NS(stop_reason=self.stop_reason, stop_details=None,
                  content=[NS(type="thinking"), NS(type="text", text="hello")])


class FakeAsyncMessages(FakeMessages):
    async def parse(self, **kwargs):
        return FakeMessages.parse(self, **kwargs)


def test_claude_judge_structured_output_and_request_shape():
    messages = FakeMessages()
    j = ClaudeJudge(client=NS(messages=messages))

    verdict = j.generate("grade this", schema=Verdict)
    assert verdict == Verdict(score=0.9, reason="ok")
    assert j.generate("free text") == "hello"

    req = messages.requests[0]
    assert req["model"] == "claude-sonnet-5-5"
    assert "temperature" not in req  # rejected by current Claude models
    assert req["extra_body"] == {"fallbacks": "default"}
    assert j.get_model_name() == "claude-sonnet-5-5"


def test_claude_judge_async():
    j = ClaudeJudge(async_client=NS(messages=FakeAsyncMessages()))
    assert asyncio.run(j.a_generate("grade", schema=Verdict)).score == 0.9


def test_claude_judge_raises_on_refusal():
    j = ClaudeJudge(client=NS(messages=FakeMessages(stop_reason="refusal")))
    with pytest.raises(JudgeRefusal, match="cyber"):
        j.generate("grade", schema=Verdict)


def test_judge_factory():
    assert isinstance(judge(), ClaudeJudge)
    assert judge("claude-opus-5-5").get_model_name() == "claude-opus-5-5"
    assert judge("gpt-4.1") == "gpt-4.1"  # handed to DeepEval's OpenAI model


# ---- regression tests ---------------------------------------------------------


def test_upload_failure_is_reported(live_transport, config, monkeypatch):
    """report.uploaded must reflect failed sends (and every chunk is attempted;
    see test_transport::test_every_chunk_is_attempted_after_a_failure)."""
    run = Run(live_transport, config, mode="external").start()
    for i in range(6):
        run.log(input=f"q{i}", output="a", expected_output="a")
    attempted = []
    monkeypatch.setattr(live_transport, "send_scores",
                        lambda run_id, ev_id, chunk: attempted.append(len(chunk)) or False)

    report = evaluate_run(run, metrics=[ExactMatch()])  # batch_size=3 -> 2 flushes
    assert attempted == [3, 3]
    assert report.uploaded is False


def test_evaluate_run_flushes_buffered_records_before_upload(live_transport, config, server_client):
    """The server rejects scores for records it hasn't received yet."""
    run = Run(live_transport, config, mode="external").start()  # not finished: the record is still buffered
    rec = run.log(input="q", output="a", expected_output="a")
    report = evaluate_run(run, metrics=[ExactMatch()])
    assert report.uploaded is True
    assert server_get(server_client, f"/v1/evaluations/{report.evaluation_id}/scores")[0]["record_id"] == rec.id


def test_evaluate_run_rejects_unsafe_run_id():
    """a run id is used in URLs and file names."""
    with pytest.raises(ValueError, match="not a valid run id"):
        evaluate_run("../../etc", metrics=[ExactMatch()])


def test_claude_judge_raises_when_cut_off():
    """parsed_output=None used to be handed to DeepEval."""
    j = ClaudeJudge(client=NS(messages=FakeMessages(stop_reason="max_tokens")))
    with pytest.raises(JudgeIncomplete, match="max_tokens"):
        j.generate("grade", schema=Verdict)

    class NoVerdict(FakeMessages):
        def parse(self, **kwargs):
            return NS(stop_reason="end_turn", stop_details=None, parsed_output=None)

    with pytest.raises(JudgeIncomplete, match="no parsable verdict"):
        ClaudeJudge(client=NS(messages=NoVerdict())).generate("grade", schema=Verdict)


class Flaky(BaseMetric):
    """Errors on inputs containing 'boom', like a judge outage."""

    _required_params = []

    def __init__(self):
        super().__init__()
        self.threshold, self.evaluation_model, self.include_reason = 0.5, None, True
        self.async_mode, self.strict_mode = False, False

    def measure(self, test_case, *args, **kwargs):
        if "boom" in test_case.input:
            raise RuntimeError("judge 529 overloaded")
        self.score, self.success, self.reason = 1.0, True, "fine"
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return bool(getattr(self, "success", False))

    @property
    def __name__(self):
        return "Flaky"


def test_metric_errors_are_not_failures(live_transport, config):
    """an errored metric counted as passed=False and lowered the pass rate."""
    run = Run(live_transport, config, mode="external").start()
    run.log(input="fine", output="a")
    run.log(input="boom", output="a")
    report = evaluate_run(run, metrics=[Flaky()], upload=False)
    errored = [s for s in report.scores if s.error]
    assert len(errored) == 1 and errored[0].passed is None
    summary = report.summary()
    assert summary.loc["Flaky", "pass_rate"] == 1.0 and summary.loc["Flaky", "errors"] == 1


def test_supersedes_from_another_run_is_caught_locally(live_sdk, live_transport, config):
    """used to fail only on the server, sending the whole evaluation to the fallback."""
    a = Run(live_transport, config, mode="external").start()
    b = Run(live_transport, config, mode="external").start()
    a.log(input="q", output="a", expected_output="a")
    b.log(input="q", output="a", expected_output="a")
    first = evaluate_run(a, metrics=[ExactMatch()])
    for value in (first, first.evaluation, first.evaluation_id):  # report, Evaluation, bare id
        with pytest.raises(ValueError, match="belongs to run"):
            evaluate_run(b, metrics=[ExactMatch()], supersedes=value)
        with pytest.raises(ValueError, match="belongs to run"):
            b.evaluation("x", supersedes=value)
    with pytest.raises(ValueError, match="not found"):
        b.evaluation("x", supersedes="no-such-evaluation")
