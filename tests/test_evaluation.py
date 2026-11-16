"""Bring-your-own-evaluation uploads and result version control."""

from datetime import datetime, timedelta

import pytest

from llmbenchmark.evaluation import Evaluation
from llmbenchmark.run import Run
from llmbenchmark.schemas import Record, Score, utcnow
from tests.conftest import AUTH


def get(client, path, **params):
    response = client.get(path, headers=AUTH, params=params)
    assert response.status_code == 200, response.text
    return response.json()


# ---- bring your own evaluation ---------------------------------------------------


def test_byo_scores_and_aggregates(live_transport, config, server_client):
    with Run(live_transport, config, mode="external", model="my-model").start() as run:
        with run.evaluation("team-x-rubric", version="1.2",
                            metrics={"accuracy": {"threshold": 1}, "fluency": {"threshold": 0.5,
                                                                              "higher_is_better": True}}) as ev:
            for q, answer, gold in [("2+2?", "4", "4"), ("3+3?", "7", "6"), ("1+1?", "2", "2")]:
                rec = run.log(input=q, output=answer, expected_output=gold)
                ev.score(rec, "accuracy", answer == gold, reason="exact compare")
                ev.score(rec, "fluency", 0.9)
            ev.metric("accuracy", 2 / 3, n=3, details={"ci95": [0.2, 0.9]})

    assert ev.finalized and ev.uploaded
    assert ev.info.metrics[1].params == {"higher_is_better": True}  # unknown keys kept, not dropped
    info = get(server_client, f"/v1/evaluations/{ev.id}")
    assert info["status"] == "finalized" and info["source"] == "external"
    assert info["evaluator"] == "team-x-rubric" and info["evaluator_version"] == "1.2"

    summary = get(server_client, f"/v1/evaluations/{ev.id}/summary")
    by_metric = {m["metric"]: m for m in summary["metrics"]}
    assert by_metric["accuracy"]["pass_rate"] == pytest.approx(2 / 3)
    assert by_metric["accuracy"]["n"] == 3
    assert by_metric["fluency"]["pass_rate"] == 1.0  # 0.9 >= threshold 0.5
    [reported] = summary["reported"]
    assert reported["name"] == "accuracy" and reported["details"] == {"ci95": [0.2, 0.9]}

    local = ev.summary()
    assert local.loc["accuracy", "pass_rate"] == pytest.approx(2 / 3)


def test_log_results_from_rows(live_transport, config, server_client):
    rows = [
        {"id": "q1", "prompt": "2+2?", "answer": "4", "gold": "4", "correct": 1, "why": "ok"},
        {"id": "q2", "prompt": "3+3?", "answer": "7", "gold": "6", "correct": 0, "why": "off by one"},
    ]
    run = Run(live_transport, config, mode="external").start()
    ev = run.log_results(
        rows, evaluator="spreadsheet", input_col="prompt", output_col="answer", expected_col="gold",
        item_id_col="id", scores={"accuracy": "correct"}, reason_cols={"accuracy": "why"},
        metrics={"accuracy": {"threshold": 1}}, aggregates={"accuracy": 0.5},
    )
    run.finish()

    records = get(server_client, f"/v1/runs/{run.id}/records")
    assert [r["dataset_item_id"] for r in records] == ["q1", "q2"]
    scores = get(server_client, f"/v1/evaluations/{ev.id}/scores")
    assert [(s["score"], s["passed"], s["reason"]) for s in scores] == [(1.0, True, "ok"), (0.0, False, "off by one")]
    assert get(server_client, f"/v1/evaluations/{ev.id}")["status"] == "finalized"


def test_scores_reference_records_flushed_first(live_transport, config, server_client):
    """A score for a record still in the run's buffer must not be rejected."""
    run = Run(live_transport, config, mode="external").start()  # batch_size=3: record stays buffered
    ev = run.evaluation("x")
    rec = run.log(input="q", output="a")
    ev.score(rec, "m", 1.0)
    ev.finalize()
    assert ev.uploaded
    assert len(get(server_client, f"/v1/evaluations/{ev.id}/scores")) == 1


def test_bad_score_values_are_recorded_as_errors(live_transport, config):
    run = Run(live_transport, config, mode="external").start()
    ev = run.evaluation("x")
    rec = run.log(input="q", output="a")
    s = ev.score(rec, "m", "not a number")
    assert s.score is None and "not a number" in s.error
    with pytest.raises(ValueError):
        ev.metric("m", "high")


def test_exception_leaves_evaluation_open(live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()
    with pytest.raises(RuntimeError):
        with run.evaluation("x") as ev:
            ev.score(run.log(input="q", output="a"), "m", 1.0)
            raise RuntimeError("scoring script crashed")
    assert not ev.finalized
    info = get(server_client, f"/v1/evaluations/{ev.id}")
    assert info["status"] == "open"
    assert len(get(server_client, f"/v1/evaluations/{ev.id}/scores")) == 1  # kept


# ---- version control -------------------------------------------------------------


def test_config_hash_tracks_scoring_config(live_transport):
    def make(**kw):
        return Evaluation(live_transport, "r", evaluator="x", upload=False, **kw)

    assert make(metrics={"acc": {"threshold": 1}}).config_hash == make(metrics={"acc": {"threshold": 1}}).config_hash
    assert make(metrics={"acc": {"threshold": 1}}).config_hash != make(metrics={"acc": {"threshold": 0.5}}).config_hash
    assert make(version="1").config_hash != make(version="2").config_hash


def test_finalized_evaluation_rejects_new_scores(live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()
    rec = run.log(input="q", output="a")
    with run.evaluation("x") as ev:
        ev.score(rec, "m", 1.0)
    with pytest.raises(RuntimeError, match="finalized"):
        ev.score(rec, "m", 0.0)

    # Directly against the server: a score that wasn't in the finalize manifest
    # is refused, even when backdated (client timestamps are not trusted)...
    for created in (utcnow(), utcnow() - timedelta(days=365)):
        late = Score(evaluation_id=ev.id, record_id=rec.id, metric="m", score=0.0, created_at=created)
        r = server_client.post(f"/v1/evaluations/{ev.id}/scores", json={"scores": [late.model_dump(mode="json")]},
                               headers=AUTH)
        assert r.status_code == 409
    # ...changing an existing score is refused...
    changed = ev.scores[0].model_copy(update={"score": 0.0})
    r = server_client.post(f"/v1/evaluations/{ev.id}/scores", json={"scores": [changed.model_dump(mode="json")]}, headers=AUTH)
    assert r.status_code == 409
    # ...but an identical resend (retry/replay) is fine.
    same = ev.scores[0]
    r = server_client.post(f"/v1/evaluations/{ev.id}/scores", json={"scores": [same.model_dump(mode="json")]}, headers=AUTH)
    assert r.status_code == 201


def test_late_delivery_to_a_finished_run_is_accepted(live_transport, config, server_client, monkeypatch):
    """A record logged during the run but not delivered (outage) still lands after
    finish, because finish() sends a manifest of every record id it logged."""
    run = Run(live_transport, config, mode="external").start()
    sent = live_transport.send_records
    monkeypatch.setattr(live_transport, "send_records", lambda run_id, records: False)  # backend "down"
    early = run.log(input="logged during the run")
    monkeypatch.setattr(live_transport, "send_records", sent)
    run.finish()

    post = lambda rec: server_client.post(f"/v1/runs/{run.id}/records",  # noqa: E731
                                          json={"records": [rec.model_dump(mode="json")]}, headers=AUTH)
    assert post(early).status_code == 201  # late delivery of a manifest id
    assert post(early).status_code == 201  # identical resend
    assert post(early.model_copy(update={"output": "edited"})).status_code == 409


def test_closed_runs_and_evaluations_ignore_client_timestamps(live_transport, config, server_client):
    """S-M1: backdated rows used to slip into finished runs / finalized evaluations,
    and a far-future close time kept them open forever."""
    run = Run(live_transport, config, mode="external").start()
    rec = run.log(input="q", output="a")
    run.finish()
    backdated = Record(run_id=run.id, input="sneaked in", created_at=utcnow() - timedelta(days=30))
    r = server_client.post(f"/v1/runs/{run.id}/records", json={"records": [backdated.model_dump(mode="json")]},
                           headers=AUTH)
    assert r.status_code == 409
    stored = server_client.get(f"/v1/runs/{run.id}", headers=AUTH).json()
    assert abs((datetime.fromisoformat(stored["ended_at"]) - utcnow()).total_seconds()) < 60  # server clock

    ev = Evaluation(live_transport, run.id, evaluator="x").start()
    ev.score(rec, "m", 1.0)
    r = server_client.post(f"/v1/evaluations/{ev.id}/finalize", json={"finalized_at": "9999-01-01T00:00:00Z"},
                           headers=AUTH)  # extra field is ignored; the server stamps the time
    assert r.status_code == 200 and r.json()["finalized_at"].startswith(str(utcnow().year))
    late = Score(evaluation_id=ev.id, record_id=rec.id, metric="m", score=0.0)
    r = server_client.post(f"/v1/evaluations/{ev.id}/scores", json={"scores": [late.model_dump(mode="json")]},
                           headers=AUTH)
    assert r.status_code == 409


def test_runs_cannot_be_created_already_finished(server_client):
    """S-M2: a run created with status="completed" used to stay mutable forever."""
    from llmbenchmark.schemas import RunCreate

    run = RunCreate(project="p", status="completed", ended_at=utcnow())
    r = server_client.post("/v1/runs", json=run.model_dump(mode="json"), headers=AUTH)
    assert r.status_code == 201 and r.json()["status"] == "running" and r.json()["ended_at"] is None
    actions = [e["action"] for e in server_client.get("/v1/audit", headers=AUTH, params={"target_id": run.id}).json()]
    assert actions == ["run.create"]


def test_finished_run_status_cannot_change(live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()
    run.finish("completed")
    patch = lambda status: server_client.patch(f"/v1/runs/{run.id}", json={"status": status}, headers=AUTH)  # noqa: E731
    assert patch("completed").status_code == 200  # replayed finish: no-op
    assert patch("failed").status_code == 409


def test_re_evaluation_creates_a_new_version(live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()
    rec = run.log(input="q", output="a")
    with run.evaluation("rubric", version="1") as v1:
        v1.score(rec, "quality", 0.4)
    with run.evaluation("rubric", version="2", supersedes=v1) as v2:
        v2.score(rec, "quality", 0.8)

    versions = get(server_client, f"/v1/runs/{run.id}/evaluations")
    assert [(v["id"], v["superseded_by"]) for v in versions] == [(v1.id, v2.id), (v2.id, None)]
    # The old scores are still there: nothing was overwritten.
    assert get(server_client, f"/v1/evaluations/{v1.id}/scores")[0]["score"] == 0.4


def test_supersedes_must_be_same_run(live_transport, config, server_client):
    a = Run(live_transport, config, mode="external").start()
    b = Run(live_transport, config, mode="external").start()
    with a.evaluation("x") as old:
        pass
    ev = Evaluation(live_transport, b.id, evaluator="x", supersedes=old.id)
    r = server_client.post(f"/v1/runs/{b.id}/evaluations", json=ev.info.model_dump(mode="json"), headers=AUTH)
    assert r.status_code == 422


def test_audit_log_records_state_changes(live_transport, config, server_client):
    run = Run(live_transport, config, mode="external").start()
    with run.evaluation("x") as ev:
        pass
    run.finish()
    actions = [e["action"] for e in get(server_client, "/v1/audit")]
    assert {"run.create", "evaluation.create", "evaluation.finalize", "run.completed"} <= set(actions)
    entry = get(server_client, "/v1/audit", target_id=ev.id)[0]
    assert entry["actor"].startswith("key-")  # the key's name, never the key


def test_non_finite_numbers_never_block_uploads(httpx_transport, config, server_client):
    """#1: an inf/NaN score used to make httpx refuse the whole batch, and every
    replay retried it forever."""
    run = Run(httpx_transport, config, mode="external").start()
    with run.evaluation("x", metrics={"m": {"threshold": 0.5}}) as ev:
        recs = [run.log(input=f"q{i}", output="a", latency_ms=float("nan")) for i in range(3)]
        ev.score(recs[0], "m", 1.0)
        bad = ev.score(recs[1], "m", float("inf"))
        ev.score(recs[2], "m", float("nan"), threshold=float("nan"))
    run.finish()
    assert ev.uploaded
    assert bad.score is None and "finite" in bad.error
    stored = server_client.get(f"/v1/evaluations/{ev.id}/scores", headers=AUTH).json()
    assert len(stored) == 3
    with pytest.raises(ValueError, match="finite"):
        Run(httpx_transport, config, mode="external").start().evaluation("y").metric("m", float("inf"))
    assert not config.fallback_dir.exists() or not list(config.fallback_dir.glob("*.jsonl"))  # nothing fell back


def test_unencodable_payload_is_rejected_not_retried_forever(httpx_transport, config):
    """#1: a payload the HTTP client can't encode is a permanent failure."""
    from llmbenchmark.transport import _Outcome  # noqa: F401

    outcome = httpx_transport._request_with_retries("POST", "/v1/runs", {"x": float("inf")})
    assert outcome.ok is False and outcome.retryable is False


def test_numpy_like_and_string_inputs_to_score(live_transport, config):
    """#12: numpy bools and string thresholds."""
    np = pytest.importorskip("numpy")
    run = Run(live_transport, config, mode="external").start()
    ev = run.evaluation("x")
    rec = run.log(input="q", output="a")
    s1 = ev.score(rec, "m", np.True_)
    s2 = ev.score(rec, "n", 0.7, threshold="0.5")
    assert (s1.score, s1.passed) == (1.0, True)
    assert (s2.threshold, s2.passed) == (0.5, True)


def test_unfinalized_evaluation_scores_are_sent_on_finish_and_exit(live_transport, config, server_client):
    """#6: scores of an evaluation never finalized used to stay in memory forever."""
    import llmbenchmark

    run = Run(live_transport, config, mode="external").start()
    ev = run.evaluation("x")
    for i in range(2):  # fewer than batch_size: still buffered
        ev.score(run.log(input=f"q{i}", output="a"), "m", 1.0)
    run.finish()
    assert len(get(server_client, f"/v1/evaluations/{ev.id}/scores")) == 2
    assert get(server_client, f"/v1/evaluations/{ev.id}")["status"] == "open"  # not silently finalized

    later = run.evaluation("y")  # evaluating a finished run, then the kernel exits
    later.score(run.records[0], "m", 0.5)
    llmbenchmark._open_runs.add(run)
    llmbenchmark._finish_open_runs()
    assert len(get(server_client, f"/v1/evaluations/{later.id}/scores")) == 1
