import json

import pytest
from fastapi.testclient import TestClient

import server.app as server_app
from llmbenchmark.schemas import EvaluationCreate, PromptVersion, Record, RunCreate, Score
from server.auth import hash_key, load_key_store
from server.keys import create_key, main as keys_main
from tests.conftest import AUTH, bearer


def _run(client, project="p", headers=AUTH):
    run = RunCreate(project=project)
    assert client.post("/v1/runs", json=run.model_dump(mode="json"), headers=headers).status_code == 201
    return run


def _post_records(client, run_id, *records, headers=AUTH):
    body = {"records": [r.model_dump(mode="json") for r in records]}
    return client.post(f"/v1/runs/{run_id}/records", json=body, headers=headers)


def _evaluation(client, run_id, headers=AUTH):
    ev = EvaluationCreate(run_id=run_id, evaluator="x", config_hash="abc")
    assert client.post(f"/v1/runs/{run_id}/evaluations", json=ev.model_dump(mode="json"), headers=headers).status_code == 201
    return ev


def test_rejects_missing_api_key(server_client):
    assert server_client.post("/v1/runs", json=RunCreate(project="p").model_dump(mode="json")).status_code == 401


def test_writes_are_idempotent_upserts(server_client):
    run = _run(server_client)
    record = Record(run_id=run.id, input="q", output="a")
    for _ in range(2):  # a retry or replay sends the same records again
        assert _post_records(server_client, run.id, record).status_code == 201
    assert len(server_client.get(f"/v1/runs/{run.id}/records", headers=AUTH).json()) == 1


def test_scores_roundtrip(server_client):
    run = _run(server_client)
    record = Record(run_id=run.id, input="q", output="a")
    _post_records(server_client, run.id, record)
    ev = _evaluation(server_client, run.id)
    score = Score(evaluation_id=ev.id, record_id=record.id, metric="Answer Relevancy", score=0.9, passed=True)
    r = server_client.post(f"/v1/evaluations/{ev.id}/scores", json={"scores": [score.model_dump(mode="json")]}, headers=AUTH)
    assert r.status_code == 201
    got = server_client.get(f"/v1/evaluations/{ev.id}/scores", headers=AUTH).json()
    assert got[0]["score"] == 0.9 and got[0]["metric"] == "Answer Relevancy"


def test_unknown_run_is_404(server_client):
    assert server_client.get("/v1/runs/nope", headers=AUTH).status_code == 404


def test_prompt_hash_must_match_content(server_client):
    wrong_hash = PromptVersion(hash="0" * 64, name="p", template="x")
    assert server_client.post("/v1/prompts", json=wrong_hash.model_dump(), headers=AUTH).status_code == 422


def test_dataset_items_are_immutable(server_client):
    body = {"hash": "a" * 64, "name": "ds", "item_count": 3}
    assert server_client.post("/v1/datasets", json=body, headers=AUTH).status_code == 201
    item = {"id": "q1", "input": "2+2?", "expected_output": "4"}
    path = f"/v1/datasets/{'a' * 64}/items"
    assert server_client.post(path, json={"items": [item]}, headers=AUTH).status_code == 201
    assert server_client.post(path, json={"items": [item]}, headers=AUTH).status_code == 201  # resend
    changed = {**item, "expected_output": "five"}
    assert server_client.post(path, json={"items": [changed]}, headers=AUTH).status_code == 409
    extra = [{"id": f"x{i}", "input": "?"} for i in range(3)]
    assert server_client.post(path, json={"items": extra}, headers=AUTH).status_code == 409  # over item_count


# ---- team isolation ---------------------------------------------------------------


def test_teams_cannot_see_each_others_data(teams_client):
    a_run = _run(teams_client, headers=bearer("key-a"))
    _post_records(teams_client, a_run.id, Record(run_id=a_run.id, input="secret prompt"), headers=bearer("key-a"))

    b = bearer("key-b")
    assert teams_client.get(f"/v1/runs/{a_run.id}", headers=b).status_code == 404
    assert teams_client.get(f"/v1/runs/{a_run.id}/records", headers=b).status_code == 404
    assert teams_client.get("/v1/runs", headers=b).json() == []
    assert _post_records(teams_client, a_run.id, Record(run_id=a_run.id, input="x"), headers=b).status_code == 404
    assert teams_client.get("/v1/audit", headers=b).json() == []

    [listed] = teams_client.get("/v1/runs", headers=bearer("key-a")).json()
    assert listed["team"] == "team-a" and listed["created_by"] == "alice"


def test_reusing_another_teams_run_id_is_refused(teams_client):
    a_run = _run(teams_client, headers=bearer("key-a"))
    r = teams_client.post("/v1/runs", json=a_run.model_dump(mode="json"), headers=bearer("key-b"))
    assert r.status_code == 409


def test_reader_keys_cannot_write(teams_client):
    run = _run(teams_client, headers=bearer("key-a"))
    reader = bearer("key-a-reader")
    assert teams_client.get(f"/v1/runs/{run.id}", headers=reader).status_code == 200  # same team: visible
    assert _post_records(teams_client, run.id, Record(run_id=run.id, input="x"), headers=reader).status_code == 403
    assert teams_client.post("/v1/runs", json=RunCreate(project="p").model_dump(mode="json"),
                             headers=reader).status_code == 403


def test_prompts_and_datasets_are_per_team(teams_client):
    from llmbenchmark.versioning import content_hash

    h = content_hash({"system": None, "template": "Q: {input}"})
    prompt = PromptVersion(hash=h, name="qa", template="Q: {input}")
    assert teams_client.post("/v1/prompts", json=prompt.model_dump(), headers=bearer("key-a")).status_code == 201
    assert teams_client.get(f"/v1/prompts/{h}", headers=bearer("key-b")).status_code == 404
    assert teams_client.post("/v1/prompts", json=prompt.model_dump(), headers=bearer("key-b")).status_code == 201


def test_keys_file_and_cli(tmp_path, capsys):
    keys_file = tmp_path / "keys.json"
    assert keys_main(["create", "--file", str(keys_file), "--team", "search", "--name", "alice"]) == 0
    key = capsys.readouterr().out.strip()
    reader_key = create_key(keys_file, team="search", name="bot", role="reader")
    stored = json.loads(keys_file.read_text())["keys"]
    assert [k["key_sha256"] for k in stored] == [hash_key(key), hash_key(reader_key)]
    assert key not in keys_file.read_text()  # only hashes on disk
    assert oct(keys_file.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(ValueError, match="already exists"):
        create_key(keys_file, team="search", name="alice")

    store = load_key_store(keys_file=keys_file)
    assert store.authenticate(key).team == "search"
    assert store.authenticate(reader_key).role == "reader"
    assert store.authenticate("lmb_wrong") is None

    app = server_app.create_app(database_url="sqlite://", keys_file=str(keys_file))
    with TestClient(app) as client:
        assert client.get("/v1/runs", headers=bearer(key)).status_code == 200
        assert client.get("/v1/runs", headers=bearer("lmb_wrong")).status_code == 401


def test_keys_file_rejects_bad_hashes(tmp_path):
    bad = tmp_path / "keys.json"
    bad.write_text(json.dumps({"keys": [{"name": "x", "team": "t", "key_sha256": "not-a-hash"}]}))
    with pytest.raises(ValueError, match="invalid key_sha256"):
        load_key_store(keys_file=bad)


# ---- regressions (AGENTS.md §14) -------------------------------------------


def test_refuses_to_start_without_keys(monkeypatch):
    """S4: auth used to be silently off when no keys were configured."""
    monkeypatch.delenv("LLMBENCHMARK_SERVER_API_KEYS", raising=False)
    monkeypatch.delenv("LLMBENCHMARK_SERVER_KEYS_FILE", raising=False)
    monkeypatch.delenv("LLMBENCHMARK_SERVER_ALLOW_ANONYMOUS", raising=False)
    with pytest.raises(RuntimeError, match="LLMBENCHMARK_SERVER_ALLOW_ANONYMOUS"):
        server_app.create_app(database_url="sqlite://")

    monkeypatch.setenv("LLMBENCHMARK_SERVER_ALLOW_ANONYMOUS", "1")
    with TestClient(server_app.create_app(database_url="sqlite://")) as anon:
        assert anon.get("/v1/runs").status_code == 200


@pytest.mark.parametrize("header,expected", [
    ("test-key", 401),          # S7: bare token without a scheme
    ("Basic test-key", 401),
    ("Bearer ", 401),
    ("Bearer wrong", 401),
    ("bearer test-key", 200),   # scheme is case-insensitive
    ("Bearer test-key", 200),
])
def test_authorization_header_parsing(server_client, header, expected):
    assert server_client.get("/v1/runs", headers={"Authorization": header}).status_code == expected


def test_cannot_take_over_a_record_from_another_run(server_client):
    """S3: posting another run's record id under your run used to move and overwrite it."""
    a, b = _run(server_client, "proj-a"), _run(server_client, "proj-b")
    record = Record(run_id=a.id, input="q", output="original")
    assert _post_records(server_client, a.id, record).status_code == 201

    hijack = record.model_copy(update={"run_id": b.id, "output": "TAMPERED"})
    assert _post_records(server_client, b.id, hijack).status_code == 409

    [kept] = server_client.get(f"/v1/runs/{a.id}/records", headers=AUTH).json()
    assert kept["output"] == "original"
    assert server_client.get(f"/v1/runs/{b.id}/records", headers=AUTH).json() == []


def test_recreating_a_run_does_not_reset_it(server_client):
    """C6 / S3: replaying POST /v1/runs used to revert a finished run to 'running'."""
    run = _run(server_client)
    server_client.patch(f"/v1/runs/{run.id}", json={"status": "completed"}, headers=AUTH)

    again = server_client.post("/v1/runs", json=run.model_dump(mode="json"), headers=AUTH)
    assert again.status_code == 200
    assert server_client.get(f"/v1/runs/{run.id}", headers=AUTH).json()["status"] == "completed"

    moved = run.model_copy(update={"project": "someone-else"})
    assert server_client.post("/v1/runs", json=moved.model_dump(mode="json"), headers=AUTH).status_code == 409


def test_score_for_unknown_record_is_rejected(server_client):
    """C5: used to be stored (SQLite) or to 500 and be retried (Postgres)."""
    a, b = _run(server_client), _run(server_client)
    other_runs_record = Record(run_id=b.id, input="q", output="a")
    _post_records(server_client, b.id, other_runs_record)
    ev = _evaluation(server_client, a.id)
    for record_id in ("does-not-exist", other_runs_record.id):
        score = Score(evaluation_id=ev.id, record_id=record_id, metric="m")
        r = server_client.post(f"/v1/evaluations/{ev.id}/scores", json={"scores": [score.model_dump(mode="json")]},
                               headers=AUTH)
        assert r.status_code == 422


def test_sqlite_enforces_foreign_keys():
    from sqlalchemy import text

    from server.db import make_engine

    with make_engine("sqlite://").connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_request_body_size_is_limited():
    """S5: arbitrarily large bodies used to be accepted."""
    app = server_app.create_app(database_url="sqlite://", api_keys={"test-key"}, max_body_bytes=2000)
    with TestClient(app) as client:
        run = _run(client)
        big = Record(run_id=run.id, input="x" * 5000)
        assert _post_records(client, run.id, big).status_code == 413
        # Chunked upload (no Content-Length) is counted too.
        chunks = iter([b"{" + b" " * 1500, b" " * 1500 + b"}"])
        r = client.post(f"/v1/runs/{run.id}/records", content=chunks,
                        headers={**AUTH, "Content-Type": "application/json"})
        assert r.status_code == 413


def test_batch_record_count_is_limited(server_client, monkeypatch):
    monkeypatch.setattr(server_app, "MAX_BATCH_RECORDS", 2)
    run = _run(server_client)
    records = [Record(run_id=run.id, input=str(i)) for i in range(3)]
    assert _post_records(server_client, run.id, *records).status_code == 413


def test_list_endpoints_paginate(server_client):
    run = _run(server_client)
    _post_records(server_client, run.id, *[Record(run_id=run.id, input=str(i)) for i in range(5)])
    page = server_client.get(f"/v1/runs/{run.id}/records", params={"limit": 2, "offset": 4}, headers=AUTH).json()
    assert len(page) == 1
    assert server_client.get("/v1/runs", params={"limit": 5000}, headers=AUTH).status_code == 422


def test_invalid_run_id_in_path_is_rejected(server_client):
    """S6: ids are restricted to a safe alphabet."""
    assert server_client.get("/v1/runs/bad.id", headers=AUTH).status_code == 422
    bad = {"project": "p", "id": "../x"}
    assert server_client.post("/v1/runs", json=bad, headers=AUTH).status_code == 422


def _nested(depth):
    value = {}
    cursor = value
    for _ in range(depth):
        cursor["a"] = {}
        cursor = cursor["a"]
    return value


def test_deeply_nested_json_is_rejected_by_the_server(server_client):
    """S-L2: deep JSON was stored, then every read of the run failed with a 500."""
    run = _run(server_client)
    body = {"records": [Record(run_id=run.id, input="q").model_dump(mode="json")]}
    body["records"][0]["metadata"] = _nested(1000)
    assert server_client.post(f"/v1/runs/{run.id}/records", json=body, headers=AUTH).status_code == 422
    assert server_client.get(f"/v1/runs/{run.id}/records", headers=AUTH).status_code == 200


def test_sdk_truncates_deep_values_instead_of_failing(live_transport, config, server_client):
    """S-L2: the SDK cuts deep nesting, so its own records are never refused."""
    from llmbenchmark.run import Run
    from llmbenchmark.schemas import TRUNCATED

    run = Run(live_transport, config, mode="external").start()
    rec = run.log(input="q", output="a", metadata={"deep": _nested(100_000)})
    run.finish()
    flat = json.dumps(rec.metadata)
    assert TRUNCATED in flat
    [stored] = server_client.get(f"/v1/runs/{run.id}/records", headers=AUTH).json()
    assert TRUNCATED in json.dumps(stored["metadata"])


def test_keys_file_requires_roles_and_unique_keys(tmp_path):
    """S-L3: an entry without a role became a writer; duplicate hashes loaded silently."""
    kf = tmp_path / "keys.json"
    kf.write_text(json.dumps({"keys": [{"name": "ro", "team": "t", "key_sha256": "a" * 64}]}))
    with pytest.raises(ValueError, match="needs role"):
        load_key_store(keys_file=kf)
    kf.write_text(json.dumps({"keys": [
        {"name": "x", "team": "t1", "role": "writer", "key_sha256": "b" * 64},
        {"name": "y", "team": "t2", "role": "reader", "key_sha256": "b" * 64},
    ]}))
    with pytest.raises(ValueError, match="configured twice"):
        load_key_store(keys_file=kf)


def test_key_cli_tightens_an_existing_keys_file(tmp_path):
    kf = tmp_path / "keys.json"
    kf.write_text(json.dumps({"keys": []}))
    kf.chmod(0o644)
    create_key(kf, team="t", name="n")
    assert oct(kf.stat().st_mode & 0o777) == "0o600"


def test_datasets_are_verified_against_their_hash(server_client, live_transport, config):
    """S-L6: the server never checked dataset hashes, and item_count could be negative."""
    import llmbenchmark
    from llmbenchmark.run import Run

    assert server_client.post("/v1/datasets", json={"hash": "c" * 64, "name": "d", "item_count": -1},
                              headers=AUTH).status_code == 422
    assert server_client.post("/v1/datasets", json={"hash": "c" * 64, "name": "d", "item_count": 0},
                              headers=AUTH).status_code == 422  # empty, but not the hash of []

    bogus = {"hash": "d" * 64, "name": "d", "item_count": 1}
    server_client.post("/v1/datasets", json=bogus, headers=AUTH)
    r = server_client.post(f"/v1/datasets/{'d' * 64}/items", json={"items": [{"id": "q", "input": "x"}]},
                           headers=AUTH)
    assert r.status_code == 422  # items don't hash to the declared value
    assert server_client.get(f"/v1/datasets/{'d' * 64}", headers=AUTH).json()["verified"] is False

    ds = llmbenchmark.Dataset("real", [{"id": "q1", "input": "2+2?", "expected_output": "4"}])
    Run(live_transport, config, mode="external", dataset=ds).start()
    assert server_client.get(f"/v1/datasets/{ds.hash}", headers=AUTH).json()["verified"] is True
