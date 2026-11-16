"""Comparing evaluations item by item, and project baselines."""

import pytest

import llmbenchmark
from llmbenchmark.run import Run

DATASET = llmbenchmark.Dataset("qa", [
    {"id": "q1", "input": "2+2?", "expected_output": "4"},
    {"id": "q2", "input": "3+3?", "expected_output": "6"},
    {"id": "q3", "input": "Capital of France?", "expected_output": "Paris"},
], version="v1")


def scored_run(transport, config, answers, *, dataset=DATASET, threshold=1, model="m"):
    with Run(transport, config, mode="external", model=model, dataset=dataset).start() as run:
        with run.evaluation("exact", metrics={"accuracy": {"threshold": threshold}}) as ev:
            for item in dataset:
                rec = run.log(item=item, output=answers[item.id])
                ev.score(rec, "accuracy", 1.0 if answers[item.id] == item.expected_output else 0.0)
    return run, ev


def test_compare_finds_regressions_and_improvements(live_sdk, config):
    _, base = scored_run(live_sdk, config, {"q1": "4", "q2": "6", "q3": "Lyon"}, model="old")
    _, head = scored_run(live_sdk, config, {"q1": "4", "q2": "7", "q3": "Paris"}, model="new")

    diff = llmbenchmark.compare(base=base, head=head)
    assert diff.warnings == []  # same dataset, same scoring config
    assert diff.differences == {"model": {"base": "old", "head": "new"}}
    status = {i["key"]: i["status"] for i in diff.items}
    assert status == {"item:q1#0": "unchanged", "item:q2#0": "regressed", "item:q3#0": "improved"}
    assert [r["input"] for r in diff.regressions] == ["3+3?"]
    summary = diff.summary()
    assert summary.loc["accuracy", "base_mean"] == pytest.approx(2 / 3)
    assert summary.loc["accuracy", "head_mean"] == pytest.approx(2 / 3)
    assert summary.loc["accuracy", "regressions"] == 1


def test_compare_warns_when_not_like_for_like(live_sdk, config):
    other = llmbenchmark.Dataset("qa", [{"id": "q1", "input": "2+2?", "expected_output": "four"}], version="v2")
    _, base = scored_run(live_sdk, config, {"q1": "4", "q2": "6", "q3": "Paris"})
    _, head = scored_run(live_sdk, config, {"q1": "4"}, dataset=other, threshold=0.5)
    diff = llmbenchmark.compare(base, head)
    assert any("different dataset versions" in w for w in diff.warnings)
    assert any("different scoring configs" in w for w in diff.warnings)
    statuses = {i["key"]: i["status"] for i in diff.items}
    assert statuses["item:q2#0"] == "removed"


def test_baseline(live_sdk, config):
    base_run, base = scored_run(live_sdk, config, {"q1": "4", "q2": "6", "q3": "Paris"})
    assert llmbenchmark.get_baseline() is None
    out = llmbenchmark.set_baseline(base)
    assert out["evaluation_id"] == base.id and out["project"] == config.project
    assert llmbenchmark.get_baseline()["run_id"] == base_run.id

    _, head = scored_run(live_sdk, config, {"q1": "5", "q2": "6", "q3": "Paris"})
    diff = llmbenchmark.compare(head=head)  # no base: the project's baseline
    assert diff.base["evaluation_id"] == base.id
    assert len(diff.regressions) == 1


def test_baseline_must_be_finalized(live_sdk, config):
    run = Run(live_sdk, config, mode="external").start()
    ev = run.evaluation("x")  # open
    with pytest.raises(llmbenchmark.APIError) as exc:
        llmbenchmark.set_baseline(ev)
    assert exc.value.status_code == 409


def test_compare_without_baseline_is_a_clear_error(live_sdk, config):
    _, head = scored_run(live_sdk, config, {"q1": "4", "q2": "6", "q3": "Paris"})
    with pytest.raises(llmbenchmark.APIError) as exc:
        llmbenchmark.compare(head=head)
    assert exc.value.status_code == 404
    assert "no baseline" in str(exc.value)


def test_compare_rejects_bad_ids(live_sdk):
    with pytest.raises(ValueError):
        llmbenchmark.compare(base="../x", head="y")


def test_errors_are_not_hidden_as_unchanged(live_sdk, config):
    """#7: base passed, head errored (e.g. judge outage) used to show as 'unchanged'."""
    evs = []
    for verdict in ("pass", "error"):
        with Run(live_sdk, config, mode="external").start() as run:
            with run.evaluation("judge") as ev:
                rec = run.log(input="same question", output="a")
                if verdict == "pass":
                    ev.score(rec, "quality", 1.0, passed=True)
                else:
                    ev.score(rec, "quality", error="judge 529 overloaded")
        evs.append(ev)
    diff = llmbenchmark.compare(*evs)
    assert [i["status"] for i in diff.items] == ["errored"]
    assert diff.summary().loc["quality", "errors"] == 1 and diff.regressions == []


def test_duplicate_items_are_matched_in_order_not_collapsed(live_sdk, config):
    """#7: records with the same input used to collapse into one, skewing means."""
    evs = []
    for answers in (["a", "b"], ["a", "x"]):
        with Run(live_sdk, config, mode="external").start() as run:
            with run.evaluation("exact", metrics={"m": {"threshold": 1}}) as ev:
                for answer in answers:  # same input twice, default sample_index
                    rec = run.log(input="repeated", output=answer)
                    ev.score(rec, "m", 1.0 if answer in ("a", "b") else 0.0)
        evs.append(ev)
    diff = llmbenchmark.compare(*evs)
    assert sorted(i["key"].split("#")[1] for i in diff.items) == ["0", "0~2"]
    assert [i["status"] for i in diff.items if i["key"].endswith("~2")] == ["regressed"]
    assert diff.summary().loc["m", "base_mean"] == 1.0 and diff.summary().loc["m", "head_mean"] == 0.5
    assert any("share a dataset item" in w for w in diff.warnings)
