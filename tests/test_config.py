from pathlib import Path

import pytest

from llmbenchmark.config import load_config


def test_env_vars_then_overrides(monkeypatch):
    monkeypatch.setenv("LLMBENCHMARK_ENDPOINT", "http://env:9000/")
    monkeypatch.setenv("LLMBENCHMARK_PROJECT", "from-env")
    monkeypatch.setenv("LLMBENCHMARK_BATCH_SIZE", "7")
    monkeypatch.setenv("LLMBENCHMARK_ENABLED", "false")
    monkeypatch.setenv("LLMBENCHMARK_FALLBACK_DIR", "/tmp/x")

    cfg = load_config(project="explicit")

    assert cfg.endpoint == "http://env:9000"  # trailing slash stripped
    assert cfg.project == "explicit"  # argument beats env var
    assert cfg.batch_size == 7
    assert cfg.enabled is False
    assert cfg.fallback_dir == Path("/tmp/x")


def test_unknown_option_rejected():
    with pytest.raises(TypeError):
        load_config(endpiont="typo")


# ---- regression tests ---------------------------------------------------------


def test_api_key_not_in_repr():
    """displaying the Config in a notebook must not leak the key."""
    cfg = load_config(api_key="sk-very-secret")
    assert "sk-very-secret" not in repr(cfg)
    assert cfg.api_key == "sk-very-secret"


def test_invalid_env_value_names_the_variable(monkeypatch):
    """a bad env var gives a clear error, not a bare int() traceback."""
    monkeypatch.setenv("LLMBENCHMARK_BATCH_SIZE", "fifty")
    with pytest.raises(ValueError, match="LLMBENCHMARK_BATCH_SIZE='fifty'"):
        load_config()


def test_invalid_bool_env_value_rejected(monkeypatch):
    monkeypatch.setenv("LLMBENCHMARK_ENABLED", "maybe")
    with pytest.raises(ValueError, match="LLMBENCHMARK_ENABLED"):
        load_config()


@pytest.mark.parametrize("option", [{"batch_size": 0}, {"max_retries": -1}, {"timeout": 0}])
def test_out_of_range_values_rejected(option):
    with pytest.raises(ValueError):
        load_config(**option)
