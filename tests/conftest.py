import httpx
import pytest
from fastapi.testclient import TestClient

import llmbenchmark
from llmbenchmark.config import load_config
from llmbenchmark.transport import Transport
from server.app import create_app

AUTH = {"Authorization": "Bearer test-key"}

# Multi-team key set for isolation tests.
TEAM_KEYS = {
    "key-a": {"team": "team-a", "role": "writer", "name": "alice"},
    "key-a-reader": {"team": "team-a", "role": "reader", "name": "auditor"},
    "key-b": {"team": "team-b", "role": "writer", "name": "bob"},
}


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def config(tmp_path, monkeypatch):
    for var in ("LLMBENCHMARK_ENDPOINT", "LLMBENCHMARK_API_KEY", "LLMBENCHMARK_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    return load_config(
        endpoint="http://testserver",
        api_key="test-key",
        project="tests",
        fallback_dir=tmp_path / "fallback",
        batch_size=3,
        max_retries=2,
        capture_git=False,  # keep tests fast; lineage has its own test
    )


@pytest.fixture
def server_client():
    """A real backend (in-memory SQLite) behind FastAPI's TestClient."""
    app = create_app(database_url="sqlite://", api_keys={"test-key"})
    with TestClient(app) as client:
        yield client


@pytest.fixture
def teams_client():
    """Backend with two teams (and a read-only key) for isolation tests."""
    app = create_app(database_url="sqlite://", keys=TEAM_KEYS)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def live_transport(config, server_client):
    """SDK transport wired straight into the in-memory backend: no network."""
    # TestClient has httpx.Client's interface (it subclasses httpx2.Client in
    # recent Starlette), and the transport only duck-types it, so it can be injected.
    return Transport(config, client=server_client, sleep=lambda s: None)


@pytest.fixture
def httpx_transport(config, server_client):
    """Like live_transport, but through a real httpx.Client (what users run).

    TestClient is built on httpx2, which encodes JSON differently (e.g. it
    accepts NaN/inf), so some production bugs only show up through httpx.
    """
    def forward(request: httpx.Request) -> httpx.Response:
        target = request.url.path + (f"?{request.url.query.decode()}" if request.url.query else "")
        r = server_client.request(request.method, target, content=request.content, headers=dict(request.headers))
        return httpx.Response(r.status_code, content=r.content, headers=dict(r.headers))

    client = httpx.Client(base_url=config.endpoint, transport=httpx.MockTransport(forward))
    return Transport(config, client=client, sleep=lambda s: None)


@pytest.fixture
def mock_transport(config):
    """SDK transport pointing at a URL that respx will intercept."""
    client = httpx.Client(base_url=config.endpoint)
    return Transport(config, client=client, sleep=lambda s: None)


@pytest.fixture
def global_sdk(monkeypatch, tmp_path):
    """Isolate llmbenchmark's module-level state for tests that call init()."""
    monkeypatch.setattr(llmbenchmark, "_config", None)
    monkeypatch.setattr(llmbenchmark, "_transport", None)
    for var in ("LLMBENCHMARK_ENDPOINT", "LLMBENCHMARK_API_KEY", "LLMBENCHMARK_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


@pytest.fixture
def live_sdk(monkeypatch, config, live_transport):
    """Module-level SDK (start_run, compare, ...) wired into the in-memory backend."""
    monkeypatch.setattr(llmbenchmark, "_config", config)
    monkeypatch.setattr(llmbenchmark, "_transport", live_transport)
    return live_transport
