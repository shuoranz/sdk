"""API keys -> principals (who is calling, which team, which role).

Keys are stored only as sha256 hashes. Each key belongs to one team and has a
role: ``reader`` (GET only) or ``writer`` (everything). Every query the API
runs is filtered by the caller's team, so teams never see each other's data.

Configure keys with a JSON file (``LLMBENCHMARK_SERVER_KEYS_FILE``):

    {"keys": [{"name": "alice", "team": "search", "role": "writer", "key_sha256": "<hex>"}]}

Create one (prints the key once, stores only its hash):

    python -m server.keys create --file keys.json --team search --name alice [--role reader]

For single-team development, ``LLMBENCHMARK_SERVER_API_KEYS=key1,key2`` still
works: those keys are writers in team "default".
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["reader", "writer"]
DEFAULT_TEAM = "default"


@dataclass(frozen=True)
class Principal:
    name: str
    team: str
    role: Role

    @property
    def can_write(self) -> bool:
        return self.role == "writer"


ANONYMOUS = Principal(name="anonymous", team=DEFAULT_TEAM, role="writer")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KeyStore:
    def __init__(self, entries: Iterable[tuple[str, Principal]] = ()) -> None:
        self._entries = [(bytes.fromhex(h), p) for h, p in entries]
        seen: set[bytes] = set()
        for digest, principal in self._entries:
            if digest in seen:  # which team/role would win would be an accident
                raise ValueError(f"the same API key is configured twice (again for {principal.name!r})")
            seen.add(digest)

    def __len__(self) -> int:
        return len(self._entries)

    def authenticate(self, token: str) -> Principal | None:
        if not token:
            return None
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        found: Principal | None = None
        for stored, principal in self._entries:  # no early exit: constant time per key
            if hmac.compare_digest(digest, stored):
                found = principal
        return found


def load_key_store(
    *,
    api_keys: Iterable[str] = (),
    keys: Mapping[str, Mapping[str, str]] | None = None,
    keys_file: str | Path | None = None,
) -> KeyStore:
    """Combine the three key sources into one store."""
    entries: list[tuple[str, Principal]] = []
    for key in api_keys:
        h = hash_key(key)
        entries.append((h, Principal(name=f"key-{h[:8]}", team=DEFAULT_TEAM, role="writer")))
    for key, info in (keys or {}).items():  # raw keys, used by tests and embedding apps
        h = hash_key(key)
        entries.append((h, _principal(info, default_name=f"key-{h[:8]}")))
    if keys_file:
        for entry in _read_keys_file(Path(keys_file)):
            entries.append((entry["key_sha256"], _principal(entry, default_name=entry.get("name", "unnamed"))))
    return KeyStore(entries)


def _principal(info: Mapping[str, str], *, default_name: str) -> Principal:
    # No default role: an entry that forgot it must not silently become a writer.
    role = info.get("role")
    if role not in ("reader", "writer"):
        raise ValueError(f"key {info.get('name') or default_name!r} needs role 'reader' or 'writer', got {role!r}")
    team = info.get("team")
    if not team:
        raise ValueError("every key needs a team")
    return Principal(name=info.get("name") or default_name, team=team, role=role)  # type: ignore[arg-type]


def _read_keys_file(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("keys", []) if isinstance(data, dict) else []
    for e in entries:
        h = e.get("key_sha256", "")
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError(f"key {e.get('name')!r} in {path} has an invalid key_sha256")
    return entries
