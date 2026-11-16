"""Manage API keys: ``python -m server.keys create|list --file keys.json``.

A new key is printed once and only its sha256 hash is stored, so a leaked
keys file doesn't leak usable keys.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from .auth import hash_key


def create_key(file: Path, *, team: str, name: str, role: str = "writer") -> str:
    if role not in ("reader", "writer"):
        raise ValueError("role must be reader or writer")
    data = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {"keys": []}
    if any(k.get("name") == name and k.get("team") == team for k in data["keys"]):
        raise ValueError(f"key {name!r} already exists in team {team!r}")
    key = "lmb_" + secrets.token_urlsafe(32)
    data["keys"].append({"name": name, "team": team, "role": role, "key_sha256": hash_key(key)})
    fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.chmod(file, 0o600)  # the mode above only applies when the file is created
    return key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m server.keys")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create a key and print it once")
    create.add_argument("--file", required=True, type=Path)
    create.add_argument("--team", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--role", default="writer", choices=["reader", "writer"])
    listing = sub.add_parser("list", help="list key names, teams and roles")
    listing.add_argument("--file", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "create":
        key = create_key(args.file, team=args.team, name=args.name, role=args.role)
        print(key)
        print(f"Saved the hash to {args.file}. This key is not shown again.", file=sys.stderr)
        return 0
    data = json.loads(args.file.read_text(encoding="utf-8"))
    for k in data.get("keys", []):
        print(f"{k['team']:20} {k['name']:20} {k['role']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
