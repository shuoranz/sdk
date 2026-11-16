"""Versioned inputs: prompts and datasets.

Both are identified by a content hash, so two runs can be checked for using
"the same prompt" or "the same dataset", and a result can always be traced
back to the exact text that produced it.

Construction is explicit user code, so invalid input raises here (unlike
recording, which never raises).
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .schemas import DatasetCreate, DatasetItem, DatasetRef, Message, PromptRef, PromptVersion, is_valid_id
from .versioning import content_hash


class Prompt:
    """A prompt template with an optional system prompt.

        prompt = Prompt("qa", system="Answer briefly.", template="Question: {input}", version="v3")
        prompt.render(input="What is 2+2?")         # "Question: What is 2+2?"
        prompt.messages(item)                        # [{"role": "user", "content": ...}]

    Template placeholders use ``str.format`` syntax. ``render(item)`` fills
    ``{input}`` from the dataset item plus any keys in its metadata.
    """

    def __init__(
        self,
        name: str,
        *,
        system: str | None = None,
        template: str | None = None,
        version: str | None = None,
    ) -> None:
        if not name:
            raise ValueError("Prompt needs a name")
        self.name = name
        self.system = system
        self.template = template
        self.version = version
        self.hash = content_hash({"system": system, "template": template})
        self.variables = _placeholders(template) if template else set()

    def __repr__(self) -> str:
        return f"<llmbenchmark.Prompt {self.name!r} version={self.version!r} hash={self.hash[:12]}>"

    def render(self, item: DatasetItem | Mapping[str, Any] | str | None = None, **variables: Any) -> str:
        """The user message for this item (or for explicit variables)."""
        values = _item_variables(item)
        values.update(variables)
        if self.template is None:
            if "input" not in values:
                raise ValueError("Prompt has no template; pass an item or input=...")
            return str(values["input"])
        missing = self.variables - set(values)
        if missing:
            raise KeyError(f"Prompt {self.name!r} is missing variable(s): {sorted(missing)}")
        return self.template.format_map(values)

    def messages(
        self,
        item: DatasetItem | Mapping[str, Any] | str | None = None,
        *,
        include_system: bool = False,
        **variables: Any,
    ) -> list[dict[str, str]]:
        """Chat messages for this item.

        Anthropic takes the system prompt separately (``system=prompt.system``),
        so it is left out by default. Pass ``include_system=True`` for OpenAI
        chat completions, which expect it as the first message.
        """
        out: list[dict[str, str]] = []
        if include_system and self.system:
            out.append({"role": "system", "content": self.system})
        out.append({"role": "user", "content": self.render(item, **variables)})
        return out

    def ref(self) -> PromptRef:
        return PromptRef(name=self.name, version=self.version, hash=self.hash)

    def to_wire(self) -> PromptVersion:
        return PromptVersion(
            hash=self.hash, name=self.name, version=self.version, system=self.system, template=self.template
        )


class Dataset:
    """A named, versioned list of test cases ("goldens").

        ds = Dataset("qa", [{"id": "q1", "input": "Capital of France?", "expected_output": "Paris"}],
                     version="2026-11")

    Item ids identify "the same question" across runs; they are what makes
    item-by-item comparison possible. Give stable ids when you can; otherwise
    an id is derived from the item's content.
    """

    def __init__(
        self,
        name: str,
        items: Iterable[DatasetItem | Mapping[str, Any]],
        *,
        version: str | None = None,
    ) -> None:
        if not name:
            raise ValueError("Dataset needs a name")
        self.name = name
        self.version = version
        self.items: list[DatasetItem] = [_to_item(i) for i in items]
        seen: set[str] = set()
        for item in self.items:
            if item.id in seen:
                raise ValueError(f"Dataset {name!r} has duplicate item id {item.id!r}")
            seen.add(item.id)
        self.hash = content_hash([i.model_dump(mode="json") for i in self.items])

    @classmethod
    def from_records(
        cls,
        rows: Iterable[Mapping[str, Any]] | Any,
        name: str,
        *,
        version: str | None = None,
        input_col: str = "input",
        expected_col: str | None = "expected_output",
        id_col: str | None = "id",
        system_col: str | None = None,
    ) -> Dataset:
        """Build from a list of dicts or a pandas DataFrame."""
        rows = _rows(rows)
        items = []
        for row in rows:
            item: dict[str, Any] = {"input": row[input_col]}
            if id_col and row.get(id_col) is not None:
                item["id"] = str(row[id_col])
            if expected_col and row.get(expected_col) is not None:
                item["expected_output"] = str(row[expected_col])
            if system_col and row.get(system_col) is not None:
                item["system"] = str(row[system_col])
            used = {input_col, expected_col, id_col, system_col}
            item["metadata"] = {k: v for k, v in row.items() if k not in used}
            items.append(item)
        return cls(name, items, version=version)

    def __repr__(self) -> str:
        return f"<llmbenchmark.Dataset {self.name!r} version={self.version!r} items={len(self)} hash={self.hash[:12]}>"

    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> DatasetItem:
        return self.items[index]

    def ref(self) -> DatasetRef:
        return DatasetRef(name=self.name, version=self.version, hash=self.hash, item_count=len(self))

    def to_wire(self) -> DatasetCreate:
        return DatasetCreate(hash=self.hash, name=self.name, version=self.version, item_count=len(self))


# ---- helpers ---------------------------------------------------------------


def _placeholders(template: str) -> set[str]:
    names = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name:
            names.add(field_name.split(".")[0].split("[")[0])
    return names


def _item_variables(item: DatasetItem | Mapping[str, Any] | str | None) -> dict[str, Any]:
    if item is None:
        return {}
    if isinstance(item, str):
        return {"input": item}
    if isinstance(item, DatasetItem):
        return {**item.metadata, "input": item.input}
    data = dict(item)
    return {**(data.get("metadata") or {}), **data}


def _to_item(raw: DatasetItem | Mapping[str, Any]) -> DatasetItem:
    if isinstance(raw, DatasetItem):
        return raw
    data = dict(raw)
    if "input" not in data:
        raise ValueError(f"Dataset item needs an 'input': {data!r}")
    if data.get("messages"):
        data["messages"] = [m if isinstance(m, Message) else Message(**m) for m in data["messages"]]
    if data.get("id") is None:
        data["id"] = content_hash(
            {"input": data["input"], "system": data.get("system"), "messages": _plain(data.get("messages"))}
        )[:32]
    else:
        data["id"] = str(data["id"])
        if not is_valid_id(data["id"]):
            raise ValueError(f"Dataset item id {data['id']!r} must match [A-Za-z0-9_-]{{1,128}}")
    known = set(DatasetItem.model_fields)
    extra = {k: v for k, v in data.items() if k not in known}
    if extra:
        data = {k: v for k, v in data.items() if k in known}
        data["metadata"] = {**extra, **(data.get("metadata") or {})}
    return DatasetItem(**data)


def _plain(messages: Any) -> Any:
    if not messages:
        return None
    return [m.model_dump() if isinstance(m, Message) else m for m in messages]


def _rows(rows: Any) -> list[dict[str, Any]]:
    """Rows as dicts; empty DataFrame cells (NaN, pd.NA, NaT) become None, not "nan"."""
    if hasattr(rows, "to_dict"):  # pandas DataFrame, without importing pandas
        rows = rows.to_dict("records")
    return [{k: (None if is_missing(v) else v) for k, v in dict(r).items()} for r in rows]


def is_missing(value: Any) -> bool:
    """None, a float NaN, or pandas' NA/NaT (checked by type name: no pandas import)."""
    if value is None:
        return True
    if isinstance(value, float):
        return value != value
    return type(value).__name__ in ("NAType", "NaTType")
