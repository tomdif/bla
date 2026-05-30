"""Object-file state store: in-memory + JSON-backed.

The "object file" is the unit of state — one persistent entity (hypothesis,
experiment, phase, gate, etc.) with typed metadata. The store holds many,
keyed by id, and persists to disk between sessions.

JSON-backed (not SQLite, not Postgres) on purpose: the MVP needs to be
trivially inspectable. `cat state.json` is the debugger.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ObjectFile:
    """One persistent entity in the conceptual world model.

    Fields:
      id              short stable identifier (slug). Caller-assigned.
      type            category — "hypothesis", "experiment", "phase",
                      "gate", "decision", "open_question", ... Free-form;
                      domain layers can validate.
      name            human-readable label.
      state           arbitrary domain-specific fields. The "what's true
                      about this entity right now" payload.
      confidence      [0, 1] — caller's current belief in this object.
      supported_by    ids of other objects that support this one.
      contradicted_by ids of other objects that contradict this one.
      open_questions  free-form strings the LLM/user wants to resolve.
      created_at      ISO timestamp, auto-set on construction.
      updated_at      ISO timestamp, auto-bumped by StateStore.update().
    """
    id: str
    type: str
    name: str
    state: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    supported_by: list[str] = field(default_factory=list)
    contradicted_by: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def __post_init__(self):
        if not self.id:
            raise ValueError("ObjectFile.id must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObjectFile:
        return cls(**d)


class StateStore:
    """Workspace of object files. Atomic JSON persistence.

    The store is the LLM's working memory in conceptual object-JEPA. It is
    deliberately small + readable. Mutations (`add`, `update`, `remove`)
    bump `updated_at` and append to a change log so trajectories can be
    reconstructed for Phase-3 predictor training.

    Args:
      path: optional Path on disk. If given, `load()` reads it on
            construction (if it exists) and `save()` writes atomically.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._objects: dict[str, ObjectFile] = {}
        self._change_log: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            self.load()

    # ---------- basic ops ----------
    def add(self, obj: ObjectFile) -> ObjectFile:
        if obj.id in self._objects:
            raise KeyError(f"ObjectFile id {obj.id!r} already exists; use update()")
        self._objects[obj.id] = obj
        self._log("add", obj.id, before=None, after=obj.to_dict())
        return obj

    def upsert(self, obj: ObjectFile) -> ObjectFile:
        """Add if absent, otherwise replace by id."""
        existed = obj.id in self._objects
        before = self._objects[obj.id].to_dict() if existed else None
        obj.updated_at = _now_iso()
        self._objects[obj.id] = obj
        self._log("upsert", obj.id, before=before, after=obj.to_dict())
        return obj

    def get(self, obj_id: str) -> ObjectFile:
        return self._objects[obj_id]

    def find(self, obj_id: str) -> Optional[ObjectFile]:
        return self._objects.get(obj_id)

    def update(self, obj_id: str, **changes: Any) -> ObjectFile:
        """Patch named fields on one object. Bumps updated_at."""
        if obj_id not in self._objects:
            raise KeyError(f"No object with id {obj_id!r}")
        before = self._objects[obj_id].to_dict()
        cur = self._objects[obj_id]
        for k, v in changes.items():
            if k in ("id", "created_at"):
                raise ValueError(f"cannot update {k!r} via update()")
            if not hasattr(cur, k):
                raise AttributeError(f"ObjectFile has no field {k!r}")
            setattr(cur, k, v)
        cur.updated_at = _now_iso()
        # Re-validate via __post_init__ on a freshly-constructed copy
        ObjectFile(**cur.to_dict())
        self._log("update", obj_id, before=before, after=cur.to_dict())
        return cur

    def remove(self, obj_id: str) -> None:
        if obj_id not in self._objects:
            raise KeyError(f"No object with id {obj_id!r}")
        before = self._objects[obj_id].to_dict()
        del self._objects[obj_id]
        self._log("remove", obj_id, before=before, after=None)

    # ---------- views ----------
    def all(self) -> list[ObjectFile]:
        return list(self._objects.values())

    def by_type(self, t: str) -> list[ObjectFile]:
        return [o for o in self._objects.values() if o.type == t]

    def __len__(self) -> int:
        return len(self._objects)

    def __contains__(self, obj_id: str) -> bool:
        return obj_id in self._objects

    @property
    def change_log(self) -> list[dict[str, Any]]:
        """Read-only view of mutations since construction (or last clear).

        Each entry: {op, id, before, after, at}. Phase-3 training will
        read this to assemble (state, transition) trajectories.
        """
        return list(self._change_log)

    def clear_change_log(self) -> None:
        self._change_log.clear()

    def _log(
        self,
        op: str,
        obj_id: str,
        *,
        before: Optional[dict[str, Any]],
        after: Optional[dict[str, Any]],
    ) -> None:
        self._change_log.append({
            "op": op,
            "id": obj_id,
            "before": before,
            "after": after,
            "at": _now_iso(),
        })

    # ---------- persistence ----------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "objects": [o.to_dict() for o in self._objects.values()],
        }

    def save(self) -> Path:
        if self.path is None:
            raise RuntimeError("StateStore has no `path`; nothing to save to")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp file in same dir + os.replace
        tmp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False,
            dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp",
        )
        try:
            json.dump(self.to_dict(), tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, self.path)
        except Exception:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise
        return self.path

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError(
                f"StateStore unknown version: {data.get('version')!r}")
        self._objects = {
            d["id"]: ObjectFile.from_dict(d) for d in data.get("objects", [])
        }
        # Do NOT replay change log from disk — load is a fresh start for it.
        self._change_log.clear()
