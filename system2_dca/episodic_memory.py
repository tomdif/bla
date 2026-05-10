"""EpisodicMemory — time-indexed (observation_summary, action, outcome,
certifier_result) tuples. The agent's lived experience.

Distinct from SymbolicMemory (typed knowledge graph) and ExecutableMemory
(callable tools). Episodic memory is what the agent did and what
happened — useful for recency / salience weighted recall, replay buffers,
self-evaluation, and post-hoc commitment-object lookup.

Backed by SQLite (separate file from symbolic memory by default to keep
write paths independent). Indexed by recency and a salience score.
Salience defaults to |reward| but is overridable per-write.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    obs_summary TEXT,           -- short text or JSON describing the observation
    action TEXT,                 -- JSON-serializable action
    outcome TEXT,                -- JSON: {success: bool, reward: float, ...}
    certifier_result TEXT,       -- optional CertifierResult.to_dict() JSON
    salience REAL NOT NULL DEFAULT 0.0,
    tags TEXT,                   -- JSON list of tags for filtering
    metadata TEXT                -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_salience ON episodes(salience);
"""


@dataclass
class Episode:
    id: str
    timestamp: float
    obs_summary: Optional[str]
    action: Any
    outcome: dict
    certifier_result: Optional[dict]
    salience: float
    tags: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class EpisodicMemory:
    """Time-indexed agent experience store."""

    def __init__(self, db_path: str | Path = "~/.bla/episodic.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.db_path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA_SQL)

    def append(
        self,
        obs_summary: Optional[str],
        action: Any,
        outcome: dict,
        certifier_result: Optional[dict] = None,
        salience: Optional[float] = None,
        tags: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        ep_id = str(uuid.uuid4())
        sal = salience if salience is not None else abs(float(outcome.get("reward", 0.0)))
        self.db.execute(
            """INSERT INTO episodes
               (id, timestamp, obs_summary, action, outcome, certifier_result,
                salience, tags, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ep_id, time.time(), obs_summary,
                json.dumps(action, default=_safe_default),
                json.dumps(outcome, default=_safe_default),
                json.dumps(certifier_result, default=_safe_default) if certifier_result else None,
                sal,
                json.dumps(list(tags or [])),
                json.dumps(metadata, default=_safe_default) if metadata else None,
            ),
        )
        self.db.commit()
        return ep_id

    def recent(self, limit: int = 50, tag: Optional[str] = None) -> list[Episode]:
        if tag:
            rows = self.db.execute(
                """SELECT id, timestamp, obs_summary, action, outcome, certifier_result,
                          salience, tags, metadata
                   FROM episodes
                   WHERE tags LIKE ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (f"%{json.dumps(tag).strip(chr(34))}%", limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT id, timestamp, obs_summary, action, outcome, certifier_result,
                          salience, tags, metadata
                   FROM episodes ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def top_salience(self, limit: int = 50, tag: Optional[str] = None) -> list[Episode]:
        if tag:
            rows = self.db.execute(
                """SELECT id, timestamp, obs_summary, action, outcome, certifier_result,
                          salience, tags, metadata
                   FROM episodes
                   WHERE tags LIKE ?
                   ORDER BY salience DESC LIMIT ?""",
                (f"%{json.dumps(tag).strip(chr(34))}%", limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT id, timestamp, obs_summary, action, outcome, certifier_result,
                          salience, tags, metadata
                   FROM episodes ORDER BY salience DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def count(self, tag: Optional[str] = None) -> int:
        if tag:
            row = self.db.execute(
                "SELECT COUNT(*) FROM episodes WHERE tags LIKE ?",
                (f"%{json.dumps(tag).strip(chr(34))}%",),
            ).fetchone()
        else:
            row = self.db.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return int(row[0] if row else 0)

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def _row_to_episode(row) -> Episode:
        (id_, ts, obs, act, out, cert, sal, tags, meta) = row
        return Episode(
            id=id_, timestamp=ts, obs_summary=obs,
            action=json.loads(act) if act else None,
            outcome=json.loads(out) if out else {},
            certifier_result=json.loads(cert) if cert else None,
            salience=sal,
            tags=json.loads(tags) if tags else [],
            metadata=json.loads(meta) if meta else {},
        )


def _safe_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return repr(o)
