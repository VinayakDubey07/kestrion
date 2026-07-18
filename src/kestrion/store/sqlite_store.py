"""
Reference CheckpointStore implementation. This is intentionally the
*simplest* thing that satisfies the protocol — swapping this for a
Postgres-backed store later should require zero changes to engine.py,
which is the whole point of the Protocol boundary in types.py.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3

from kestrion.core.types import AgentState, Checkpoint, Event, EventType


class SQLiteCheckpointStore:
    """Implements the CheckpointStore protocol structurally (no inheritance needed)."""

    def __init__(self, path: str = "agent_runs.db"):
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        with contextlib.closing(sqlite3.connect(self.path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    node TEXT,
                    payload TEXT NOT NULL,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0
                );
                CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq);

                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    state_blob BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id, event_seq);
                """
            )
            conn.commit()

    async def append_event(self, event: Event) -> int:
        with contextlib.closing(sqlite3.connect(self.path)) as conn:
            cur = conn.execute(
                """INSERT INTO events
                   (event_id, run_id, type, timestamp, node, payload, tokens_in, tokens_out, cost_usd)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.run_id,
                    event.type.value,
                    event.timestamp.isoformat(),
                    event.node,
                    json.dumps(event.payload),
                    event.tokens_in,
                    event.tokens_out,
                    event.cost_usd,
                ),
            )
            conn.commit()
            return cur.lastrowid or 0

    async def events_since(self, run_id: str, seq: int) -> list[Event]:
        with contextlib.closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute(
                "SELECT event_id, run_id, type, timestamp, node, payload, tokens_in, tokens_out, cost_usd "
                "FROM events WHERE run_id = ? AND seq > ? ORDER BY seq",
                (run_id, seq),
            ).fetchall()
        from datetime import datetime

        return [
            Event(
                event_id=r[0],
                run_id=r[1],
                type=EventType(r[2]),
                timestamp=datetime.fromisoformat(r[3]),
                node=r[4],
                payload=json.loads(r[5]),
                tokens_in=r[6],
                tokens_out=r[7],
                cost_usd=r[8],
            )
            for r in rows
        ]

    async def save(self, checkpoint: Checkpoint) -> None:
        try:
            state_json = json.dumps(checkpoint.state.to_dict())
        except TypeError as exc:
            # Fail loudly: a non-JSON-serializable value snuck into
            # AgentState.scratch. Silently falling back to pickle here
            # would defeat the entire point of making the format explicit.
            raise ValueError(
                f"AgentState.scratch for run {checkpoint.run_id} contains a "
                f"non-JSON-serializable value: {exc}. Only JSON-compatible "
                f"types (str, int, float, bool, None, list, dict) are "
                f"allowed in scratch."
            ) from exc
            
        with contextlib.closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO checkpoints
                   (checkpoint_id, run_id, created_at, event_seq, state_blob)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    checkpoint.checkpoint_id,
                    checkpoint.run_id,
                    checkpoint.created_at.isoformat(),
                    checkpoint.event_seq,
                    state_json,
                ),
            )
            conn.commit()

    async def latest(self, run_id: str) -> Checkpoint | None:
        with contextlib.closing(sqlite3.connect(self.path)) as conn:
            row = conn.execute(
                "SELECT checkpoint_id, created_at, event_seq, state_blob "
                "FROM checkpoints WHERE run_id = ? ORDER BY event_seq DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        from datetime import datetime

        return Checkpoint(
            checkpoint_id=row[0],
            run_id=run_id,
            state=AgentState.from_dict(json.loads(row[3])),
            created_at=datetime.fromisoformat(row[1]),
            event_seq=row[2],
        )
