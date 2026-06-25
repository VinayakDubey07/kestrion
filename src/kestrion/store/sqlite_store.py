"""
Reference CheckpointStore implementation. This is intentionally the
*simplest* thing that satisfies the protocol — swapping this for a
Postgres-backed store later should require zero changes to engine.py,
which is the whole point of the Protocol boundary in types.py.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path

from kestrion.core.types import Checkpoint, Event, EventType


class SQLiteCheckpointStore:
    """Implements the CheckpointStore protocol structurally (no inheritance needed)."""

    def __init__(self, path: str = "agent_runs.db"):
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.path)
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
        conn.close()

    async def append_event(self, event: Event) -> int:
        conn = sqlite3.connect(self.path)
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
        seq = cur.lastrowid
        conn.close()
        return seq

    async def events_since(self, run_id: str, seq: int) -> list[Event]:
        conn = sqlite3.connect(self.path)
        rows = conn.execute(
            "SELECT event_id, run_id, type, timestamp, node, payload, tokens_in, tokens_out, cost_usd "
            "FROM events WHERE run_id = ? AND seq > ? ORDER BY seq",
            (run_id, seq),
        ).fetchall()
        conn.close()
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
        conn = sqlite3.connect(self.path)
        conn.execute(
            """INSERT OR REPLACE INTO checkpoints
               (checkpoint_id, run_id, created_at, event_seq, state_blob)
               VALUES (?, ?, ?, ?, ?)""",
            (
                checkpoint.checkpoint_id,
                checkpoint.run_id,
                checkpoint.created_at.isoformat(),
                checkpoint.event_seq,
                pickle.dumps(checkpoint.state),
            ),
        )
        conn.commit()
        conn.close()

    async def latest(self, run_id: str) -> Checkpoint | None:
        conn = sqlite3.connect(self.path)
        row = conn.execute(
            "SELECT checkpoint_id, created_at, event_seq, state_blob "
            "FROM checkpoints WHERE run_id = ? ORDER BY event_seq DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        from datetime import datetime

        return Checkpoint(
            checkpoint_id=row[0],
            run_id=run_id,
            state=pickle.loads(row[3]),
            created_at=datetime.fromisoformat(row[1]),
            event_seq=row[2],
        )
