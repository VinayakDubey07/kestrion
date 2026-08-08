"""
Production-grade CheckpointStore implementation using PostgreSQL.
Uses asyncpg for high-performance async database access and connection pooling.
"""

from __future__ import annotations

import json

import asyncpg  # type: ignore

from kestrion.core.types import AgentState, Checkpoint, Event, EventType


class PostgresCheckpointStore:
    """
    PostgreSQL-backed implementation of the CheckpointStore protocol.
    Designed for concurrent, multi-worker deployments.
    """

    def __init__(self, dsn: str):
        """
        Initialize the store with a Postgres connection string.
        Note: You MUST call `await store.setup()` before using the store,
        as asyncpg requires an async context to establish the connection pool.
        """
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def setup(self) -> None:
        """Initialize the connection pool and create tables if they do not exist."""
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(self.dsn)
        
        # Ensure tables exist
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    seq SERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    node TEXT,
                    payload JSONB NOT NULL,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    cost_usd DOUBLE PRECISION DEFAULT 0.0
                );
                CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq);

                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    event_seq INTEGER NOT NULL,
                    state_blob JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_run_seq ON checkpoints(run_id, event_seq);
            """)

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "PostgresCheckpointStore pool is not initialized. "
                "Did you forget to call `await store.setup()`?"
            )
        return self._pool

    async def append_event(self, event: Event) -> int:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            seq = await conn.fetchval(
                """
                INSERT INTO events 
                (event_id, run_id, type, timestamp, node, payload, tokens_in, tokens_out, cost_usd)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9)
                RETURNING seq
                """,
                event.event_id,
                event.run_id,
                event.type.value,
                event.timestamp,
                event.node,
                json.dumps(event.payload),
                event.tokens_in,
                event.tokens_out,
                event.cost_usd,
            )
            return seq

    async def events_since(self, run_id: str, seq: int) -> list[Event]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event_id, run_id, type, timestamp, node, payload, tokens_in, tokens_out, cost_usd
                FROM events 
                WHERE run_id = $1 AND seq > $2 
                ORDER BY seq ASC
                """,
                run_id,
                seq,
            )
            
            return [
                Event(
                    event_id=row["event_id"],
                    run_id=row["run_id"],
                    type=EventType(row["type"]),
                    timestamp=row["timestamp"],
                    node=row["node"],
                    # asyncpg parses JSONB if a custom decoder isn't set, or returns string depending on configuration.
                    # json.loads() is safe if it's a string.
                    payload=json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"],
                    tokens_in=row["tokens_in"],
                    tokens_out=row["tokens_out"],
                    cost_usd=row["cost_usd"],
                )
                for row in rows
            ]

    async def events_up_to(self, run_id: str, max_seq: int) -> list[Event]:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event_id, run_id, type, timestamp, node, payload, tokens_in, tokens_out, cost_usd
                FROM events 
                WHERE run_id = $1 AND seq <= $2 
                ORDER BY seq ASC
                """,
                run_id,
                max_seq,
            )
            
            return [
                Event(
                    event_id=row["event_id"],
                    run_id=row["run_id"],
                    type=EventType(row["type"]),
                    timestamp=row["timestamp"],
                    node=row["node"],
                    payload=json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"],
                    tokens_in=row["tokens_in"],
                    tokens_out=row["tokens_out"],
                    cost_usd=row["cost_usd"],
                )
                for row in rows
            ]

    async def save(self, checkpoint: Checkpoint) -> None:
        pool = self._get_pool()
        try:
            state_json = json.dumps(checkpoint.state.to_dict())
        except TypeError as exc:
            raise ValueError(
                f"AgentState.scratch for run {checkpoint.run_id} contains a "
                f"non-JSON-serializable value: {exc}. Only JSON-compatible "
                f"types (str, int, float, bool, None, list, dict) are "
                f"allowed in scratch."
            ) from exc

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO checkpoints 
                (checkpoint_id, run_id, created_at, event_seq, state_blob)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (checkpoint_id) DO UPDATE SET
                    run_id = EXCLUDED.run_id,
                    created_at = EXCLUDED.created_at,
                    event_seq = EXCLUDED.event_seq,
                    state_blob = EXCLUDED.state_blob
                """,
                checkpoint.checkpoint_id,
                checkpoint.run_id,
                checkpoint.created_at,
                checkpoint.event_seq,
                state_json,
            )

    async def latest(self, run_id: str) -> Checkpoint | None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT checkpoint_id, created_at, event_seq, state_blob
                FROM checkpoints 
                WHERE run_id = $1 
                ORDER BY event_seq DESC 
                LIMIT 1
                """,
                run_id,
            )
            
            if row is None:
                return None
                
            state_dict = json.loads(row["state_blob"]) if isinstance(row["state_blob"], str) else row["state_blob"]
            
            return Checkpoint(
                checkpoint_id=row["checkpoint_id"],
                run_id=run_id,
                state=AgentState.from_dict(state_dict),
                created_at=row["created_at"],
                event_seq=row["event_seq"],
            )
