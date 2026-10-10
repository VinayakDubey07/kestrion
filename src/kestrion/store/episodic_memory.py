"""
Cross-Run Episodic Memory Store.
Allows agents to save and retrieve long-term facts across multiple runs.
"""

from typing import Protocol, Any, List, Dict
import sqlite3
import json
from datetime import datetime
import uuid
import asyncio


class MemoryFact:
    def __init__(self, fact_id: str, content: str, metadata: dict[str, Any], created_at: str):
        self.fact_id = fact_id
        self.content = content
        self.metadata = metadata
        self.created_at = created_at
        
    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


class EpisodicMemoryStore(Protocol):
    """Protocol for storing and retrieving cross-run memories."""
    
    async def save_fact(self, user_id: str, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Save a new memory fact for a user."""
        ...
        
    async def search_facts(self, user_id: str, query: str, limit: int = 5) -> list[MemoryFact]:
        """Search a user's memory facts."""
        ...
        
    async def delete_fact(self, fact_id: str) -> None:
        """Delete a memory fact by ID."""
        ...


class SQLiteEpisodicMemoryStore:
    """
    SQLite implementation of EpisodicMemoryStore using FTS5 for text search.
    This provides a lightweight, dependency-free long-term memory solution.
    """
    
    def __init__(self, path: str = "kestrion_memory.db"):
        self.path = path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.path) as conn:
            # Create the main table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    fact_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_blob TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)")
            
            # Create the FTS5 virtual table for search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    content='memories',
                    content_rowid='rowid'
                )
            """)
            
            # Triggers to keep FTS table in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
            """)
            conn.commit()

    async def save_fact(self, user_id: str, content: str, metadata: dict[str, Any] | None = None) -> str:
        fact_id = f"fact_{uuid.uuid4().hex[:12]}"
        created_at = datetime.utcnow().isoformat() + "Z"
        metadata_blob = json.dumps(metadata) if metadata else "{}"
        
        def _insert():
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "INSERT INTO memories (fact_id, user_id, content, metadata_blob, created_at) VALUES (?, ?, ?, ?, ?)",
                    (fact_id, user_id, content, metadata_blob, created_at)
                )
                conn.commit()
                
        await asyncio.to_thread(_insert)
        return fact_id

    async def search_facts(self, user_id: str, query: str, limit: int = 5) -> list[MemoryFact]:
        def _search():
            with sqlite3.connect(self.path) as conn:
                # Basic full text search. We sanitize the query simply by wrapping in quotes
                # to prevent FTS syntax errors for normal text.
                safe_query = '"' + query.replace('"', '""') + '"'
                
                # If query is empty, just return latest
                if not query.strip():
                    rows = conn.execute(
                        "SELECT fact_id, content, metadata_blob, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                        (user_id, limit)
                    ).fetchall()
                else:
                    sql = """
                        SELECT m.fact_id, m.content, m.metadata_blob, m.created_at
                        FROM memories m
                        JOIN memories_fts f ON m.rowid = f.rowid
                        WHERE m.user_id = ? AND memories_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    """
                    rows = conn.execute(sql, (user_id, safe_query, limit)).fetchall()
                    
                facts = []
                for r in rows:
                    fact_id, content, meta_blob, created_at = r
                    try:
                        meta = json.loads(meta_blob)
                    except:
                        meta = {}
                    facts.append(MemoryFact(fact_id, content, meta, created_at))
                return facts
                
        return await asyncio.to_thread(_search)

    async def delete_fact(self, fact_id: str) -> None:
        def _delete():
            with sqlite3.connect(self.path) as conn:
                conn.execute("DELETE FROM memories WHERE fact_id = ?", (fact_id,))
                conn.commit()
        await asyncio.to_thread(_delete)
