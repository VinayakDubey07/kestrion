"""
Shared fixtures. Kept deliberately small — each test file defines its own
nodes/tools where that makes the test easier to read in isolation, but the
store fixture is common enough to share.
"""

import tempfile
from pathlib import Path

import pytest

from kestrion.store.sqlite_store import SQLiteCheckpointStore


@pytest.fixture
def tmp_store():
    """
    A fresh SQLite-backed store per test, in a temp file (not in-memory
    sqlite, since the whole point of several tests is reopening the same
    file from what's meant to simulate an independent process).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test_runs.db")
        yield SQLiteCheckpointStore(path=path)
