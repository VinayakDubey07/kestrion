"""
In-memory CheckpointStore implementation.
Useful for tests or single-process temporary runs where durability isn't required.
"""

from __future__ import annotations

import copy
from typing import Dict, List

from kestrion.core.types import Checkpoint, Event


class MemoryCheckpointStore:
    """An in-memory CheckpointStore. Data is lost when the process exits."""

    def __init__(self):
        # run_id -> list of Events
        self.events: Dict[str, List[Event]] = {}
        # run_id -> latest Checkpoint
        self.checkpoints: Dict[str, Checkpoint] = {}
        self._seq = 0

    async def append_event(self, event: Event) -> int:
        self._seq += 1
        event_copy = copy.deepcopy(event)
        
        run_events = self.events.setdefault(event.run_id, [])
        run_events.append((self._seq, event_copy))
        return self._seq

    async def events_since(self, run_id: str, seq: int) -> list[Event]:
        run_events = self.events.get(run_id, [])
        return [evt for s, evt in run_events if s > seq]

    async def save(self, checkpoint: Checkpoint) -> None:
        # We deepcopy to avoid mutating saved state inadvertently
        self.checkpoints[checkpoint.run_id] = copy.deepcopy(checkpoint)

    async def latest(self, run_id: str) -> Checkpoint | None:
        ckpt = self.checkpoints.get(run_id)
        if ckpt:
            return copy.deepcopy(ckpt)
        return None
