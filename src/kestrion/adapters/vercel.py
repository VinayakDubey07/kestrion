"""
Adapter for the Vercel AI SDK Data Stream Protocol.
Allows Kestrion to yield real-time events that drive Generative UI React components.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator

from kestrion.agent.agent import Agent
from kestrion.core.types import Event, EventType


def _format_vercel_events(events: list[Event]) -> list[str]:
    """Helper to convert Kestrion Events into Data Stream Protocol format."""
    chunks = []

    for ev in events:

        if ev.type == EventType.LLM_CALL_COMPLETED:
            payload = ev.payload
            
            # Send text chunks
            if "text" in payload and payload["text"]:
                chunks.append(f'0:{json.dumps(payload["text"])}\n')
            
            # Send tool call starts (9:) and custom UI data (d:)
            if "tool_calls" in payload and payload["tool_calls"]:
                for tc in payload["tool_calls"]:
                    tool_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                    tool_payload = {
                        "toolCallId": tool_id,
                        "toolName": tc.get("name"),
                        "args": tc.get("arguments", {})
                    }
                    chunks.append(f'9:{json.dumps(tool_payload)}\n')
                    
                    # Generative UI Data hook
                    ui_payload = {
                        "type": "tool_call_ui",
                        "toolName": tc.get("name"),
                        "args": tc.get("arguments", {})
                    }
                    chunks.append(f'd:{json.dumps([ui_payload])}\n')

        elif ev.type == EventType.TOOL_CALL_COMPLETED:
            payload = ev.payload
            result_payload = [{
                "toolCallId": payload.get("tool_call_id", f"call_{uuid.uuid4().hex[:8]}"),
                "result": str(payload.get("output", ""))
            }]
            chunks.append(f'8:{json.dumps(result_payload)}\n')

        elif ev.type == EventType.RUN_FAILED:
            error_msg = ev.payload.get("error", "Unknown error")
            chunks.append(f'e:{json.dumps(str(error_msg))}\n')

    return chunks


async def stream_to_vercel(agent: Agent, prompt: str, run_id: str | None = None) -> AsyncGenerator[str, None]:
    """
    Yields chunks formatted in the Vercel AI SDK Data Stream Protocol.
    Use with FastAPI StreamingResponse to stream directly to Next.js.
    """
    run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    
    # Fire off the agent run in the background
    task = asyncio.create_task(agent.run(prompt, run_id=run_id))
    
    # Poll event store while the task is running
    seen_count = 0
    while not task.done():
        # Fetch all events from the start (seq > 0)
        all_events = await agent._engine.store.events_since(run_id, 0)
        new_events = all_events[seen_count:]
        
        if new_events:
            chunks = _format_vercel_events(new_events)
            for chunk in chunks:
                yield chunk
            seen_count = len(all_events)
            
        await asyncio.sleep(0.1)
        
    # Pick up any trailing events generated precisely as the task finished
    all_events = await agent._engine.store.events_since(run_id, 0)
    new_events = all_events[seen_count:]
    if new_events:
        chunks = _format_vercel_events(new_events)
        for chunk in chunks:
            yield chunk

    # If the task crashed, yield the exception as a Vercel error
    try:
        task.result()
    except Exception as e:
        yield f'e:{json.dumps(str(e))}\n'
