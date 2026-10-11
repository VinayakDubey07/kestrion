"""
kestrion serve <agent_file.py>

A WebSocket server that runs an agent and streams its events in real-time.
This implements the Real-Time Streaming Server feature via a custom TelemetryProvider.
"""

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from kestrion.core.types import Event, TelemetryProvider

try:
    import websockets
    from websockets.asyncio.server import serve
except ImportError:
    websockets = None  # type: ignore


class WebSocketStreamer(TelemetryProvider):
    """
    Subscribes to Engine events and broadcasts them over a WebSocket connection.
    This provides a completely real-time, non-polling API for external UIs.
    """
    def __init__(self, websocket: Any):
        self.websocket = websocket
        
    async def on_event(self, event: Event) -> None:
        try:
            # Broadcast the event exactly as it was folded into the DB log
            await self.websocket.send(json.dumps({
                "type": event.type.value,
                "node": event.node,
                "payload": event.payload,
            }))
        except Exception:
            # If the client disconnects mid-stream, we silently swallow the exception
            # so the agent isn't interrupted by a broken pipe.
            pass


async def _run_ws_server(agent_module: Any, host: str, port: int):
    if websockets is None:
        raise ImportError(
            "kestrion serve requires 'websockets'. Install with `pip install kestrion[serve]`."
        )

    async def handler(websocket: Any):
        print(f"Client connected: {websocket.remote_address}")
        try:
            # We wait for the client to send a JSON payload with a "prompt"
            message = await websocket.recv()
            data = json.loads(message)
            prompt = data.get("prompt", "")
            
            if not prompt:
                await websocket.send(json.dumps({"error": "No prompt provided"}))
                return
                
            # Extract the user's pre-configured Agent instance from the file
            agent = agent_module.agent
            
            # Inject our WebSocket streamer into the Engine's telemetry hook
            # This requires no modifications to the core engine; it hooks right into
            # the existing event sourcing pipeline.
            streamer = WebSocketStreamer(websocket)
            agent._engine.telemetry = streamer
            
            # Run the agent autonomously
            result = await agent.run(prompt)
            
            # Broadcast the final terminal state
            await websocket.send(json.dumps({
                "type": "RUN_COMPLETED",
                "output": result.output,
                "status": result.status.value,
            }))
            
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected cleanly.")
        except Exception as e:
            print(f"Error handling WebSocket client: {e}")

    print(f"WebSocket real-time server listening on ws://{host}:{port}")
    async with serve(handler, host, port):
        await asyncio.Future()  # run forever


def cmd_serve(args: argparse.Namespace) -> int:
    """CLI entry point for `kestrion serve`"""
    agent_file = Path(args.file)
    if not agent_file.exists():
        print(f"error: file {agent_file} not found", file=sys.stderr)
        return 1

    spec = importlib.util.spec_from_file_location("agent_module", agent_file)
    if not spec or not spec.loader:
        print(f"error: could not load {agent_file}", file=sys.stderr)
        return 1
        
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "agent"):
        print(f"error: {agent_file} must define an 'agent' variable at the module level", file=sys.stderr)
        return 1

    try:
        asyncio.run(_run_ws_server(module, args.host, args.port))
    except KeyboardInterrupt:
        print("\nServer stopped.")
    return 0
