import sys
import os
import json
import tempfile
import urllib.request
import threading
import time
import asyncio
from pathlib import Path
import pytest
from datetime import datetime, timezone

from kestrion.core.types import Event, EventType, AgentState, Checkpoint, new_id
from kestrion.store.sqlite_store import SQLiteCheckpointStore

# PYTHONPATH setup
ENV_EXTRA = {"PYTHONPATH": str(Path(__file__).parent.parent.parent / "src")}
CLI = [sys.executable, "-m", "kestrion.cli.main"]

def run_cli(*args, env_extra=None):
    import subprocess
    env = {**os.environ, **ENV_EXTRA, **(env_extra or {})}
    return subprocess.run(
        CLI + list(args),
        capture_output=True, text=True, env=env,
    )

@pytest.fixture
def temp_db_with_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        store = SQLiteCheckpointStore(path=db_path)
        
        run_id = "test_run_123"
        state = AgentState(run_id=run_id, total_tokens=100, total_cost_usd=0.005)
        
        # Save start event
        evt_start = Event.create(run_id=run_id, type=EventType.RUN_STARTED)
        # Save message received event
        evt_msg = Event.create(run_id=run_id, type=EventType.MESSAGE_RECEIVED, payload={"content": "hello agent"})
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        loop.run_until_complete(store.append_event(evt_start))
        seq2 = loop.run_until_complete(store.append_event(evt_msg))
        
        state.last_event_seq = seq2
        checkpoint = Checkpoint(
            checkpoint_id=new_id("ckpt"),
            run_id=run_id,
            state=state,
            created_at=datetime.now(timezone.utc),
            event_seq=seq2
        )
        loop.run_until_complete(store.save(checkpoint))
        loop.close()
        
        yield db_path, run_id

def test_trace_command(temp_db_with_run):
    db_path, run_id = temp_db_with_run
    result = run_cli("trace", run_id, "--store", db_path)
    
    assert result.returncode == 0
    assert run_id in result.stdout
    assert "RUN_STARTED" in result.stdout
    assert "hello agent" in result.stdout

def test_dashboard_api(temp_db_with_run):
    db_path, run_id = temp_db_with_run
    
    from kestrion.cli.dashboard import DashboardHTTPHandler
    DashboardHTTPHandler.db_path = db_path
    
    from http.server import HTTPServer
    server = HTTPServer(("127.0.0.1", 0), DashboardHTTPHandler)
    port = server.server_port
    
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    
    time.sleep(0.5) # Let the server start
    
    try:
        # Test /api/runs
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs") as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert len(data) == 1
            assert data[0]["run_id"] == run_id
            
        # Test /api/runs/<run_id>
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/{run_id}") as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert data["run_id"] == run_id
            assert data["total_tokens"] == 100
            
        # Test /api/runs/<run_id>/events
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/{run_id}/events") as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert len(data) == 2
            assert data[0]["type"] == "run_started"
            assert data[1]["type"] == "message_received"
            
        # Test POST /api/runs/<run_id>/approve
        req_data = json.dumps({"tool": "deploy", "role": "manager"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/runs/{run_id}/approve",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert data["success"] is True
            
        # Verify that state.scratch["_approved_tools"] was updated correctly
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/{run_id}") as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert data["scratch"]["_approved_tools"]["deploy"] == ["manager"]
    finally:
        server.shutdown()
        server.server_close()
