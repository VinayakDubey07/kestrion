import json
import sqlite3
import sys
import asyncio
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone

from kestrion.core.types import Checkpoint, new_id
from kestrion.core.engine import Engine
from kestrion.store.sqlite_store import SQLiteCheckpointStore


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    db_path = "kestrion_runs.db"

    def log_message(self, format, *args):
        # Silence standard HTTP logs to keep CLI output clean
        pass

    def do_GET(self):
        # Static files
        if self.path in ("/", "/index.html"):
            self.serve_static_file("dashboard.html", "text/html")
            return

        # API: list all runs
        if self.path == "/api/runs":
            self.handle_list_runs()
            return

        # API: run details
        if self.path.startswith("/api/runs/"):
            parts = self.path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "events":
                self.handle_run_events(parts[2])
                return
            if len(parts) == 3:
                self.handle_run_details(parts[2])
                return

        self.send_error(404, "Not Found")

    def do_POST(self):
        # API: approve pending tool
        if self.path.startswith("/api/runs/") and self.path.endswith("/approve"):
            parts = self.path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "approve":
                self.handle_approve_run(parts[2])
                return

        self.send_error(404, "Not Found")

    def serve_static_file(self, filename, content_type):
        template_dir = Path(__file__).parent / "templates"
        file_path = template_dir / filename
        if not file_path.exists():
            self.send_error(404, "File Not Found")
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(file_path.read_bytes())

    def handle_list_runs(self):
        try:
            conn = sqlite3.connect(self.db_path)
            # Fetch latest checkpoint for each run
            query = """
                SELECT c.run_id, c.created_at, c.state_blob
                FROM checkpoints c
                INNER JOIN (
                    SELECT run_id, MAX(event_seq) as max_seq
                    FROM checkpoints
                    GROUP BY run_id
                ) latest ON c.run_id = latest.run_id AND c.event_seq = latest.max_seq
                ORDER BY c.created_at DESC
            """
            rows = conn.execute(query).fetchall()
            conn.close()

            runs = []
            for r in rows:
                run_id, created_at, state_blob = r
                state = json.loads(state_blob)
                runs.append({
                    "run_id": run_id,
                    "status": state.get("status", "unknown"),
                    "total_tokens": state.get("total_tokens", 0),
                    "total_cost_usd": state.get("total_cost_usd", 0.0),
                    "timestamp": created_at
                })

            self.send_json(runs)
        except Exception as exc:
            self.send_error(500, f"Database error: {exc}")

    def handle_run_details(self, run_id):
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT state_blob FROM checkpoints WHERE run_id = ? ORDER BY event_seq DESC LIMIT 1",
                (run_id,)
            ).fetchone()
            conn.close()

            if row is None:
                self.send_error(404, f"Run {run_id} not found")
                return

            self.send_json(json.loads(row[0]))
        except Exception as exc:
            self.send_error(500, f"Database error: {exc}")

    def handle_run_events(self, run_id):
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT event_id, type, timestamp, node, payload, tokens_in, tokens_out, cost_usd "
                "FROM events WHERE run_id = ? ORDER BY seq",
                (run_id,)
            ).fetchall()
            conn.close()

            events = [
                {
                    "event_id": r[0],
                    "type": r[1],
                    "timestamp": r[2],
                    "node": r[3],
                    "payload": json.loads(r[4]),
                    "tokens_in": r[5],
                    "tokens_out": r[6],
                    "cost_usd": r[7]
                }
                for r in rows
            ]
            self.send_json(events)
        except Exception as exc:
            self.send_error(500, f"Database error: {exc}")

    def handle_approve_run(self, run_id):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
        except Exception:
            self.send_json({"success": False, "error": "Invalid JSON body"}, status=400)
            return

        tool = data.get("tool")
        role = data.get("role", "__any__")

        if not tool:
            self.send_json({"success": False, "error": "Missing 'tool' in request"}, status=400)
            return

        # Perform approval persistence asynchronously using stdlib-compatible loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(self._persist_approval(run_id, tool, role))
            loop.close()
            
            if success:
                self.send_json({"success": True})
            else:
                self.send_json({"success": False, "error": "Run not found"}, status=404)
        except Exception as exc:
            self.send_json({"success": False, "error": str(exc)}, status=500)

    async def _persist_approval(self, run_id: str, tool: str, role: str) -> bool:
        store = SQLiteCheckpointStore(path=self.db_path)
        checkpoint = await store.latest(run_id)
        if checkpoint is None:
            return False

        state = checkpoint.state
        Engine.record_approval(state, tool, role)
        
        # Advance the status back to RUNNING if it was waiting on human
        # (This aligns with Engine.resume() checking whether wait is over)
        from kestrion.core.types import RunStatus
        if state.status == RunStatus.WAITING_ON_HUMAN:
            state.status = RunStatus.RUNNING
        
        new_ckpt = Checkpoint(
            checkpoint_id=new_id("ckpt"),
            run_id=run_id,
            state=state,
            created_at=datetime.now(timezone.utc),
            event_seq=state.last_event_seq
        )
        await store.save(new_ckpt)
        return True

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def start_dashboard(db_path: str, host: str, port: int) -> None:
    # Set the DB path on the handler class before instantiation
    DashboardHTTPHandler.db_path = db_path
    
    server = HTTPServer((host, port), DashboardHTTPHandler)
    print(f"Kestrion Console running at http://{host}:{port}")
    print(f"Reading database: {db_path}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping console server...")
        server.server_close()
