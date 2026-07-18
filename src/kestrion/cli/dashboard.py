import json
import sqlite3
import asyncio
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from kestrion.core.engine import Engine
from kestrion.store.sqlite_store import SQLiteCheckpointStore


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    db_path = "kestrion_runs.db"

    def log_message(self, format, *args):
        # Silence standard HTTP logs to keep CLI output clean
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Static files
        if path in ("/", "/index.html"):
            self.serve_static_file("dashboard.html", "text/html")
            return

        # API: analytics overview
        if path == "/api/analytics":
            self.handle_analytics()
            return

        # API: list all runs
        if path == "/api/runs":
            self.handle_list_runs()
            return

        # API: run details and sub-resources
        if path.startswith("/api/runs/"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "events":
                self.handle_run_events(parts[2])
                return
            if len(parts) == 4 and parts[3] == "export":
                self.handle_export_run(parts[2])
                return
            if len(parts) == 3:
                self.handle_run_details(parts[2])
                return

        self.send_error(404, "Not Found")

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/api/runs/") and path.endswith("/approve"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "approve":
                self.handle_approve_run(parts[2])
                return

        if path.startswith("/api/runs/") and path.endswith("/chat"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "chat":
                self.handle_chat(parts[2])
                return

        if path.startswith("/api/runs/") and path.endswith("/fork"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "fork":
                self.handle_fork_run(parts[2])
                return

        if path.startswith("/api/runs/") and path.endswith("/input"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "input":
                self.handle_provide_input(parts[2])
                return

        self.send_error(404, "Not Found")

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/runs/"):
            parts = path.strip("/").split("/")
            if len(parts) == 3:
                self.handle_delete_run(parts[2])
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

    # ------------------------------------------------------------------
    # API handlers
    # ------------------------------------------------------------------

    def handle_analytics(self):
        """Aggregated statistics across all runs."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Count runs by status from latest checkpoints
            rows = conn.execute("""
                SELECT c.state_blob
                FROM checkpoints c
                INNER JOIN (
                    SELECT run_id, MAX(event_seq) as max_seq
                    FROM checkpoints
                    GROUP BY run_id
                ) latest ON c.run_id = latest.run_id AND c.event_seq = latest.max_seq
            """).fetchall()

            status_counts = {}
            total_tokens = 0
            total_cost = 0.0
            total_runs = 0

            for (blob,) in rows:
                try:
                    state = json.loads(blob)
                    status = state.get("status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
                    total_tokens += state.get("total_tokens", 0)
                    total_cost += state.get("total_cost_usd", 0.0)
                    total_runs += 1
                except (json.JSONDecodeError, TypeError):
                    pass

            # Event type distribution
            event_rows = conn.execute(
                "SELECT type, COUNT(*) FROM events GROUP BY type ORDER BY COUNT(*) DESC"
            ).fetchall()
            event_distribution = {r[0]: r[1] for r in event_rows}

            # Total events
            total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

            conn.close()

            completed = status_counts.get("completed", 0)
            failed = status_counts.get("failed", 0)
            success_rate = (completed / (completed + failed) * 100) if (completed + failed) > 0 else 0.0

            self.send_json({
                "total_runs": total_runs,
                "status_counts": status_counts,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost,
                "total_events": total_events,
                "avg_tokens_per_run": round(total_tokens / total_runs) if total_runs > 0 else 0,
                "success_rate": round(success_rate, 1),
                "event_distribution": event_distribution,
            })
        except Exception as exc:
            self.send_error(500, f"Database error: {exc}")

    def handle_list_runs(self):
        try:
            conn = sqlite3.connect(self.db_path)
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

            # Get event counts per run
            event_counts = {}
            for r in conn.execute("SELECT run_id, COUNT(*) FROM events GROUP BY run_id").fetchall():
                event_counts[r[0]] = r[1]

            conn.close()

            runs = []
            for r in rows:
                run_id, created_at, state_blob = r
                try:
                    state = json.loads(state_blob)
                    scratch = state.get("scratch", {})
                    runs.append({
                        "run_id": run_id,
                        "task_name": scratch.get("_pipeline_task_name"),
                        "status": state.get("status", "unknown"),
                        "total_tokens": state.get("total_tokens", 0),
                        "total_cost_usd": state.get("total_cost_usd", 0.0),
                        "timestamp": created_at,
                        "event_count": event_counts.get(run_id, 0),
                        "current_node": state.get("current_node"),
                    })
                except (json.JSONDecodeError, TypeError):
                    pass

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

            if row is None:
                conn.close()
                self.send_error(404, f"Run {run_id} not found")
                return

            state = json.loads(row[0])

            # Compute duration from first to last event
            time_row = conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM events WHERE run_id = ?",
                (run_id,)
            ).fetchone()
            conn.close()

            if time_row and time_row[0] and time_row[1]:
                state["_first_event_ts"] = time_row[0]
                state["_last_event_ts"] = time_row[1]

            self.send_json(state)
        except Exception as exc:
            self.send_error(500, f"Database error: {exc}")

    def handle_run_events(self, run_id):
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT seq, event_id, type, timestamp, node, payload, tokens_in, tokens_out, cost_usd "
                "FROM events WHERE run_id = ? ORDER BY seq",
                (run_id,)
            ).fetchall()
            conn.close()

            events = [
                {
                    "seq": r[0],
                    "event_id": r[1],
                    "type": r[2],
                    "timestamp": r[3],
                    "node": r[4],
                    "payload": json.loads(r[5]),
                    "tokens_in": r[6],
                    "tokens_out": r[7],
                    "cost_usd": r[8]
                }
                for r in rows
            ]
            self.send_json(events)
        except Exception as exc:
            self.send_error(500, f"Database error: {exc}")

    def handle_export_run(self, run_id):
        """Export full run state + events as a single JSON download."""
        try:
            conn = sqlite3.connect(self.db_path)
            state_row = conn.execute(
                "SELECT state_blob FROM checkpoints WHERE run_id = ? ORDER BY event_seq DESC LIMIT 1",
                (run_id,)
            ).fetchone()

            if state_row is None:
                conn.close()
                self.send_error(404, f"Run {run_id} not found")
                return

            event_rows = conn.execute(
                "SELECT seq, event_id, type, timestamp, node, payload, tokens_in, tokens_out, cost_usd "
                "FROM events WHERE run_id = ? ORDER BY seq",
                (run_id,)
            ).fetchall()
            conn.close()

            export_data = {
                "run_id": run_id,
                "state": json.loads(state_row[0]),
                "events": [
                    {
                        "seq": r[0],
                        "event_id": r[1],
                        "type": r[2],
                        "timestamp": r[3],
                        "node": r[4],
                        "payload": json.loads(r[5]),
                        "tokens_in": r[6],
                        "tokens_out": r[7],
                        "cost_usd": r[8],
                    }
                    for r in event_rows
                ],
                "exported_at": __import__("datetime").datetime.now().isoformat(),
            }

            payload = json.dumps(export_data, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Disposition", f'attachment; filename="{run_id}_export.json"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:
            self.send_error(500, f"Database error: {exc}")

    def handle_fork_run(self, run_id):
        """Fork a run at a given sequence number."""
        data = self._read_json_body()
        if data is None:
            return

        at_seq = data.get("at_seq")
        new_run_id = data.get("new_run_id")

        if at_seq is None:
            self.send_json({"success": False, "error": "Missing 'at_seq' in request"}, status=400)
            return

        try:
            forked_id = asyncio.run(self._do_fork(run_id, int(at_seq), new_run_id))
            self.send_json({"success": True, "forked_run_id": forked_id})
        except Exception as exc:
            self.send_json({"success": False, "error": str(exc)}, status=500)

    def handle_provide_input(self, run_id):
        """Provide human input for a paused run."""
        data = self._read_json_body()
        if data is None:
            return

        text = data.get("text", "")
        tool = data.get("tool")

        if not text:
            self.send_json({"success": False, "error": "Missing 'text' in request"}, status=400)
            return

        try:
            asyncio.run(self._do_provide_input(run_id, text, tool))
            self.send_json({"success": True})
        except Exception as exc:
            self.send_json({"success": False, "error": str(exc)}, status=500)

    def handle_approve_run(self, run_id):
        data = self._read_json_body()
        if data is None:
            return

        tool = data.get("tool")
        role = data.get("role", "__any__")

        if not tool:
            self.send_json({"success": False, "error": "Missing 'tool' in request"}, status=400)
            return

        try:
            success = asyncio.run(self._persist_approval(run_id, tool, role))
            if success:
                self.send_json({"success": True})
            else:
                self.send_json({"success": False, "error": "Run not found"}, status=404)
        except Exception as exc:
            self.send_json({"success": False, "error": str(exc)}, status=500)

    def handle_delete_run(self, run_id):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
            conn.commit()
            conn.close()
            self.send_json({"success": True})
        except Exception as exc:
            self.send_json({"success": False, "error": str(exc)}, status=500)

    def handle_chat(self, run_id):
        body = self._read_json_body()
        if not body or "message" not in body:
            return
        
        try:
            asyncio.run(self._do_chat(run_id, body["message"]))
            self.send_json({"success": True})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.send_json({"success": False, "error": str(exc)}, status=500)

    # ------------------------------------------------------------------
    # Async helpers (called via asyncio.run, fixing BUG-006)
    # ------------------------------------------------------------------

    async def _persist_approval(self, run_id: str, tool: str, role: str) -> bool:
        if getattr(self, "agent", None):
            try:
                await getattr(self, "agent").approve # type: ignore(run_id, tool, True)
                await getattr(self, "agent").resume # type: ignore(run_id)
                return True
            except Exception:
                pass

        store = SQLiteCheckpointStore(path=self.db_path)
        engine = Engine(nodes={}, tools={}, store=store, entry_node="")
        try:
            await engine.approve_pending_tool(run_id, tool=tool, role=role)
            return True
        except Exception:
            return False

    async def _do_fork(self, run_id: str, at_seq: int, new_run_id: str | None) -> str:
        store = SQLiteCheckpointStore(path=self.db_path)
        engine = Engine(nodes={}, tools={}, store=store, entry_node="")
        return await engine.fork(run_id, at_seq=at_seq, new_run_id=new_run_id)

    async def _do_provide_input(self, run_id: str, text: str, tool: str | None) -> None:
        store = SQLiteCheckpointStore(path=self.db_path)
        engine = Engine(nodes={}, tools={}, store=store, entry_node="")
        await engine.provide_input(run_id, text, tool=tool)
        
        if getattr(self, "agent", None):
            await getattr(self, "agent").resume # type: ignore(run_id)

    async def _do_chat(self, run_id: str, message: str) -> None:
        if not getattr(self, "agent", None):
            raise Exception("No agent script was provided to the dashboard.")
            
        checkpoint = await self.agent._store.latest(run_id)
        if not checkpoint:
            raise Exception("Run not found")
            
        messages = checkpoint.state.scratch.get("_messages", [])
        messages.append({"role": "user", "content": message})
        
        await getattr(self, "agent").run_with_history # type: ignore(messages, run_id=run_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_json_body(self) -> dict | None:
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        try:
            return json.loads(post_data.decode('utf-8'))
        except Exception:
            self.send_json({"success": False, "error": "Invalid JSON body"}, status=400)
            return None

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def start_dashboard(db_path: str, host: str, port: int, script: str | None = None) -> None:
    # Ensure the database tables exist
    _ = SQLiteCheckpointStore(path=db_path)

    # Set the DB path on the handler class before instantiation
    DashboardHTTPHandler.db_path = db_path
    
    agent = None
    if script:
        import sys
        import importlib.util
        from pathlib import Path
        script_path = Path(script)
        if script_path.exists():
            spec = importlib.util.spec_from_file_location("__kestrion_agent__", script_path)
            if spec and spec.loader:
                sys.path.insert(0, str(script_path.parent.resolve()))
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    from kestrion.agent.agent import Agent
                    if hasattr(module, "agent") and isinstance(getattr(module, "agent"), Agent):
                        agent = getattr(module, "agent")
                    else:
                        for val in vars(module).values():
                            if isinstance(val, Agent):
                                agent = val
                                break
                except SystemExit:
                    pass
                except Exception as e:
                    print(f"Warning: Failed to load agent script {script_path}: {e}")
                    
    DashboardHTTPHandler.agent = agent # type: ignore

    server = HTTPServer((host, port), DashboardHTTPHandler)
    print(f"Kestrion Console running at http://{host}:{port}")
    print(f"Reading database: {db_path}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping console server...")
        server.server_close()
