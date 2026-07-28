"""
Code Execution Sandbox Toolkit for Kestrion agents.
Provides a secure workspace for writing files and executing Python code
with resource limits, timeouts, and environment cleansing.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from kestrion.agent.decorators import tool


class SandboxSecurityError(Exception):
    """Raised when a security violation (e.g., path traversal) is detected in sandbox operations."""
    pass


class CodeSandboxToolkit:
    """
    A stateful code execution sandbox for Kestrion agents.
    Provides tools for executing Python code, creating scripts, and managing files
    within an isolated workspace directory.
    
    Supports two execution modes:
    - `subprocess`: Light-weight local sandbox running in a clean subprocess with stripped environment secrets.
    - `docker`: Containerized execution using Docker (requires Docker CLI to be available).
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        mode: str = "subprocess",
        timeout: int = 30,
        max_output_size: int = 10000,
    ):
        if mode not in ("subprocess", "docker"):
            raise ValueError(f"Invalid sandbox mode '{mode}'. Must be 'subprocess' or 'docker'.")

        self.mode = mode
        self.timeout = timeout
        self.max_output_size = max_output_size

        if workspace_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="kestrion_sandbox_")
            self.workspace_dir = Path(self._temp_dir.name).resolve()
        else:
            self._temp_dir = None
            self.workspace_dir = Path(workspace_dir).resolve()
            self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, filename: str) -> Path:
        """Resolve filename within workspace and strictly prevent path traversal."""
        clean_name = filename.lstrip("/\\")
        target_path = (self.workspace_dir / clean_name).resolve()
        try:
            target_path.relative_to(self.workspace_dir)
        except ValueError:
            raise SandboxSecurityError(
                f"Security violation: Path '{filename}' attempts to traverse outside sandbox workspace."
            )
        return target_path

    def _clean_environment(self) -> dict[str, str]:
        """Strip sensitive secrets (API keys, cloud credentials) from subprocess environment."""
        safe_env = {}
        allowed_keys = {
            "PATH", "PYTHONPATH", "HOME", "USER", "LANG", "LC_ALL", "TZ",
            "TERM", "TMPDIR", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"
        }
        for k, v in os.environ.items():
            upper_k = k.upper()
            if any(secret_word in upper_k for secret_word in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH", "PRIVATE")):
                continue
            if k in allowed_keys or upper_k.startswith("PYTHON"):
                safe_env[k] = v
        
        safe_env["PYTHONPATH"] = str(self.workspace_dir)
        return safe_env

    def get_tools(self) -> list[Any]:
        """Returns the list of @tool decorated methods to pass to an Agent."""
        return [self.execute_python, self.write_file, self.read_file, self.list_files]

    @tool
    async def execute_python(self, code: str) -> str:
        """
        Execute Python code in the isolated sandbox workspace.
        Captures stdout and stderr, enforces execution timeout, and isolates sensitive environment secrets.
        """
        script_path = self.workspace_dir / "_exec_script.py"
        try:
            script_path.write_text(code, encoding="utf-8")
        except Exception as e:
            return f"Failed to write script to workspace: {str(e)}"

        start_time = time.monotonic()
        
        if self.mode == "docker":
            if not shutil.which("docker"):
                return "Error: Docker mode requested but 'docker' command is not found in PATH."
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{self.workspace_dir}:/workspace",
                "-w", "/workspace",
                "--network", "none",
                "python:3.11-slim",
                "python", "_exec_script.py"
            ]
            env = None
        else:
            cmd = ["python3", str(script_path)]
            env = self._clean_environment()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_dir),
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                return f"Execution failed: Timed out after {self.timeout} seconds."

            duration = time.monotonic() - start_time
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")

            output_lines = []
            output_lines.append(f"Exit code: {proc.returncode} ({duration:.2f}s)")
            if stdout_str:
                if len(stdout_str) > self.max_output_size:
                    stdout_str = stdout_str[:self.max_output_size] + "\n... [stdout truncated]"
                output_lines.append(f"--- STDOUT ---\n{stdout_str.strip()}")
            if stderr_str:
                if len(stderr_str) > self.max_output_size:
                    stderr_str = stderr_str[:self.max_output_size] + "\n... [stderr truncated]"
                output_lines.append(f"--- STDERR ---\n{stderr_str.strip()}")

            return "\n".join(output_lines) if len(output_lines) > 1 else f"Exit code: 0 ({duration:.2f}s) (No output)"

        except Exception as e:
            return f"Failed to execute Python script: {str(e)}"
        finally:
            if script_path.exists():
                try:
                    script_path.unlink()
                except Exception:
                    pass

    @tool
    async def write_file(self, filename: str, content: str) -> str:
        """
        Write text content to a file in the sandbox workspace.
        Creates parent directories if needed. Useful for saving data files or multi-file codebases.
        """
        try:
            target_path = self._resolve_path(filename)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to '{filename}'."
        except SandboxSecurityError as e:
            return str(e)
        except Exception as e:
            return f"Failed to write file '{filename}': {str(e)}"

    @tool
    async def read_file(self, filename: str) -> str:
        """
        Read text content from a file in the sandbox workspace.
        """
        try:
            target_path = self._resolve_path(filename)
            if not target_path.exists():
                return f"Error: File '{filename}' does not exist in workspace."
            if not target_path.is_file():
                return f"Error: '{filename}' is not a file."
            content = target_path.read_text(encoding="utf-8", errors="replace")
            if len(content) > self.max_output_size:
                return content[:self.max_output_size] + "\n... [content truncated]"
            return content
        except SandboxSecurityError as e:
            return str(e)
        except Exception as e:
            return f"Failed to read file '{filename}': {str(e)}"

    @tool
    async def list_files(self) -> str:
        """
        List all files and subdirectories currently in the sandbox workspace along with file sizes.
        """
        try:
            items = []
            for path in sorted(self.workspace_dir.rglob("*")):
                if path.name == "_exec_script.py":
                    continue
                rel_path = path.relative_to(self.workspace_dir)
                if path.is_dir():
                    items.append(f"[DIR]  {rel_path}/")
                else:
                    size_kb = path.stat().st_size / 1024
                    items.append(f"[FILE] {rel_path} ({size_kb:.1f} KB)")
            if not items:
                return "Workspace is empty."
            return "\n".join(items)
        except Exception as e:
            return f"Failed to list workspace files: {str(e)}"

    def close(self) -> None:
        """Clean up temporary workspace resources."""
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
