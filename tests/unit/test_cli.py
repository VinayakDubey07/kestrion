"""
Tests for the Kestrion CLI (kestrion init / run / deploy).

All tests use subprocess fixtures and temporary directories — no mocking
of the CLI internals, which would just test whether mocks work. These
tests call the actual commands the same way a user would.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

CLI = [sys.executable, "-m", "kestrion.cli.main"]

# kestrion.cli.main lives in src/ which may not be on the test runner's
# path — use the same PYTHONPATH pattern as the rest of the test suite.
ENV_EXTRA = {"PYTHONPATH": str(Path(__file__).parent.parent.parent / "src")}


def run_cli(*args, cwd=None, env_extra=None):
    import os
    env = {**os.environ, **ENV_EXTRA, **(env_extra or {})}
    return subprocess.run(
        CLI + list(args),
        capture_output=True, text=True, cwd=cwd, env=env,
    )


# ---------------------------------------------------------------------------
# kestrion --version / --help
# ---------------------------------------------------------------------------

def test_version_flag_outputs_version():
    result = run_cli("--version")
    assert result.returncode == 0
    assert "0.2" in result.stdout


def test_help_flag_lists_all_commands():
    result = run_cli("--help")
    assert result.returncode == 0
    for cmd in ("init", "run", "deploy"):
        assert cmd in result.stdout


# ---------------------------------------------------------------------------
# kestrion init
# ---------------------------------------------------------------------------

def test_init_creates_agent_and_gitignore_in_current_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cli("init", cwd=tmpdir)
        assert result.returncode == 0
        assert (Path(tmpdir) / "agent.py").exists()
        assert (Path(tmpdir) / ".gitignore").exists()


def test_init_creates_files_in_specified_subdirectory():
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "my-agent"
        result = run_cli("init", str(target), cwd=tmpdir)
        assert result.returncode == 0
        assert (target / "agent.py").exists()


def test_init_refuses_to_overwrite_without_force():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_cli("init", cwd=tmpdir)   # first call succeeds
        result = run_cli("init", cwd=tmpdir)  # second should refuse
        assert result.returncode == 1
        assert "--force" in result.stderr or "--force" in result.stdout


def test_init_overwrites_with_force():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_cli("init", cwd=tmpdir)
        result = run_cli("init", "--force", cwd=tmpdir)
        assert result.returncode == 0


def test_init_scaffolded_agent_is_valid_python():
    import ast
    with tempfile.TemporaryDirectory() as tmpdir:
        run_cli("init", cwd=tmpdir)
        content = (Path(tmpdir) / "agent.py").read_text()
        ast.parse(content)  # raises SyntaxError if invalid


def test_init_prints_next_steps():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cli("init", cwd=tmpdir)
        assert "pip install" in result.stdout
        assert "kestrion run" in result.stdout


# ---------------------------------------------------------------------------
# kestrion run
# ---------------------------------------------------------------------------

def test_run_executes_a_simple_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "simple.py"
        script.write_text("print('hello from kestrion run')\n")
        result = run_cli("run", str(script), cwd=tmpdir)
        assert result.returncode == 0
        assert "hello from kestrion run" in result.stdout


def test_run_executes_async_main_if_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "async_agent.py"
        script.write_text(
            "import asyncio\n"
            "async def main():\n"
            "    print('async main ran')\n"
        )
        result = run_cli("run", str(script), cwd=tmpdir)
        assert result.returncode == 0
        assert "async main ran" in result.stdout


def test_run_errors_clearly_on_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cli("run", "nonexistent.py", cwd=tmpdir)
        assert result.returncode == 1
        assert "not found" in result.stderr or "not found" in result.stdout


def test_run_surfaces_script_errors_without_crashing_cli():
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "broken.py"
        script.write_text("raise ValueError('intentional error')\n")
        result = run_cli("run", str(script), cwd=tmpdir)
        assert result.returncode == 1
        assert "error" in (result.stderr + result.stdout).lower()


# ---------------------------------------------------------------------------
# kestrion deploy --target k8s
# ---------------------------------------------------------------------------

def test_deploy_k8s_generates_yaml_and_dockerfile():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cli("deploy", "--target", "k8s", "--name", "test-agent", cwd=tmpdir)
        assert result.returncode == 0
        assert (Path(tmpdir) / "test-agent-k8s.yaml").exists()
        assert (Path(tmpdir) / "Dockerfile").exists()


def test_deploy_k8s_yaml_contains_correct_name_and_namespace():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_cli("deploy", "--target", "k8s", "--name", "my-agent", "--namespace", "production", cwd=tmpdir)
        content = (Path(tmpdir) / "my-agent-k8s.yaml").read_text()
        assert "my-agent" in content
        assert "production" in content


def test_deploy_k8s_yaml_uses_provided_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_cli("deploy", "--target", "k8s", "--name", "agent",
                "--image", "registry.example.com/agent:v2", cwd=tmpdir)
        content = (Path(tmpdir) / "agent-k8s.yaml").read_text()
        assert "registry.example.com/agent:v2" in content


def test_deploy_k8s_yaml_never_contains_real_api_key():
    """
    Generated manifests must never contain an actual API key — only a
    clear placeholder that the user knows to replace. This is a security
    property, not just a style check.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        run_cli("deploy", "--target", "k8s", "--name", "agent", cwd=tmpdir)
        content = (Path(tmpdir) / "agent-k8s.yaml").read_text()
        assert "REPLACE_WITH_REAL_KEY" in content
        # No real-looking keys (simplified heuristic: no 40+ char alphanumeric strings)
        import re
        suspicious = re.findall(r'[A-Za-z0-9+/]{40,}', content)
        assert not suspicious, f"Possible real key found: {suspicious}"


def test_deploy_prints_next_steps():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cli("deploy", "--target", "k8s", "--name", "agent", cwd=tmpdir)
        assert "kubectl apply" in result.stdout
        assert "docker build" in result.stdout


def test_deploy_custom_output_filename():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_cli("deploy", "--target", "k8s", "--name", "agent",
                "--output", "custom-name.yaml", cwd=tmpdir)
        assert (Path(tmpdir) / "custom-name.yaml").exists()


def test_deploy_dockerfile_references_correct_provider():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_cli("deploy", "--target", "k8s", "--name", "agent",
                "--provider", "ollama", cwd=tmpdir)
        dockerfile = (Path(tmpdir) / "Dockerfile").read_text()
        assert "kestrion[ollama]" in dockerfile


def test_deploy_unsupported_target_errors_clearly():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cli("deploy", "--target", "heroku", cwd=tmpdir)
        assert result.returncode != 0