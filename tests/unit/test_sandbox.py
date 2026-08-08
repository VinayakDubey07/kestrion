import pytest

from kestrion.tools.sandbox import CodeSandboxToolkit


@pytest.mark.asyncio
async def test_sandbox_lifecycle():
    with CodeSandboxToolkit(mode="subprocess") as sandbox:
        assert sandbox.workspace_dir.exists()
        assert sandbox.workspace_dir.is_dir()
    
    # Workspace should be cleaned up after exit for temporary dirs
    assert not sandbox.workspace_dir.exists()


@pytest.mark.asyncio
async def test_sandbox_file_operations():
    with CodeSandboxToolkit(mode="subprocess") as sandbox:
        # Write file
        res = await sandbox.write_file("test.txt", "hello world")
        assert "Successfully wrote 11 bytes" in res

        # Read file
        content = await sandbox.read_file("test.txt")
        assert content == "hello world"

        # List files
        files = await sandbox.list_files()
        assert "[FILE] test.txt" in files


@pytest.mark.asyncio
async def test_sandbox_path_traversal_protection():
    with CodeSandboxToolkit(mode="subprocess") as sandbox:
        # Try writing outside workspace
        res = await sandbox.write_file("../evil.txt", "hacked")
        assert "Security violation" in res

        # Try reading outside workspace
        res = await sandbox.read_file("../../etc/passwd")
        assert "Security violation" in res


@pytest.mark.asyncio
async def test_sandbox_execute_python():
    with CodeSandboxToolkit(mode="subprocess") as sandbox:
        code = "print('Hello from sandbox')\nimport sys\nprint('Error message', file=sys.stderr)"
        output = await sandbox.execute_python(code)
        assert "Exit code: 0" in output
        assert "--- STDOUT ---\nHello from sandbox" in output
        assert "--- STDERR ---\nError message" in output


@pytest.mark.asyncio
async def test_sandbox_timeout():
    with CodeSandboxToolkit(mode="subprocess", timeout=1) as sandbox:
        code = "import time\ntime.sleep(5)"
        output = await sandbox.execute_python(code)
        assert "Execution failed: Timed out after 1 seconds" in output
