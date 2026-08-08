"""
Tools for Auto-SWE to read, write, and execute code within the sandbox.
"""

from pathlib import Path
from kestrion.agent.decorators import tool
from kestrion.core.types import AgentState
from kestrion.tools.sandbox import CodeSandboxToolkit

# We assume a global sandbox instance will be attached to the state scratchpad
# Or we can just create a module-level sandbox for simplicity in this example.

import os
sandbox_dir = str(Path(__file__).parent.resolve() / "workspace")
os.makedirs(sandbox_dir, exist_ok=True)
sandbox = CodeSandboxToolkit(mode="subprocess", workspace_dir=sandbox_dir)

@tool
async def run_shell(command: str) -> str:
    """
    Execute a shell command in the workspace. Use this to run tests (e.g. pytest),
    list files (ls), git commands, or python scripts.
    """
    try:
        result = await sandbox.execute_command(command)
        if result["exit_code"] != 0:
            return f"Command failed with exit code {result['exit_code']}\nStdout:\n{result['stdout']}\nStderr:\n{result['stderr']}"
        return f"Stdout:\n{result['stdout']}\nStderr:\n{result['stderr']}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

@tool
async def read_file(filepath: str) -> str:
    """Read the contents of a file in the workspace."""
    try:
        return await sandbox.read_file(filepath)
    except Exception as e:
        return f"Error reading file {filepath}: {str(e)}"

@tool
async def write_file(filepath: str, content: str) -> str:
    """Write or overwrite a file in the workspace with new content."""
    try:
        await sandbox.write_file(filepath, content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing to {filepath}: {str(e)}"

@tool
async def patch_file(filepath: str, search_string: str, replacement_string: str) -> str:
    """
    Replace all exact occurrences of `search_string` with `replacement_string` in the specified file.
    Use this for targeted edits instead of rewriting the entire file.
    """
    try:
        content = await sandbox.read_file(filepath)
        if search_string not in content:
            return f"Error: Could not find exact search_string in {filepath}. Make sure indentation and whitespace matches exactly."
        
        new_content = content.replace(search_string, replacement_string)
        await sandbox.write_file(filepath, new_content)
        return f"Successfully patched {filepath}"
    except Exception as e:
        return f"Error patching {filepath}: {str(e)}"

@tool(requires_approval=True)
def ask_human_for_help(question: str) -> str:
    """
    If you are completely stuck on a bug or need clarification on requirements,
    use this tool to ask the human for help. 
    """
    # The agent pauses for approval, and the human provides feedback.
    return "Human acknowledged the question."

def get_tools() -> list:
    return [run_shell, read_file, write_file, patch_file, ask_human_for_help]
