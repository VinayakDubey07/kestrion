import argparse
import asyncio
import os
import sys

from kestrion.agent.agent import Agent
from kestrion.llm.ollama_provider import OllamaProvider
from tools import get_tools, sandbox

# Persona defining how the Auto-SWE agent behaves
SYSTEM_PROMPT = """You are an elite, autonomous Software Engineer. 
Your workspace is completely sandboxed. You have tools to run shell commands, read files, and write code.

CRITICAL RULE: You MUST use the provided tool calling API natively. DO NOT write raw JSON blocks like `{"name": "run_shell"}` in your text response. You must actually invoke the function through the tool calling interface!

Follow this strict workflow:
1. EXPLORE: If given a repository or a path, use `run_shell` (e.g., `ls -la`, `cat`, `grep`) to explore the codebase and understand the structure.
2. PLAN: Think step-by-step about what needs to be changed to fulfill the user's task.
3. EXECUTE: Use `write_file` or `patch_file` to modify the source code.
4. VERIFY: You MUST run the test suite or execute the script via `run_shell` to verify your fix works. If it fails, read the error and try again.
5. FINISH: Only return a final answer to the user once you have verified the fix is working.

If you are completely stuck after multiple tries, use `ask_human_for_help`.
"""
import json
import os
import sys
import re
from dataclasses import replace
from kestrion.llm.base import ToolCallRequest
from kestrion.llm.openai_provider import OpenAIProvider

class RobustOllamaProvider(OllamaProvider):
    """
    A wrapper around OllamaProvider that intercepts raw JSON strings 
    hallucinated in the text response and converts them into actual ToolCallRequests.
    This fixes the issue where small models like qwen2.5-coder:7b fail to use the native API.
    """
    async def complete(self, messages, tools, system=None, output_schema=None):
        response = await super().complete(messages, tools, system, output_schema)
        
        # If no native tool calls were parsed, check the text for hallucinated JSON
        if not response.tool_calls and response.text:
            text = response.text
            extracted_data = None
            
            # 1. Try to find a markdown json block
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                try:
                    extracted_data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
                    
            # 2. Try to find raw JSON brackets if no markdown block
            if not extracted_data and '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}')
                try:
                    extracted_data = json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    pass
                    
            # If we successfully extracted JSON and it looks like a tool call...
            if extracted_data and "name" in extracted_data:
                # Some models output "arguments", some output "parameters"
                args = extracted_data.get("arguments", extracted_data.get("parameters", {}))
                
                tool_call = ToolCallRequest(
                    id="call_" + str(hash(text))[-8:],
                    name=extracted_data["name"],
                    arguments=args if isinstance(args, dict) else {}
                )
                return replace(response, tool_calls=[tool_call], text=None)
                
        return response

async def run_auto_swe(task: str):
    # Check for Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nError: GEMINI_API_KEY environment variable is missing.")
        print("Get a free key here: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    # Use Gemini's free OpenAI compatibility endpoint!
    provider = OpenAIProvider(
        model="gemini-2.0-flash", 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key
    )
    
    agent = Agent(
        provider=provider,
        tools=get_tools(),
        system_prompt=SYSTEM_PROMPT,
        store="sqlite:///auto_swe.db", # Durable state tracking
    )

    print(f"\n[Task]: {task}\n")
    print("--- Execution Log ---")
    
    try:
        result = await agent.run(task)
        print("\n--- Final State ---")
        print(f"Status: {result.status.name}")
        if result.output:
            print(f"\nFinal Output:\n{result.output}")
            
    except KeyboardInterrupt:
        print("\n\n[Stopped] Run was interrupted. You can resume later.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-SWE: Autonomous Software Engineer")
    parser.add_argument("task", type=str, help="The task for the agent to perform (e.g., 'Fix the bug in main.py')")
    args = parser.parse_args()
    
    asyncio.run(run_auto_swe(args.task))
