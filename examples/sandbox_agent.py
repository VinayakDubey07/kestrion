import asyncio
from kestrion.agent.agent import Agent
from kestrion.llm.ollama_provider import OllamaProvider
from kestrion.tools.sandbox import CodeSandboxToolkit

async def main():
    # Initialize a local subprocess sandbox workspace
    with CodeSandboxToolkit(mode="subprocess") as sandbox:
        agent = Agent(
            provider=OllamaProvider(model="llama3.2"),
            tools=sandbox.get_tools(),
            store="sqlite:///sandbox_runs.db",
            system_prompt=(
                "You are a Python data analysis agent. You have access to a secure code execution sandbox. "
                "Always write clean code and execute it using execute_python to verify your calculations or process data."
            )
        )
        
        print("Agent is thinking and running code in sandbox...")
        result = await agent.run("Calculate the sum of primes below 10,000 using Python.")
        
        print("\n--- Final Output ---")
        print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
