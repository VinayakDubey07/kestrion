import asyncio
from kestrion.agent.agent import Agent
from kestrion.llm.ollama_provider import OllamaProvider
from kestrion.tools.browser import BrowserToolkit

async def main():
    browser_toolkit = BrowserToolkit(headless=True)
    
    # We use Ollama here as it's locally available, but Anthropic/OpenAI work too
    agent = Agent(
        provider=OllamaProvider(model="llama3.2"),
        tools=browser_toolkit.get_tools(),
        store="sqlite:///agent_runs.db",
        system_prompt="You are a browser automation agent. Use your tools to navigate websites and extract information. If you don't know the answer, use navigate() and extract_text()."
    )
    
    print("Agent is thinking...")
    result = await agent.run("Go to https://example.com and tell me what the main header says.")
    
    print("\n--- Final Output ---")
    print(result.output)
    
    await browser_toolkit.close()

if __name__ == "__main__":
    asyncio.run(main())
