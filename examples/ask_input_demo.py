import asyncio

from kestrion.agent.agent import Agent
from kestrion.agent.tools import ask_human
from kestrion.llm.ollama_provider import OllamaProvider
from kestrion.core.types import RunStatus

async def main():
    print("Initializing Agent with 'ask_human' tool using Ollama...")
    
    # Configure the provider to point to your local Ollama instance
    provider = OllamaProvider(model="llama3.2")
    
    # Initialize the Kestrion Agent
    agent = Agent(
        provider=provider,
        system_prompt=(
            "You are a helpful assistant. If you need any specific personal information "
            "(like the user's favorite color, age, or location) to fulfill a request, "
            "you MUST use the ask_human tool to ask for it. Do not guess."
        ),
        tools=[ask_human],
        store="sqlite:///ask_input_runs.db"
    )

    print("\nSystem: Tell the agent to write a poem about your favorite color, without telling it what the color is.")
    
    run_id = None
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ("exit", "quit"):
                break
            
            print("\nAgent is thinking...", end="\r")
            
            if run_id is None:
                result = await agent.run(user_input)
                run_id = result.run_id
            else:
                result = await agent.run(user_input, run_id=run_id)
                
            # Keep looping while it's waiting for input
            while result.status == RunStatus.WAITING_ON_HUMAN:
                # Get the pending input
                checkpoint = await agent._engine.store.latest(run_id)
                pending_input = checkpoint.state.scratch.get("_pending_input", {})
                tool = pending_input.get("tool")
                question = pending_input.get("question", "Agent needs input.")
                
                if tool == "ask_human":
                    print(f"\n[Agent used ask_human]: {question}")
                    human_ans = input("Your answer: ")
                    
                    print("\nAgent is resuming...", end="\r")
                    result = await agent.provide_input(run_id, human_ans, tool=tool)
                else:
                    # In case it asks for approval
                    print(f"\n[Agent paused for tool {tool}]")
                    break

            # Print over the "is typing..." message
            print(f"Agent: {result.output}")
            
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"\n[Error: {e}]")
            break

if __name__ == "__main__":
    asyncio.run(main())
