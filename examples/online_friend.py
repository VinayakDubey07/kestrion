import asyncio

from kestrion.agent.agent import Agent
from kestrion.llm.ollama_provider import OllamaProvider

async def main():
    print("Initializing Online Friend Agent using Ollama...")
    
    # Configure the provider to point to your local Ollama instance
    # The default is llama3.2, but you can change it here.
    provider = OllamaProvider(model="llama3.2")
    
    # Initialize the Kestrion Agent
    agent = Agent(
        provider=provider,
        system_prompt=(
            "You are a warm, casual, and supportive online friend. "
            "You use emojis occasionally, keep responses relatively short "
            "and conversational, and ask questions to keep the chat going. "
            "Do not sound robotic or like a typical AI assistant."
        ),
        tools=[],  # A simple chat friend might not need tools
        store="sqlite:///kestrion_runs.db"
    )

    print("\nFriend: Hey there! It's so good to talk to you. How's your day going?")
    
    # Keep track of the run ID so we maintain a conversation history
    run_id = None
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ("exit", "quit"):
                print("Friend: Bye! Talk to you later!")
                break
            
            # The agent.run() method returns a RunResult which contains the output
            # By passing the run_id back on subsequent calls, we maintain context!
            print("\nFriend is typing...", end="\r")
            
            result = await agent.run(user_input, run_id=run_id)
            run_id = result.run_id
            
            # Print over the "is typing..." message
            print("Friend:", result.output)
            
        except KeyboardInterrupt:
            print("\nFriend: Bye! Talk to you later!")
            break
        except Exception as e:
            print(f"\n[Error communicating with Ollama: {e}]")
            print("Make sure Ollama is running and the model (llama3.2) is pulled.")
            break

if __name__ == "__main__":
    asyncio.run(main())
