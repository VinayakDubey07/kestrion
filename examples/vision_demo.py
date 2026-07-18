import asyncio
import base64
import os
from kestrion.agent.agent import Agent
from kestrion.llm.base import TextBlock, ImageBlock
from kestrion.llm.ollama_provider import OllamaProvider

# Optional: To use Anthropic or OpenAI, uncomment these:
# from kestrion.llm.anthropic_provider import AnthropicProvider
# from kestrion.llm.openai_provider import OpenAIProvider

# A tiny 1x1 red PNG image base64 encoded for demonstration
DUMMY_RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

async def main():
    print("Kestrion Vision / Multimodal Demo")
    print("---------------------------------")
    
    # 1. Prepare image data
    # We will look for 'examples/sample.png' in the workspace.
    # If it doesn't exist, we fall back to the dummy red PNG.
    image_path = "examples/sample.png"
    if os.path.exists(image_path):
        print(f"Loading image from: {image_path}")
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
        media_type = "image/png"
    else:
        print("No 'examples/sample.png' found, using a dummy 1x1 red PNG.")
        base64_data = DUMMY_RED_PNG_B64
        media_type = "image/png"

    # 2. Setup the LLM Provider
    # Ollama requires a multimodal model like llama3.2-vision or llava.
    # To run this, make sure ollama is running and has downloaded the model:
    #   ollama pull llama3.2-vision
    model_name = "llama3.2-vision"
    print(f"Using OllamaProvider with model: {model_name}")
    provider = OllamaProvider(model=model_name)
    
    # Alternatively:
    # provider = AnthropicProvider(model="claude-3-5-sonnet-latest")
    # provider = OpenAIProvider(model="gpt-4o")

    # 3. Initialize Agent
    agent = Agent(
        provider=provider,
        store="sqlite:///vision_demo.db"
    )

    # 4. Construct Content Blocks
    prompt = [
        TextBlock(text="What color is this image, and what is its structure? Please be concise."),
        ImageBlock(data=base64_data, media_type=media_type)
    ]

    print("\nSending multimodal prompt to agent...")
    try:
        result = await agent.run(prompt)
        print(f"\nStatus: {result.status}")
        print(f"Agent response: {result.output}")
    except Exception as e:
        print(f"\n[Error running agent: {e}]")
        print("\nMake sure your local Ollama server is running with the visual model:")
        print("  ollama run llama3.2-vision")

if __name__ == "__main__":
    asyncio.run(main())
