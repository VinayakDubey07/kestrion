"""
Example of using the VercelStreamAdapter to power a Generative UI backend.
This FastAPI server provides an endpoint that streams events matching the Vercel AI SDK Data Stream Protocol.

To run:
    uv pip install fastapi uvicorn
    uvicorn examples.vercel_generative_ui:app --reload

To test (in another terminal):
    curl -N -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"prompt": "What is the weather in New York?"}'
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from kestrion.agent.agent import Agent
from kestrion.llm.ollama_provider import OllamaProvider
from kestrion.agent.decorators import tool
from kestrion.adapters.vercel import stream_to_vercel

app = FastAPI(title="Kestrion Generative UI Backend")

# Define a tool that a Next.js frontend will render as a React component
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # In a real app, you'd fetch from an API.
    # The frontend receives this tool call and renders a <WeatherCard />!
    return f"The weather in {location} is 72 degrees and sunny."

class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    # Initialize the Kestrion Agent
    agent = Agent(
        provider=OllamaProvider(model="llama3.2"),
        tools=[get_weather],
        store="sqlite:///agent_runs.db",
        system_prompt="You are a helpful assistant. If the user asks for the weather, use the get_weather tool."
    )
    
    # Use the Vercel adapter to stream the agent's run
    return StreamingResponse(
        stream_to_vercel(agent, req.prompt),
        media_type="text/plain"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
