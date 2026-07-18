import pytest
from kestrion.agent.agent import Agent
from kestrion.llm.base import LLMResponse, ToolCallRequest

class FakeSummarizingProvider:
    def __init__(self):
        self.call_count = 0

    async def complete(self, messages, tools, system=None):
        self.call_count += 1
        
        # If the last message asks to summarize, return a summary
        if "Summarize the preceding conversation" in messages[-1].content:
            return LLMResponse(text="This is a summary of the past.", tool_calls=[], stop_reason="stop")
            
        # Otherwise, just return a normal response
        return LLMResponse(
            text=f"Response {self.call_count}",
            tool_calls=[],
            stop_reason="stop"
        )

@pytest.mark.asyncio
async def test_compaction_truncates_history(tmp_store):
    store_url = tmp_store
    provider = FakeSummarizingProvider()
    
    # max_history_turns = 2, so the 3rd turn triggers compaction
    agent = Agent(provider=provider, tools=[], store=store_url, max_history_turns=2, keep_turns=1)
    # Simulate a long history being handed to the agent
    # 4 messages = 2 turns
    messages = [
        {"role": "user", "content": "Hello 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Hello 2"},
        {"role": "assistant", "content": "Response 2"},
        {"role": "user", "content": "Hello 3"}, # 5 messages > max_history_turns (2) * 2
    ]
    
    # This single run will trigger compaction immediately because len(messages) = 5 > 2
    res = await agent.run_with_history(messages, run_id="run_compact_1")
    state = (await agent._engine.store.latest("run_compact_1")).state
    
    # History should be truncated and prepended with a summary
    # 1 item for the summary, + 1 item for the new llm_response = 2 items total!
    assert len(state.history) == 2
    assert state.history[0]["type"] == "summary"
    assert state.history[0]["content"] == "This is a summary of the past."
    assert state.history[1]["type"] == "llm_response"
    assert "stop_reason" in state.history[1]
