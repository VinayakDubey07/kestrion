import pytest
import tempfile
from pathlib import Path
from kestrion.agent.agent import Agent
from kestrion.llm.base import LLMResponse
from kestrion.core.types import EventType

class FakeCompactionProvider:
    def __init__(self):
        self.complete_calls = 0
        self.summarize_calls = 0

    async def complete(self, messages, tools=None, system=None):
        self.complete_calls += 1
        
        # Check if this is a request to summarize (contains "Summarize the preceding conversation")
        if messages and "Summarize the preceding conversation" in messages[-1].content:
            self.summarize_calls += 1
            return LLMResponse(
                text="The user asked for greetings, and the assistant responded hello.",
                tool_calls=[],
                tokens_in=10, tokens_out=5, cost_usd=0.0001,
                stop_reason="end_turn"
            )
            
        # Return a regular response
        return LLMResponse(
            text="Hello there!",
            tool_calls=[],
            tokens_in=20, tokens_out=10, cost_usd=0.0002,
            stop_reason="end_turn"
        )

@pytest.fixture
def temp_db_url():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield f"sqlite:///{Path(tmpdir) / 'test.db'}"

async def test_memory_compaction_triggered_by_turns(temp_db_url):
    provider = FakeCompactionProvider()
    agent = Agent(
        provider=provider,
        store=temp_db_url,
        max_history_turns=4,
        keep_turns=2
    )

    # Let's seed the run with 5 messages (which is > max_history_turns of 4)
    initial_messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "turn 3"},
    ]

    result = await agent.run_with_history(initial_messages)
    
    assert result.status.value == "completed"
    # Verify summary call was made
    assert provider.summarize_calls == 1
    
    # Replay events to check if EventType.CONTEXT_COMPACTED is in the event log
    store = agent._store
    events = await store.events_since(result.run_id, 0)
    compact_events = [e for e in events if e.type == EventType.CONTEXT_COMPACTED]
    assert len(compact_events) == 1
    assert compact_events[0].payload["original_turns"] == 5
    assert compact_events[0].payload["compacted_turns"] == 4  # 1 summary user + 1 understood assistant + 2 keep turns = 4
    assert "The user asked for greetings" in compact_events[0].payload["summary"]

    # Verify scratch message history matches compacted structure
    compacted_history = result.state.scratch["_messages"]
    assert len(compacted_history) == 5  # 1 summary user + 1 understood assistant + 2 keep turns + 1 final hello reply = 5
    assert "[System Context: Summary of preceding conversation:" in compacted_history[0]["content"]
    assert compacted_history[1]["content"] == "Understood. I will continue the conversation using this context."
