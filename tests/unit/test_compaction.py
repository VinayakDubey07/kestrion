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

    # Let's seed the run with 6 messages (which is > max_history_turns of 4)
    initial_messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "turn 3"},
        {"role": "assistant", "content": "reply 3"},
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
    assert compact_events[0].payload["original_turns"] == 6
    assert compact_events[0].payload["compacted_turns"] == 4  # 1 summary user + 1 understood assistant + 2 keep turns = 4
    assert "The user asked for greetings" in compact_events[0].payload["summary"]

    # Verify scratch message history matches compacted structure
    compacted_history = result.state.scratch["_messages"]
    assert len(compacted_history) == 5  # 1 summary user + 1 understood assistant + 2 keep turns + 1 final hello reply = 5
    assert "[System Context: Summary of preceding conversation:" in compacted_history[0]["content"]
    assert compacted_history[1]["content"] == "Understood. I will continue the conversation using this context."


async def test_compaction_strict_alternating_roles(temp_db_url):
    provider = FakeCompactionProvider()
    agent = Agent(
        provider=provider,
        store=temp_db_url,
        max_history_turns=4,
        keep_turns=2
    )

    # Let's seed the run with history that forces keep_turns first kept message
    # to be of role 'assistant' (Reply 2 is role assistant, which will be the first kept turn)
    initial_messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "reply 2"},  # keep starts here (assistant)
        {"role": "user", "content": "turn 3"},       # and here (user)
    ]

    result = await agent.run_with_history(initial_messages)
    assert result.status.value == "completed"

    # Verify message alternation in the final compacted messages
    compacted_history = result.state.scratch["_messages"]
    
    # We should have 4 turns now:
    # 1: User summary message (dummy assistant is omitted to keep roles alternating)
    # 2: Assistant "reply 2" (kept turn)
    # 3: User "turn 3" (kept turn)
    # 4: Assistant "Hello there!" (final reply)
    assert len(compacted_history) == 4
    assert "[System Context: Summary of preceding conversation:" in compacted_history[0]["content"]
    assert compacted_history[0]["role"] == "user"
    assert compacted_history[1]["role"] == "assistant"
    assert compacted_history[1]["content"] == "reply 2"
    assert compacted_history[2]["role"] == "user"
    assert compacted_history[2]["content"] == "turn 3"
    assert compacted_history[3]["role"] == "assistant"
    assert compacted_history[3]["content"] == "Hello there!"

    # Ensure no consecutive roles
    for i in range(len(compacted_history) - 1):
        assert compacted_history[i]["role"] != compacted_history[i+1]["role"], (
            f"Role violation at index {i}: consecutive {compacted_history[i]['role']} roles!"
        )
