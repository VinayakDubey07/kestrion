"""
Toolkit for agents to interact with long-term episodic memory.
"""

from typing import Any

from kestrion.store.episodic_memory import EpisodicMemoryStore
from kestrion.agent.decorators import tool


class MemoryToolkit:
    """
    A collection of tools that allows an agent to save and search long-term memory facts.
    """
    
    def __init__(self, store: EpisodicMemoryStore, user_id: str):
        self.store = store
        self.user_id = user_id

        @tool(name="save_memory")
        async def save_memory(content: str) -> dict[str, Any]:
            """
            Save a fact, user preference, or piece of information to the user's long-term memory.
            Use this to record important context that needs to be remembered across separate conversation runs.
            """
            fact_id = await self.store.save_fact(self.user_id, content)
            return {"status": "success", "fact_id": fact_id, "content": content}

        @tool(name="search_memory")
        async def search_memory(query: str, limit: int = 5) -> list[dict[str, Any]]:
            """
            Search the user's long-term memory for past facts or preferences matching the query.
            If the query is empty, it returns the most recent memories.
            """
            facts = await self.store.search_facts(self.user_id, query, limit)
            return [f.to_dict() for f in facts]

        self.tools = [save_memory, search_memory]
