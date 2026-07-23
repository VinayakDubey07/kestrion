"""
ToolRegistry allows for dynamic discovery and loading of tools at runtime.
"""
from __future__ import annotations

from typing import Iterable
from kestrion.core.types import Tool


class ToolRegistry:
    """
    A registry of tools that an Agent can search and load dynamically.
    """

    def __init__(self, tools: Iterable[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self.register(t)

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def search(self, query: str) -> list[Tool]:
        """
        Search for tools whose name or description loosely match the query.
        This is a simple substring match for v1.
        """
        query_lower = query.lower()
        results = []
        for tool in self._tools.values():
            if query_lower in tool.spec.name.lower() or query_lower in tool.spec.description.lower():
                results.append(tool)
        return results
