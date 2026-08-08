import re
from typing import Any

from kestrion.core.types import ToolSpec
from kestrion.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    TextBlock,
    ImageBlock,
    ContentBlock,
    ToolCallRequest,
)

# Default patterns for demonstration. In a real system, these might be more robust
# or use a dedicated library like presidio.
DEFAULT_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b",
}


class PIIRedactionMiddleware(LLMProvider):
    """
    An LLMProvider wrapper that intercepts messages before they are sent to the
    underlying provider and scrubs them of sensitive PII (Data Loss Prevention).

    It maintains a two-way vault:
    1. Scans outgoing messages for PII and replaces them with `[REDACTED_TYPE_ID]`.
    2. Scans incoming tool call arguments from the LLM and restores the original
       values before passing them to the agent's tools.
    """

    def __init__(self, provider: LLMProvider, patterns: dict[str, str] | None = None):
        self.provider = provider
        self.patterns = patterns or DEFAULT_PATTERNS
        # Compile patterns for efficiency
        self._compiled_patterns = {
            name: re.compile(pattern) for name, pattern in self.patterns.items()
        }
        
        # The Vault: Maps `[REDACTED_TYPE_ID]` -> `Original Value`
        self._vault: dict[str, str] = {}
        # Counter for unique IDs per type
        self._counters: dict[str, int] = {name: 0 for name in self.patterns}
        # Reverse lookup for deduplication: `Original Value` -> `[REDACTED_TYPE_ID]`
        self._reverse_vault: dict[str, str] = {}

    def _redact_text(self, text: str) -> str:
        if not text:
            return text

        redacted_text = text
        for pii_type, pattern in self._compiled_patterns.items():
            # Find all non-overlapping matches
            matches = pattern.finditer(redacted_text)
            # Replace from the end to avoid offset shifting
            for match in reversed(list(matches)):
                original_value = match.group(0)
                
                # Check if we already redacted this exact value
                if original_value in self._reverse_vault:
                    token = self._reverse_vault[original_value]
                else:
                    token_id = self._counters[pii_type]
                    self._counters[pii_type] += 1
                    token = f"[REDACTED_{pii_type}_{token_id}]"
                    
                    self._vault[token] = original_value
                    self._reverse_vault[original_value] = token

                # Replace in string
                start, end = match.span()
                redacted_text = redacted_text[:start] + token + redacted_text[end:]

        return redacted_text

    def _restore_text(self, text: str) -> str:
        if not text:
            return text
            
        restored_text = text
        # Simple replacement for all known tokens
        for token, original_value in self._vault.items():
            if token in restored_text:
                restored_text = restored_text.replace(token, original_value)
        return restored_text

    def _restore_value(self, value: Any) -> Any:
        """Recursively restore PII in nested JSON structures."""
        if isinstance(value, str):
            return self._restore_text(value)
        elif isinstance(value, list):
            return [self._restore_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._restore_value(v) for k, v in value.items()}
        return value

    def _redact_message(self, message: Message) -> Message:
        """Redact a single message."""
        new_content = message.content
        if isinstance(new_content, str):
            new_content = self._redact_text(new_content)
        elif isinstance(new_content, list):
            new_blocks: list[ContentBlock] = []
            for block in message.content:  # type: ignore
                if isinstance(block, TextBlock):
                    new_blocks.append(TextBlock(text=self._redact_text(block.text)))
                else:
                    new_blocks.append(block)  # type: ignore[arg-type]
            new_content = new_blocks

        # We also need to redact tool results if they are strings
        # Tool calls shouldn't contain new PII from the LLM, but just in case
        redacted_tool_calls = []
        for call in message.tool_calls:
            redacted_args = self._redact_value(call.arguments)
            redacted_tool_calls.append(
                ToolCallRequest(id=call.id, name=call.name, arguments=redacted_args)
            )

        return Message(
            role=message.role,
            content=new_content,
            tool_call_id=message.tool_call_id,
            tool_calls=redacted_tool_calls,
        )

    def _redact_value(self, value: Any) -> Any:
        """Recursively redact PII in nested JSON structures (e.g. tool arguments)."""
        if isinstance(value, str):
            return self._redact_text(value)
        elif isinstance(value, list):
            return [self._redact_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._redact_value(v) for k, v in value.items()}
        return value

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
        output_schema: Any = None,
    ) -> LLMResponse:
        
        # 1. Redact incoming messages and system prompt
        redacted_system = self._redact_text(system) if system else None
        redacted_messages = [self._redact_message(m) for m in messages]

        # 2. Call the underlying provider
        response = await self.provider.complete(
            messages=redacted_messages,
            tools=tools,
            system=redacted_system,
            output_schema=output_schema,
        )

        # 3. Restore PII in the response before returning to the agent
        restored_text = self._restore_text(response.text) if response.text else None
        
        restored_tool_calls = []
        for call in response.tool_calls:
            restored_args = self._restore_value(call.arguments)
            restored_tool_calls.append(
                ToolCallRequest(id=call.id, name=call.name, arguments=restored_args)
            )

        return LLMResponse(
            text=restored_text,
            tool_calls=restored_tool_calls,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost_usd=response.cost_usd,
            stop_reason=response.stop_reason,
        )
