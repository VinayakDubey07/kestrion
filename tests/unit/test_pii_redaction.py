import pytest

from kestrion.core.types import ToolSpec
from kestrion.llm.base import LLMResponse, Message, ToolCallRequest, LLMProvider
from kestrion.llm.middleware import PIIRedactionMiddleware


class MockProvider(LLMProvider):
    def __init__(self, expected_response: LLMResponse):
        self.expected_response = expected_response
        self.received_messages: list[Message] = []

    async def complete(self, messages: list[Message], tools: list[ToolSpec], system: str | None = None, output_schema: dict | None = None) -> LLMResponse:
        self.received_messages = messages
        return self.expected_response


@pytest.mark.asyncio
async def test_pii_redaction_and_restoration():
    # Setup the mock provider that returns a tool call with the REDACTED token
    # This simulates an LLM receiving [REDACTED_SSN_0] and deciding to call a tool with it.
    mock_response = LLMResponse(
        text="I am submitting the application for [REDACTED_SSN_0].",
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="submit_background_check",
                arguments={"ssn": "[REDACTED_SSN_0]", "email": "[REDACTED_EMAIL_0]"},
            )
        ],
    )
    mock_provider = MockProvider(mock_response)
    
    # Wrap it with our middleware
    middleware = PIIRedactionMiddleware(provider=mock_provider)

    # 1. The original message containing raw PII
    original_messages = [
        Message(
            role="user", 
            content="My SSN is 123-45-6789 and my email is test@example.com."
        )
    ]
    
    tools = [
        ToolSpec(
            name="submit_background_check",
            description="Submit a background check",
            parameters={"type": "object", "properties": {"ssn": {"type": "string"}, "email": {"type": "string"}}},
        )
    ]

    # 2. Call the middleware
    response = await middleware.complete(messages=original_messages, tools=tools)

    # 3. VERIFY OUTGOING (Redaction): The mock provider should ONLY see redacted values
    received_content = mock_provider.received_messages[0].content
    assert "123-45-6789" not in received_content
    assert "test@example.com" not in received_content
    assert "[REDACTED_SSN_0]" in received_content
    assert "[REDACTED_EMAIL_0]" in received_content
    assert received_content == "My SSN is [REDACTED_SSN_0] and my email is [REDACTED_EMAIL_0]."

    # 4. VERIFY INCOMING (Restoration): The Agent should receive the RESTORED real values
    assert "123-45-6789" in response.text
    assert "test@example.com" not in response.text # text only had SSN in the mock response
    assert response.text == "I am submitting the application for 123-45-6789."

    # And the tool arguments should be restored
    tool_call = response.tool_calls[0]
    assert tool_call.arguments["ssn"] == "123-45-6789"
    assert tool_call.arguments["email"] == "test@example.com"

@pytest.mark.asyncio
async def test_pii_redaction_deduplication():
    # If the same SSN appears twice, it should use the same token
    mock_response = LLMResponse(text="Okay.")
    mock_provider = MockProvider(mock_response)
    middleware = PIIRedactionMiddleware(provider=mock_provider)

    messages = [
        Message(role="user", content="SSN: 123-45-6789. Again: 123-45-6789.")
    ]
    
    await middleware.complete(messages=messages, tools=[])
    
    received_content = mock_provider.received_messages[0].content
    # It should use [REDACTED_SSN_0] twice, not [REDACTED_SSN_1]
    assert received_content == "SSN: [REDACTED_SSN_0]. Again: [REDACTED_SSN_0]."
