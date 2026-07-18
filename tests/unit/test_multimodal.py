import pytest
from typing import Any

from kestrion.llm.base import Message, TextBlock, ImageBlock
from kestrion.llm.anthropic_provider import AnthropicProvider
from kestrion.llm.openai_provider import OpenAIProvider
from kestrion.llm.ollama_provider import OllamaProvider


def test_anthropic_multimodal_mapping():
    provider = AnthropicProvider(api_key="test")
    messages = [
        Message(role="user", content=[
            TextBlock(text="What is in this image?"),
            ImageBlock(data="base64data", media_type="image/png"),
        ])
    ]
    
    mapped = provider._to_anthropic_messages(messages)
    assert len(mapped) == 1
    assert mapped[0]["role"] == "user"
    content = mapped[0]["content"]
    assert len(content) == 2
    
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "What is in this image?"
    
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"
    assert content[1]["source"]["data"] == "base64data"


def test_openai_multimodal_mapping():
    provider = OpenAIProvider(api_key="test")
    messages = [
        Message(role="user", content=[
            TextBlock(text="What is in this image?"),
            ImageBlock(data="base64data", media_type="image/png"),
        ])
    ]
    
    # system is None
    mapped = provider._to_openai_messages(messages, None)
    assert len(mapped) == 1
    assert mapped[0]["role"] == "user"
    content = mapped[0]["content"]
    assert len(content) == 2
    
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "What is in this image?"
    
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,base64data"


def test_ollama_multimodal_mapping():
    provider = OllamaProvider()
    messages = [
        Message(role="user", content=[
            TextBlock(text="What is in this image?"),
            ImageBlock(data="base64data", media_type="image/png"),
        ])
    ]
    
    mapped = provider._to_ollama_messages(messages, None)
    assert len(mapped) == 1
    assert mapped[0]["role"] == "user"
    
    # Ollama combines text into content and puts images in `images` array
    assert mapped[0]["content"] == "What is in this image?"
    assert "images" in mapped[0]
    assert len(mapped[0]["images"]) == 1
    assert mapped[0]["images"][0] == "base64data"
