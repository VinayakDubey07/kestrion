# LLM Providers

`kestrion.llm.base.LLMProvider` is the Protocol every provider satisfies — `Agent` only ever calls
`provider.complete(...)`, never anything provider-specific. See the README's Known Gaps for which
providers are live-verified versus implemented against documentation only.

::: kestrion.llm.base

::: kestrion.llm.anthropic_provider

::: kestrion.llm.openai_provider

::: kestrion.llm.ollama_provider