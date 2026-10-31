"""Chat-model provider abstraction.

The agent only ever calls :func:`build_chat_model`; to add another provider
(OpenAI-compatible, Ollama, ...) register a factory and set ``LLM_PROVIDER``.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from config import Settings
from exceptions import ConfigurationError, ServiceNotConfiguredError

ProviderFactory = Callable[[Settings, str, float], BaseChatModel]
_PROVIDERS: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    _PROVIDERS[name.lower()] = factory


def _groq_factory(settings: Settings, model: str, temperature: float) -> BaseChatModel:
    if settings.groq_api_key is None:
        raise ServiceNotConfiguredError(
            "The language model is not configured. Set GROQ_API_KEY on the server (see .env.example)."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        streaming=True,
    )


register_provider("groq", _groq_factory)


def build_chat_model(settings: Settings, model: str | None = None, temperature: float | None = None) -> BaseChatModel:
    factory = _PROVIDERS.get(settings.llm_provider.lower())
    if factory is None:
        raise ConfigurationError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Available: {', '.join(sorted(_PROVIDERS))}."
        )
    return factory(settings, model or settings.groq_model,
                   settings.default_temperature if temperature is None else temperature)
