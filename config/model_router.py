"""
Enterprise Dynamic Model Router
Provides intelligent model resolution across:
1. High-tier reasoning nodes (Supervisor, Safety Gate, Refund Resolver) -> Groq Llama 3.3 70B / Gemini Flash
2. Worker & filtering nodes (Diet Specialist, Stock/Formatters) -> Local Ollama (with automatic Cloud fallback)
"""

import logging
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from config.settings import settings

logger = logging.getLogger("model_router")


def get_high_tier_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Resolves the high-tier reasoning LLM for complex orchestrations.
    Prioritizes Groq (Llama 3.3 70B) for ultra-fast structured JSON tool-calling,
    with automatic fallback to Google Gemini Flash.
    """
    # 1. Try Groq if API key is provided
    if settings.GROQ_API_KEY and not settings.GROQ_API_KEY.startswith("gsk_your"):
        try:
            from langchain_groq import ChatGroq
            logger.info("Routing High-Tier task to Groq: %s", settings.GROQ_MODEL)
            return ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL,
                temperature=temperature,
            )
        except Exception as exc:
            logger.warning("Groq initialization failed: %s. Attempting Gemini fallback.", exc)

    # 2. Try Gemini Flash if API key is provided
    if settings.GEMINI_API_KEY and not settings.GEMINI_API_KEY.startswith("your_gemini"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.info("Routing High-Tier task to Gemini: %s", settings.GEMINI_MODEL)
            return ChatGoogleGenerativeAI(
                google_api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_MODEL,
                temperature=temperature,
            )
        except Exception as exc:
            logger.warning("Gemini initialization failed: %s.", exc)

    # 3. Fallback to Local Ollama if no cloud keys configured
    try:
        from langchain_ollama import ChatOllama
        logger.info("Routing to Local Ollama fallback: %s", settings.OLLAMA_WORKER_MODEL)
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_WORKER_MODEL,
            temperature=temperature,
        )
    except Exception as exc:
        raise RuntimeError(
            "No LLM provider available! Please configure GROQ_API_KEY, GEMINI_API_KEY, or run Ollama locally."
        ) from exc


def get_worker_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Resolves lightweight worker models for filtering, formatting, and diet translations.
    Prioritizes Local Ollama (zero-cost compute) with graceful fallback to Groq/Gemini cloud endpoints.
    """
    # 1. Attempt local Ollama connection
    try:
        import httpx
        # Quick health ping to Ollama daemon
        resp = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/version", timeout=1.0)
        if resp.status_code == 200:
            from langchain_ollama import ChatOllama
            logger.info("Routing Worker task to Local Ollama: %s", settings.OLLAMA_WORKER_MODEL)
            return ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_WORKER_MODEL,
                temperature=temperature,
            )
    except Exception:
        logger.info("Local Ollama not reachable; falling back to Cloud High-Tier provider for worker.")

    # 2. Fallback directly to High-Tier cloud provider
    return get_high_tier_llm(temperature=temperature)
