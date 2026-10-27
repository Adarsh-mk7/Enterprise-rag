import logfire
from typing import Optional
from portkey_ai import Portkey, PORTKEY_GATEWAY_URL, createHeaders
from app.config import settings

# --- PORTKEY CONFIGURATION ---
# Sets up automatic fallback between primary and secondary Groq models,
# simple response caching, and automatic retries.
PORTKEY_CONFIG = {
    "strategy": {
        "mode": "fallback"
    },
    "retry": {
        "attempts": 3
    },
    "cache": {
        "mode": "simple"
    },
    "targets": [
        {
            "virtual_key": settings.GROQ_SLUG,
            "override_params": {"model": settings.GROQ_MODEL}
        },
        {
            "virtual_key": settings.GROQ_SLUG_2,
            "override_params": {"model": "llama-3.1-8b-instant"}
        }
    ]
}

# --- NATIVE PORTKEY CLIENT ---
# Used for direct synthesis (e.g. in responder.py) to read response headers like x-portkey-cache-status
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY or "dummy_key",
    config=PORTKEY_CONFIG
)


def extract_cache_status(response) -> str:
    """
    Extracts the x-portkey-cache-status header from a Portkey API response.
    Returns 'HIT' if served from cache, otherwise 'MISS'.
    """
    headers = getattr(response, "_headers", None)
    if headers:
        for key, value in headers.items():
            if key.lower() == "x-portkey-cache-status":
                return str(value).strip().upper()
    return "MISS"


def get_langchain_llm(feature: str = "general", temperature: float = 0.1):
    """
    Returns a LangChain-compatible LLM.
    If PORTKEY_API_KEY is configured, routes through the Portkey Gateway with fallback and cache.
    Otherwise, gracefully falls back directly to ChatGroq.
    """
    if settings.PORTKEY_API_KEY:
        from langchain_openai import ChatOpenAI

        headers = createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=PORTKEY_CONFIG,
            metadata={"feature": feature}
        )
        return ChatOpenAI(
            api_key="dummy_key",  # Portkey uses x-portkey-api-key header for auth
            base_url=PORTKEY_GATEWAY_URL,
            default_headers=headers,
            temperature=temperature
        )
    else:
        logfire.warning("PORTKEY_API_KEY not set. Falling back directly to ChatGroq.")
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=temperature
        )
