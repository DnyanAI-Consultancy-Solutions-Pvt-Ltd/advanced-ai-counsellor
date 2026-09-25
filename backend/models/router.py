import os
import itertools
from langchain_groq import ChatGroq

raw_keys = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY", "")
GROQ_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]
_key_cycle = itertools.cycle(GROQ_KEYS) if GROQ_KEYS else None


def get_next_key() -> str:
    return next(_key_cycle) if _key_cycle else os.getenv("GROQ_API_KEY", "")


def get_llm(provider: str = "groq"):
    key = get_next_key()

    if provider == "groq_fast":
        return ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.0,
            max_retries=2,
            groq_api_key=key,
        )

    primary = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_retries=1,
        groq_api_key=key,
    )
    fallback = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        max_retries=2,
        groq_api_key=key,
    )

    return primary.with_fallbacks([fallback])
