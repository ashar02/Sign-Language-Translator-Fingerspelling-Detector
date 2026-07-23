"""
Text Fix Module

Corrects and formats text from ASL fingerspelling recognition.
Default provider: Groq (free tier). Optional: OpenAI. Fallback: basic formatting.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, "..", "..", ".env"))
load_dotenv()

CORRECTION_PROVIDER = os.getenv("CORRECTION_PROVIDER", "groq").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

API_AVAILABLE = False
client = None
ACTIVE_MODEL = ""
ACTIVE_PROVIDER = "none"


def _is_placeholder(value: str) -> bool:
    if not value:
        return True
    lowered = value.lower()
    return lowered.startswith("your_") or lowered in {"changeme", "none", "null"}


def _init_client() -> None:
    global API_AVAILABLE, client, ACTIVE_MODEL, ACTIVE_PROVIDER, CORRECTION_PROVIDER

    provider = CORRECTION_PROVIDER
    if provider not in {"groq", "openai", "none"}:
        logger.warning(f"Unknown CORRECTION_PROVIDER={provider!r}; using groq")
        provider = "groq"
        CORRECTION_PROVIDER = provider

    if provider == "none":
        logger.info("Text correction provider disabled (CORRECTION_PROVIDER=none)")
        return

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Install with: pip install openai")
        return

    if provider == "groq":
        if _is_placeholder(GROQ_API_KEY):
            logger.warning("GROQ_API_KEY not configured. Text correction will use basic formatting.")
            logger.warning("Get a free key at https://console.groq.com/keys")
            return
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        ACTIVE_MODEL = GROQ_MODEL or "llama-3.1-8b-instant"
        ACTIVE_PROVIDER = "groq"
        API_AVAILABLE = True
        logger.info(f"Groq client initialized (model={ACTIVE_MODEL})")
        return

    # openai
    if _is_placeholder(OPENAI_API_KEY):
        logger.warning("OPENAI_API_KEY not configured. Text correction will use basic formatting.")
        return
    client = OpenAI(api_key=OPENAI_API_KEY)
    ACTIVE_MODEL = OPENAI_MODEL or "gpt-4o-mini"
    ACTIVE_PROVIDER = "openai"
    API_AVAILABLE = True
    logger.info(f"OpenAI client initialized (model={ACTIVE_MODEL})")


_init_client()

# System prompt for text correction
SYSTEM_PROMPT = """CRITICAL RULE: Return ONLY the corrected sentence. No explanations, no input/output labels.
1. Return ONLY the corrected sentence - no explanations, labels, or quotes
2. NEVER change original words or grammar structures
3. NEVER add, remove, or modify words
4. ONLY fix spelling errors and join spaced letters
5. Add punctuation ONLY when clearly needed
6. Maintain ALL original word forms exactly as given
7. Keep exact same sentence structure
8. Preserve word order exactly as input
9. Keep formal/informal tone as provided
10. No semantic or meaning changes
11. Keep known acronyms exactly as they are (BMW stays BMW)
12. Add period after single words and acronyms
13. NEVER convert acronyms to words
14. Fix common typing errors and misspellings
15. Add appropriate punctuation
16. Convert repeated letters only if they're typos (like 'helllo' to 'hello')
19. Maintain proper sentence case
20. Handle contractions properly (dont -> don't, cant -> can't)

Examples:
Input: D O N O T C R Y N O W
Output: Do Not Cry Now.

Input: H E L L O W O R L D
Output: Hello World.

Input: W H E R E A R E Y O U
Output: Where Are You?

Input: B M W
Output: BMW.

Input: I W O R K A T I B M
Output: I Work At IBM.
"""


def _basic_format(input_text: str) -> str:
    """Fallback when no LLM provider is available."""
    formatted = " ".join(input_text.split()).strip()
    if not formatted:
        return ""
    # Join spaced single letters: "H E L L O" -> "HELLO"
    parts = formatted.split(" ")
    if parts and all(len(p) == 1 for p in parts):
        formatted = "".join(parts)
    if len(formatted) > 1:
        formatted = formatted[0].upper() + formatted[1:]
    else:
        formatted = formatted.upper()
    if not formatted.endswith((".", "!", "?")):
        formatted += "."
    return formatted


def generate_sentences(input_text: str) -> str:
    """Generate corrected sentences from spaced letter input.

    Args:
        input_text: Raw text with spaced letters from ASL recognition.

    Returns:
        Corrected and formatted sentence.
    """
    if not input_text or not input_text.strip():
        return ""

    if not API_AVAILABLE or client is None:
        logger.warning("Correction API not available, returning basic formatting")
        return _basic_format(input_text)

    try:
        logger.info(f"Processing text via {ACTIVE_PROVIDER}/{ACTIVE_MODEL}: {input_text[:50]}...")

        response = client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0.0,
            max_tokens=max(64, len(input_text.split()) + 50),
        )

        result = response.choices[0].message.content if response.choices else None
        cleaned = (result or "").strip().strip('"').strip("'")
        logger.info(f"Generated result: {cleaned}")
        return cleaned if cleaned else _basic_format(input_text)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"{ACTIVE_PROVIDER} API error: {error_msg}")

        if "401" in error_msg or "invalid_api_key" in error_msg.lower():
            logger.error(f"Invalid {ACTIVE_PROVIDER} API key. Check your .env file.")
        elif "429" in error_msg or "rate_limit" in error_msg.lower():
            logger.error(f"{ACTIVE_PROVIDER} rate limit exceeded. Try again later.")

        return _basic_format(input_text)


if __name__ == "__main__":
    for sample in ["H E L L O W O R L D", "I L O V E C O D I N G", "B M W"]:
        print(f"Input:  {sample}")
        print(f"Output: {generate_sentences(sample)}")
        print("-" * 40)
