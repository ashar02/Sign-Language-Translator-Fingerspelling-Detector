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

# System prompt for text correction (ASL fingerspelling → readable text)
SYSTEM_PROMPT = """You reconstruct American Sign Language fingerspelling into readable English.

The input is letters detected from hand signs. Letters are often spaced (H E L L O) and may include mistakes: missing letters, extra letters, wrong letters, or repeated letters.

YOUR JOB:
1. Return ONLY the final corrected sentence. No labels, quotes, or explanation.
2. Join spaced letters into words (H E L L O → Hello).
3. Split into words when the letter stream clearly contains multiple words (H E L L O W O R L D → Hello World).
4. GUESS the closest real English word/phrase when letters are noisy or incomplete. Prefer common everyday words.
5. Fix typos and near-miss spellings aggressively when intent is clear (H E L O → Hello, T H A M K → Thank, A H I → Ahi only if that fits; prefer real words like "Hi" / "Ah" contextually when short).
6. Keep known acronyms as acronyms (B M W → BMW, N A S A → NASA).
7. Use normal sentence case and end with . ! or ? when appropriate.
8. Do NOT invent long unrelated sentences. Stay close to the signed letters.
9. Do NOT output the input with spaces still between every letter.
10. If only a few letters are given, still form the best short word/greeting you can (H I → Hi., O K → OK., Y E S → Yes.).

Examples:
Input: H E L L O
Output: Hello.

Input: H E L O
Output: Hello.

Input: H I
Output: Hi.

Input: T H A N K Y O U
Output: Thank You.

Input: T H A M K Y O U
Output: Thank You.

Input: G O O D M O R N I N G
Output: Good Morning.

Input: W H E R E A R E Y O U
Output: Where Are You?

Input: B M W
Output: BMW.

Input: I W O R K A T I B M
Output: I Work At IBM.

Input: H E L L L O W O R L D D
Output: Hello World.

Input: P L E A S N E H E L P
Output: Please Help.
"""


def _compact_original(input_text: str) -> str:
    """Join spaced fingerspelled letters into a compact original token string."""
    parts = [p for p in input_text.strip().split() if p]
    if not parts:
        return ""
    if all(len(p) == 1 for p in parts):
        return "".join(parts).upper()
    return " ".join(parts)


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


def _with_original(input_text: str, guessed: str) -> str:
    """Show compact original letters and guessed text: ORIGINAL / Guessed."""
    original = _compact_original(input_text)
    guessed = (guessed or "").strip()
    if not guessed:
        guessed = _basic_format(input_text)
    if not original:
        return guessed
    # Avoid "HELLO / Hello." duplication noise when identical ignoring punctuation/case
    guess_cmp = guessed.rstrip(".!?").replace(" ", "").upper()
    orig_cmp = original.replace(" ", "").upper()
    if guess_cmp == orig_cmp:
        return guessed
    return f"{original} / {guessed}"


def generate_sentences(input_text: str) -> str:
    """Generate corrected sentences from spaced letter input.

    Returns:
        String like \"HELO / Hello.\" (original letters / guessed sentence).
    """
    if not input_text or not input_text.strip():
        return ""

    if not API_AVAILABLE or client is None:
        logger.warning("Correction API not available, returning basic formatting")
        return _with_original(input_text, _basic_format(input_text))

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
        # If model already returned "A / B", keep only the guess part after last " / "
        if " / " in cleaned:
            cleaned = cleaned.split(" / ")[-1].strip()
        logger.info(f"Generated result: {cleaned}")
        guessed = cleaned if cleaned else _basic_format(input_text)
        return _with_original(input_text, guessed)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"{ACTIVE_PROVIDER} API error: {error_msg}")

        if "401" in error_msg or "invalid_api_key" in error_msg.lower():
            logger.error(f"Invalid {ACTIVE_PROVIDER} API key. Check your .env file.")
        elif "429" in error_msg or "rate_limit" in error_msg.lower():
            logger.error(f"{ACTIVE_PROVIDER} rate limit exceeded. Try again later.")

        return _with_original(input_text, _basic_format(input_text))


if __name__ == "__main__":
    for sample in ["H E L L O W O R L D", "I L O V E C O D I N G", "B M W"]:
        print(f"Input:  {sample}")
        print(f"Output: {generate_sentences(sample)}")
        print("-" * 40)
