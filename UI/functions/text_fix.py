"""
Text Fix Module

Corrects and formats text from ASL fingerspelling recognition.
Default provider: Groq (free tier). Optional: OpenAI. Fallback: basic + local fuzzy guess.
"""

from __future__ import annotations

import logging
import os
import re
from difflib import SequenceMatcher
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, "..", "..", ".env"))
load_dotenv()

CORRECTION_PROVIDER = os.getenv("CORRECTION_PROVIDER", "groq").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# 70B is much better at aggressive typo/word guessing than 8B instant
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
# Higher = more aggressive guesses (0.0–1.0)
CORRECTION_TEMPERATURE = float(os.getenv("CORRECTION_TEMPERATURE", "0.7"))

API_AVAILABLE = False
client = None
ACTIVE_MODEL = ""
ACTIVE_PROVIDER = "none"

# Common ASL / demo phrases for local aggressive guessing (letter-only match keys built at runtime)
_COMMON_PHRASES = [
    "hi",
    "hello",
    "hey",
    "yes",
    "no",
    "ok",
    "okay",
    "please",
    "thanks",
    "thank you",
    "please help",
    "help me",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "goodbye",
    "bye",
    "how are you",
    "how are you doing",
    "i am fine",
    "i love you",
    "i love coding",
    "nice to meet you",
    "what is your name",
    "my name is",
    "where are you",
    "where is",
    "who are you",
    "see you later",
    "see you soon",
    "welcome",
    "sorry",
    "excuse me",
    "i am sorry",
    "can you help",
    "i need help",
    "hello world",
    "good luck",
    "have a nice day",
    "take care",
    "what time is it",
    "i understand",
    "i do not understand",
    "speak slower",
    "repeat please",
    "yes please",
    "no thank you",
    "bmw",
    "nasa",
    "ibm",
    "asl",
]

_COMMON_WORDS = sorted(
    {
        "hi",
        "hello",
        "hey",
        "yes",
        "no",
        "ok",
        "okay",
        "please",
        "help",
        "thanks",
        "thank",
        "you",
        "good",
        "morning",
        "afternoon",
        "evening",
        "night",
        "bye",
        "goodbye",
        "how",
        "are",
        "doing",
        "fine",
        "love",
        "coding",
        "nice",
        "meet",
        "what",
        "name",
        "where",
        "who",
        "see",
        "later",
        "soon",
        "welcome",
        "sorry",
        "excuse",
        "need",
        "world",
        "luck",
        "have",
        "day",
        "take",
        "care",
        "time",
        "understand",
        "speak",
        "slower",
        "repeat",
        "work",
        "at",
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "my",
        "your",
        "our",
        "am",
        "is",
        "was",
        "can",
        "will",
        "want",
        "like",
        "home",
        "school",
        "friend",
        "family",
        "water",
        "food",
        "stop",
        "start",
        "go",
        "come",
        "wait",
        "more",
        "less",
        "big",
        "small",
        "happy",
        "sad",
        "tired",
        "hungry",
        "thirsty",
    },
    key=len,
    reverse=True,
)


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
        ACTIVE_MODEL = GROQ_MODEL or "llama-3.3-70b-versatile"
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

SYSTEM_PROMPT = """You are an aggressive ASL fingerspelling decoder.

Input is noisy hand-detected letters (often spaced like H E L O). Expect wrong letters, missing letters, extra letters, and repeats.

GOAL: Output the MOST LIKELY real English words/phrase a person meant. Be BOLD with corrections.

Rules:
- Return ONLY the corrected phrase. No quotes, labels, or explanation.
- ALWAYS join spaced letters into words. Never leave spaces between every letter.
- Prefer common everyday English over rare/obscure words.
- Correct aggressively when letters are close: HELO→Hello, THAMKYOU→Thank You, PLEASNEHELP→Please Help, GDMORNING→Good Morning.
- Split into multiple words when that yields real English (HELLOWORLD→Hello World).
- Keep known acronyms: BMW, NASA, IBM, ASL.
- Title Case normal phrases. End with . ! or ? when it fits.
- Short inputs still get a real word: HI→Hi., OK→OK., YES→Yes.
- Do NOT just echo the raw letter string with title case if a nearby real word exists.
- Do NOT invent long unrelated paragraphs. Stay in the same meaning neighborhood as the letters.

Examples:
H E L L O → Hello.
H E L O → Hello.
H L L O → Hello.
T H A M K Y O U → Thank You.
P L E A S N E H E L P → Please Help.
G O O D M O R N I N G → Good Morning.
W H E R E A R E Y O U → Where Are You?
H E L L L O W O R L D D → Hello World.
I L O V E Y O U → I Love You.
H O W A R E Y O U → How Are You?
B M W → BMW.
"""

_USER_TEMPLATE = """Noisy fingerspelled letters below. Guess the closest real English phrase AGGRESSIVELY.
Fix wrong/missing/extra letters. Join into words. Output only the corrected phrase.

Letters: {letters}
"""


def _letters_only(text: str) -> str:
    return re.sub(r"[^a-zA-Z]", "", text or "").lower()


def _title_phrase(phrase: str) -> str:
    phrase = " ".join(phrase.split()).strip()
    if not phrase:
        return ""
    # Keep short all-caps acronyms
    if phrase.isupper() and len(phrase) <= 5 and " " not in phrase:
        return phrase
    words = []
    for w in phrase.split(" "):
        if w.isupper() and len(w) <= 5:
            words.append(w)
        else:
            words.append(w[:1].upper() + w[1:].lower() if w else w)
    out = " ".join(words)
    if not out.endswith((".", "!", "?")):
        out += "."
    return out


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
    parts = formatted.split(" ")
    if parts and all(len(p) == 1 for p in parts):
        formatted = "".join(parts)
    return _title_phrase(formatted)


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _local_phrase_guess(compact: str) -> Optional[str]:
    """Fuzzy-match compacted letters against common phrases."""
    if not compact or len(compact) < 2:
        return None

    best_phrase = None
    best_score = 0.0
    for phrase in _COMMON_PHRASES:
        key = _letters_only(phrase)
        if not key:
            continue
        # Length gate: avoid matching tiny phrases to long garbage and vice versa
        if abs(len(key) - len(compact)) > max(3, len(compact) // 2):
            continue
        score = _similarity(compact, key)
        # Bonus if one contains the other (missing letters)
        if key in compact or compact in key:
            score = max(score, 0.82)
        if score > best_score:
            best_score = score
            best_phrase = phrase

    # Aggressive threshold: accept fairly noisy matches
    if best_phrase and best_score >= 0.62:
        return _title_phrase(best_phrase)
    return None


def _segment_words(compact: str) -> Optional[str]:
    """Greedy left-to-right word segmentation with fuzzy single-word matches."""
    if not compact or len(compact) < 2:
        return None

    n = len(compact)
    # dp[i] = (score, words ending at i)
    dp: list[Optional[tuple[float, list[str]]]] = [None] * (n + 1)
    dp[0] = (0.0, [])

    for i in range(n):
        if dp[i] is None:
            continue
        base_score, base_words = dp[i]
        # Try exact and fuzzy word matches of length 1..12
        for length in range(1, min(13, n - i + 1)):
            chunk = compact[i : i + length]
            best_word = None
            best = 0.0
            for word in _COMMON_WORDS:
                if abs(len(word) - length) > 2:
                    continue
                score = _similarity(chunk, word)
                if word.startswith(chunk) or chunk.startswith(word[: max(2, len(word) - 1)]):
                    score = max(score, 0.75 if abs(len(word) - length) <= 1 else score)
                if score > best:
                    best = score
                    best_word = word
            # Require decent match; shorter chunks need higher confidence
            min_score = 0.78 if length <= 2 else 0.68
            if best_word and best >= min_score:
                new_score = base_score + best * length
                cand = (new_score, base_words + [best_word])
                j = i + length
                if dp[j] is None or cand[0] > dp[j][0]:
                    dp[j] = cand

    if dp[n] is None:
        return None
    _, words = dp[n]
    if not words:
        return None
    return _title_phrase(" ".join(words))


def _local_aggressive_guess(input_text: str) -> str:
    """Local dictionary/fuzzy guess used as fallback or boost."""
    compact = _letters_only(_compact_original(input_text) or input_text)
    if not compact:
        return ""

    phrase = _local_phrase_guess(compact)
    segmented = _segment_words(compact)

    candidates = [c for c in (phrase, segmented) if c]
    if not candidates:
        return _basic_format(input_text)

    # Prefer the candidate whose letters are closest to input
    best = max(candidates, key=lambda c: _similarity(compact, _letters_only(c)))
    return best


def _looks_like_raw_echo(input_text: str, guessed: str) -> bool:
    """True when the model mostly echoed joined letters without real-word correction."""
    src = _letters_only(_compact_original(input_text) or input_text)
    dst = _letters_only(guessed)
    if not src or not dst:
        return True
    if src == dst:
        return True
    # Nearly identical (title-cased join of noisy letters)
    return _similarity(src, dst) >= 0.92


def _boost_guess(input_text: str, guessed: str) -> str:
    """If LLM is too conservative, replace with a stronger local guess when better."""
    guessed = (guessed or "").strip()
    local = _local_aggressive_guess(input_text)
    if not guessed:
        return local
    if not local:
        return guessed

    src = _letters_only(_compact_original(input_text) or input_text)
    llm_sim = _similarity(src, _letters_only(guessed))
    local_sim = _similarity(src, _letters_only(local))

    if _looks_like_raw_echo(input_text, guessed) and local_sim >= 0.62:
        logger.info(f"Boosting conservative LLM guess {guessed!r} → {local!r}")
        return local

    # Prefer local when it is clearly a real phrase and still close to letters
    if local_sim >= 0.72 and local_sim >= llm_sim - 0.05 and _looks_like_raw_echo(input_text, guessed):
        return local

    return guessed


def _with_original(input_text: str, guessed: str) -> str:
    """Always show compact original letters and guessed text: ORIGINAL / Guessed."""
    original = _compact_original(input_text)
    guessed = (guessed or "").strip()
    if not guessed:
        guessed = _local_aggressive_guess(input_text) or _basic_format(input_text)
    if not original:
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
        logger.warning("Correction API not available, returning local aggressive guess")
        return _with_original(input_text, _local_aggressive_guess(input_text))

    try:
        logger.info(f"Processing text via {ACTIVE_PROVIDER}/{ACTIVE_MODEL}: {input_text[:50]}...")

        response = client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(letters=input_text.strip()),
                },
            ],
            temperature=CORRECTION_TEMPERATURE,
            max_tokens=max(80, len(input_text.split()) + 80),
        )

        result = response.choices[0].message.content if response.choices else None
        cleaned = (result or "").strip().strip('"').strip("'")
        # Drop accidental "ORIGINAL / guess" wrappers from the model
        if " / " in cleaned:
            cleaned = cleaned.split(" / ")[-1].strip()
        # Strip common prefixes
        for prefix in ("output:", "corrected:", "guess:", "phrase:"):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()

        logger.info(f"LLM result: {cleaned}")
        guessed = _boost_guess(input_text, cleaned if cleaned else "")
        return _with_original(input_text, guessed)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"{ACTIVE_PROVIDER} API error: {error_msg}")

        if "401" in error_msg or "invalid_api_key" in error_msg.lower():
            logger.error(f"Invalid {ACTIVE_PROVIDER} API key. Check your .env file.")
        elif "429" in error_msg or "rate_limit" in error_msg.lower():
            logger.error(f"{ACTIVE_PROVIDER} rate limit exceeded. Try again later.")
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            logger.error(
                f"Model {ACTIVE_MODEL!r} unavailable. Set GROQ_MODEL=llama-3.1-8b-instant in .env"
            )

        return _with_original(input_text, _local_aggressive_guess(input_text))


if __name__ == "__main__":
    samples = [
        "H E L L O W O R L D",
        "H E L O",
        "T H A M K Y O U",
        "P L E A S N E H E L P",
        "I L O V E C O D I N G",
        "B M W",
        "H O W A R E Y O U",
    ]
    for sample in samples:
        print(f"Input:  {sample}")
        print(f"Local:  {_with_original(sample, _local_aggressive_guess(sample))}")
        print(f"Full:   {generate_sentences(sample)}")
        print("-" * 40)
