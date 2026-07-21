"""
Voice Module

Text-to-speech via ElevenLabs (optional). Browser Web Speech API is the
default free provider for the web UI — see SPEAK_PROVIDER in .env.
"""

import os
import logging
import tempfile
from typing import Optional
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, '..', '..', '.env'))
load_dotenv()

API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID: Optional[str] = os.getenv("ELEVENLABS_VOICE_ID")

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
DEFAULT_STABILITY = 0.7
DEFAULT_SIMILARITY_BOOST = 0.8


def _has_valid_elevenlabs_config() -> bool:
    return bool(
        API_KEY and VOICE_ID
        and not API_KEY.startswith("your_")
        and not VOICE_ID.startswith("your_")
    )


def text_to_speech(
    text: str,
    stability: float = DEFAULT_STABILITY,
    similarity_boost: float = DEFAULT_SIMILARITY_BOOST
) -> Optional[bytes]:
    """Convert text to speech with ElevenLabs. Returns MP3 bytes or None."""
    if not text or not text.strip():
        logger.warning("Empty text provided to text_to_speech")
        return None

    if not _has_valid_elevenlabs_config():
        logger.error("ElevenLabs keys missing or still placeholders in .env")
        return None

    cleaned_text = ' '.join(text.split())
    logger.info(f"Converting text to speech: {cleaned_text[:50]}...")

    url = f"{ELEVENLABS_API_URL}/{VOICE_ID}"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    data = {
        "text": cleaned_text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
        },
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200 and response.content:
            return response.content
        logger.error(f"ElevenLabs API error: {response.status_code}, {response.text[:300]}")
        return None
    except requests.RequestException as e:
        logger.error(f"ElevenLabs API request failed: {e}")
        return None


def text_to_speech_and_play(
    text: str,
    output_file: str = "output.mp3",
    stability: float = DEFAULT_STABILITY,
    similarity_boost: float = DEFAULT_SIMILARITY_BOOST
) -> bool:
    """Generate speech and play locally (CLI). Prefer browser TTS for the web UI."""
    audio = text_to_speech(text, stability=stability, similarity_boost=similarity_boost)
    if not audio:
        return False

    path = output_file if os.path.isabs(output_file) else os.path.join(
        tempfile.gettempdir(), os.path.basename(output_file)
    )
    try:
        with open(path, "wb") as file:
            file.write(audio)
        try:
            from playsound import playsound
            playsound(path)
        except Exception as e:
            logger.error(f"Error playing audio locally: {e}")
            return False
        return True
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    sample_text = "Hello! This is a demonstration of ElevenLabs text-to-speech API."
    success = text_to_speech_and_play(sample_text)
    print(f"Text-to-speech {'succeeded' if success else 'failed'}")
