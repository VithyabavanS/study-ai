"""Read answers aloud with Microsoft Edge's online neural voices (edge-tts).

Uses a Sri Lankan Tamil voice for Tamil answers. Needs an internet connection;
edge-tts is an unofficial client, so the service can change without notice.
"""

import asyncio
import re

import edge_tts

from exceptions import SpeechError
from language import detect_language

VOICES = {"Tamil": "ta-LK-SaranyaNeural", "English": "en-GB-SoniaNeural"}
RATES = {"Slow": "-20%", "Normal": "+0%", "Fast": "+20%"}


def speech_text(answer):
    """Answer text without citation marks and formatting symbols."""
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", answer)
    text = re.sub(r"[*_#`$\\{}]", " ", text)
    return " ".join(text.split())


async def _synthesize(text, voice, rate):
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


def synthesize(answer, speed="Normal"):
    """Return MP3 bytes of the answer read aloud in its own language."""
    text = speech_text(answer)
    if not text:
        raise SpeechError("There is no text to read aloud.")
    voice = VOICES[detect_language(text)]
    try:
        audio = asyncio.run(_synthesize(text, voice, RATES[speed]))
    except Exception as e:
        raise SpeechError(f"The read-aloud service could not be reached: {e}") from e
    if not audio:
        raise SpeechError("The read-aloud service returned no audio.")
    return audio