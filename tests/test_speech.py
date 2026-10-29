import pytest

import speech
from exceptions import SpeechError


def test_speech_text_removes_citations_and_formatting():
    answer = "The dot products are scaled by $1/\\sqrt{d_k}$ [1, 2]. **Adam** was used [3]."
    expected = "The dot products are scaled by 1/ sqrt d k . Adam was used ."
    assert speech.speech_text(answer) == expected


def test_tamil_answer_uses_tamil_voice(monkeypatch):
    used = {}

    async def fake(text, voice, rate):
        used["voice"], used["rate"] = voice, rate
        return b"mp3"

    monkeypatch.setattr(speech, "_synthesize", fake)
    assert speech.synthesize("நொதியம் RuBisCO ஆகும் [1].", "Slow") == b"mp3"
    assert used == {"voice": speech.VOICES["Tamil"], "rate": "-20%"}


def test_service_failure_raises_speech_error(monkeypatch):
    async def broken(text, voice, rate):
        raise OSError("no network")

    monkeypatch.setattr(speech, "_synthesize", broken)
    with pytest.raises(SpeechError):
        speech.synthesize("Hello")