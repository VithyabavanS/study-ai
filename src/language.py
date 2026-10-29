"""Detect whether a question is written in Tamil or English."""

TAMIL_START, TAMIL_END = "\u0b80", "\u0bff"


def is_tamil_char(ch):
    return TAMIL_START <= ch <= TAMIL_END


def detect_language(text):
    """Return "Tamil" if at least half of the letters are Tamil script, else "English".

    Tamil vowel signs are not counted by str.isalpha(), so Tamil characters are
    counted separately. Using a share (not "any Tamil character") handles
    code-mixed questions that combine Tamil and English words.
    """
    tamil = sum(1 for ch in text if is_tamil_char(ch))
    letters = sum(1 for ch in text if ch.isalpha() or is_tamil_char(ch))
    if letters and tamil / letters >= 0.5:
        return "Tamil"
    return "English"