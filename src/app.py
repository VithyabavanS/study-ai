"""StudyAI: ask questions about a PDF in Tamil or English."""

import html
import os

import streamlit as st
from dotenv import load_dotenv

from chat_engine import ChatEngine
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DEVICE,
    LLM_MODEL,
    RERANKER_MODEL,
    RETRIEVAL_MODE,
    TOP_K,
)
from document_processor import DocumentProcessor
from exceptions import StudyAIError
from language import detect_language
from retrieval import build_retriever
from speech import RATES as SPEECH_RATES
from speech import synthesize

load_dotenv()

st.set_page_config(page_title="StudyAI", page_icon="📖", layout="wide")

FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400"
    "&family=Literata:opsz,wght@7..72,400;7..72,600"
    "&family=Noto+Sans+Tamil:wght@400;600"
    "&family=Noto+Serif+Tamil:wght@400;600&display=swap"
)

TEXT_SIZES = {"Normal": 16, "Large": 19, "Extra large": 22}  # base font size in px

# Design: a study desk under frosted glass. The page background is soft, blurred ink
# blue, highlighter yellow and margin red; the panels are frosted glass on top. The
# answer itself stays solid paper (an exercise-book page with a red margin line),
# because reading must never happen on a see-through surface.
BASE_CSS = """
@import url('{fonts}');

:root {{
  --paper: #EEF2F7;
  --sheet: #FFFFFF;
  --ink: #1B2A41;
  --ink-soft: #46566F;
  --pen: #2546A8;
  --rule: #C9D6E6;
  --margin: #C8433A;
  --highlight: #FFE27A;
  --glass: rgba(255, 255, 255, 0.58);
  --glass-edge: rgba(255, 255, 255, 0.85);
  --glass-shadow: 0 10px 40px rgba(27, 42, 65, 0.10);
  --ui: 'Atkinson Hyperlegible', 'Noto Sans Tamil', system-ui, sans-serif;
  --read: 'Literata', 'Noto Serif Tamil', Georgia, serif;
}}

html {{ font-size: {base_px}px; }}

.stApp {{
  color: var(--ink);
  background:
    radial-gradient(circle at 8% 12%, rgba(37, 70, 168, 0.20), transparent 38%),
    radial-gradient(circle at 92% 6%, rgba(255, 226, 122, 0.45), transparent 34%),
    radial-gradient(circle at 80% 92%, rgba(200, 67, 58, 0.12), transparent 40%),
    var(--paper);
  background-attachment: fixed;
}}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp h1, .stApp h2, .stApp h3 {{ font-family: var(--ui); }}
.stApp p, .stApp li {{ font-size: 1.05rem; line-height: 1.7; }}
.stApp input {{ font-size: 1.05rem; }}

header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }}

.stApp :focus-visible {{ outline: 3px solid var(--pen) !important; outline-offset: 2px; }}
.sr-only {{
  position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap;
}}

/* Frosted glass panels */
.st-key-masthead, .st-key-panel_doc, .st-key-panel_ask {{
  background: var(--glass);
  -webkit-backdrop-filter: blur(18px) saturate(150%);
  backdrop-filter: blur(18px) saturate(150%);
  border: 1px solid var(--glass-edge);
  border-radius: 20px;
  box-shadow: var(--glass-shadow), inset 0 1px 0 rgba(255, 255, 255, 0.9);
  padding: 1.5rem 1.75rem;
}}
.st-key-masthead {{ margin-bottom: 1.5rem; }}
.stApp .brand {{
  font-size: 2.1rem; font-weight: 700; letter-spacing: -0.01em; line-height: 1.2;
  margin: 0 0 .35rem;
}}
.stApp .lede {{ color: var(--ink-soft); margin: 0; max-width: 62ch; }}
.stApp .mode {{ color: var(--ink-soft); font-size: .9rem; margin: .75rem 0 0; }}

.stApp h2 {{ font-size: 1.15rem; font-weight: 700; margin: 0 0 .5rem; padding: 0; }}
.stApp .status {{ color: var(--ink-soft); font-size: .95rem; }}

[data-testid="stTextInputRootElement"] {{ border-radius: 10px; }}

/* The answer: solid paper resting on the glass */
.st-key-answer {{
  background:
    linear-gradient(to right, transparent 3.25rem, var(--margin) 3.25rem,
                    var(--margin) calc(3.25rem + 2px), transparent calc(3.25rem + 2px)),
    var(--sheet);
  border: 1px solid var(--rule);
  border-radius: 6px;
  box-shadow: 0 1px 2px rgba(27, 42, 65, 0.06), 0 12px 28px rgba(27, 42, 65, 0.10);
  padding: 1.5rem 2rem 1.25rem 4.5rem;
  margin-top: 1rem;
}}
.st-key-answer .asked {{ color: var(--ink-soft); font-weight: 700; margin: 0 0 .5rem; }}
.st-key-answer [data-testid="stMarkdownContainer"] p,
.st-key-answer [data-testid="stMarkdownContainer"] li {{
  font-family: var(--read); font-size: 1.15rem; line-height: 1.8;
}}
.st-key-answer [data-testid="stMarkdownContainer"] .sources-title {{
  font-family: var(--ui); font-size: 1rem; color: var(--ink-soft); font-weight: 700;
  margin: 1.25rem 0 .5rem; padding-top: 1rem; border-top: 1px solid var(--rule);
}}
.st-key-answer [data-testid="stMarkdownContainer"] ol.sources {{
  list-style: none; padding: 0; margin: 0 0 0 -3.9rem;
}}
.st-key-answer [data-testid="stMarkdownContainer"] ol.sources li {{
  display: grid; grid-template-columns: 1.6rem 1fr; column-gap: 2.3rem;
  margin: 0 0 1rem; padding: 0; font-family: var(--ui); font-size: 1rem;
}}
.sources .mark {{
  background: var(--highlight); color: var(--ink); font-weight: 700; text-align: center;
  height: 1.6rem; line-height: 1.6rem; border-radius: 2px; font-size: .95rem;
}}
.st-key-answer [data-testid="stMarkdownContainer"] .sources p {{
  margin: 0; font-family: var(--ui); color: var(--ink-soft); font-size: 1rem;
  line-height: 1.7;
}}
.sources .where {{ color: var(--ink-soft); font-size: .85rem; }}
.st-key-answer button [data-testid="stMarkdownContainer"] p {{
  font-family: var(--ui); font-size: 1rem; line-height: 1.4;
}}

/* People who ask their system for less transparency get solid panels */
@media (prefers-reduced-transparency: reduce) {{
  .st-key-masthead, .st-key-panel_doc, .st-key-panel_ask {{
    background: var(--sheet); -webkit-backdrop-filter: none; backdrop-filter: none;
  }}
}}

@media (max-width: 640px) {{
  .block-container {{ padding-left: .75rem; padding-right: .75rem; }}
  .st-key-masthead, .st-key-panel_doc, .st-key-panel_ask {{ padding: 1.1rem 1rem; }}
  .st-key-answer {{
    padding: 1.25rem 1rem 1rem 3.25rem;
    background:
      linear-gradient(to right, transparent 2.25rem, var(--margin) 2.25rem,
                      var(--margin) calc(2.25rem + 2px), transparent calc(2.25rem + 2px)),
      var(--sheet);
  }}
  .st-key-answer [data-testid="stMarkdownContainer"] ol.sources {{ margin-left: -2.9rem; }}
  .st-key-answer [data-testid="stMarkdownContainer"] ol.sources li {{ column-gap: 1.3rem; }}
}}
"""

# WCAG 1.4.12 text spacing: helps many readers with dyslexia or low vision
SPACING_CSS = """
.stApp p, .stApp li, .stApp label, .stApp input {
  letter-spacing: 0.06em; word-spacing: 0.16em;
}
.stApp p, .stApp li { line-height: 2 !important; margin-bottom: 1em; }
"""

# Night mode: deep ink-blue desk, the answer page turns to dark paper
NIGHT_CSS = """
:root {
  --paper: #10192B; --sheet: #1B263B; --ink: #E7EDF6; --ink-soft: #AAB7CB;
  --pen: #93AAFF; --rule: #34435F; --margin: #E0675D; --highlight: #F2D35C;
  --glass: rgba(27, 38, 59, 0.55); --glass-edge: rgba(255, 255, 255, 0.10);
  --glass-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
}
.stApp {
  background:
    radial-gradient(circle at 8% 12%, rgba(80, 110, 220, 0.28), transparent 40%),
    radial-gradient(circle at 92% 6%, rgba(242, 211, 92, 0.10), transparent 34%),
    radial-gradient(circle at 80% 92%, rgba(224, 103, 93, 0.14), transparent 40%),
    var(--paper) !important;
}
.st-key-masthead, .st-key-panel_doc, .st-key-panel_ask {
  box-shadow: var(--glass-shadow), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
}
.stApp p, .stApp li, .stApp label, .stApp h2, .stApp span, .stApp small,
[data-testid="stMarkdownContainer"] { color: var(--ink); }
.stApp .lede, .stApp .mode, .stApp .status, .st-key-answer .asked { color: var(--ink-soft); }
.sources .mark { color: #1B2A41 !important; }
.st-key-answer { box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35) !important; }
[data-testid="stTextInputRootElement"] { background: var(--sheet); border-color: var(--rule); }
[data-testid="stTextInputRootElement"] input { background: transparent; color: var(--ink); }
[data-testid="stTextInputRootElement"] input::placeholder { color: var(--ink-soft); }
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploaderFile"],
[data-testid="stPopoverBody"], [data-testid="stExpander"] details {
  background: var(--sheet) !important; color: var(--ink) !important;
  border-color: var(--rule) !important;
}
[data-testid^="stBaseButton-secondary"], button[data-variant="segmented_control"] {
  background: var(--sheet) !important; border-color: var(--rule) !important;
}
[data-testid^="stBaseButton-secondary"] *, button[data-variant="segmented_control"] * {
  color: var(--ink) !important;
}
button[data-variant="segmented_control"][data-selected="true"] {
  background: var(--pen) !important; border-color: var(--pen) !important;
}
button[data-variant="segmented_control"][data-selected="true"] *,
[data-testid^="stBaseButton-primary"] * { color: #10192B !important; }
[data-testid^="stBaseButton-primary"] { background: var(--pen) !important; }
[data-testid="stCheckbox"] label > div:first-of-type { background: #5A6B8A !important; }
[data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type {
  background: var(--pen) !important;
}
"""

# Solid black on white, no glass, strong borders and underlined focus
CONTRAST_CSS = """
:root {
  --paper: #FFFFFF; --sheet: #FFFFFF; --ink: #000000; --ink-soft: #000000;
  --pen: #0033CC; --rule: #000000; --highlight: #FFD400;
  --glass: #FFFFFF; --glass-edge: #000000; --glass-shadow: none;
}
.stApp { background: #FFFFFF !important; }
.st-key-masthead, .st-key-panel_doc, .st-key-panel_ask, .st-key-answer {
  -webkit-backdrop-filter: none !important; backdrop-filter: none !important;
  border-width: 2px !important; box-shadow: none !important;
}
[data-testid="stTextInputRootElement"] { border: 2px solid #000000 !important; }
[data-testid^="stBaseButton-primary"] {
  background: #000000 !important; border-color: #000000 !important; color: #FFFFFF !important;
}
.stApp :focus-visible { outline: 4px solid #0033CC !important; outline-offset: 3px; }
"""

LEDE_EN = (
    "Ask questions about your PDF in Tamil or English. "
    "Answers come only from the document, with the passages they came from."
)
LEDE_TA = "உங்கள் PDF பற்றித் தமிழிலோ ஆங்கிலத்திலோ கேள்வி கேளுங்கள்."
PLACEHOLDER = "e.g. Why is attention scaled by √dk?  or  ஒளித்தொகுப்பு எங்கே நடைபெறுகிறது?"


@st.cache_resource
def load_reranker():
    """Load the cross-encoder once and reuse it across reruns and users."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL, device=EMBEDDING_DEVICE)


def lang_attr(text):
    """HTML lang code, so screen readers pronounce Tamil text correctly."""
    return "ta" if detect_language(text) == "Tamil" else "en"


def init_state():
    defaults = {
        "file_id": None, "file_name": None, "passages": 0, "retriever": None, "history": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def setting(key, default):
    """Current value of a reading setting (widgets live in the settings popover)."""
    value = st.session_state.get(key)
    return default if value is None else value


def apply_styles():
    """Build the page CSS from the reader's settings."""
    css = BASE_CSS.format(fonts=FONTS, base_px=TEXT_SIZES[setting("text_size", "Normal")])
    if setting("wide_spacing", False):
        css += SPACING_CSS
    if setting("night_mode", False):
        css += NIGHT_CSS
    if setting("high_contrast", False):  # wins over night mode
        css += CONTRAST_CSS
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def reading_settings():
    with st.popover("Reading settings", icon=":material/text_fields:", width="stretch"):
        st.segmented_control(
            "Text size", list(TEXT_SIZES), default="Normal", key="text_size", required=True
        )
        st.toggle("Night mode", key="night_mode", help="Light text on a dark background.")
        st.toggle("High contrast", key="high_contrast",
                  help="Black text on white, with no see-through panels.")
        st.toggle("Wider spacing", key="wide_spacing",
                  help="More space between letters, words and lines. "
                       "Helps many readers with dyslexia.")
        st.segmented_control(
            "Read-aloud speed", list(SPEECH_RATES), default="Normal", key="speech_rate",
            required=True,
        )


def render_masthead():
    if os.getenv("GEMINI_API_KEY"):
        mode = f"Answers written by {LLM_MODEL}. Retrieval: {RETRIEVAL_MODE.replace('_', ' + ')}."
    else:
        mode = ("Free mode: answers are key sentences from the document. "
                "Add GEMINI_API_KEY to .env for written answers.")
    with st.container(key="masthead"):
        text, settings = st.columns([2.2, 1], vertical_alignment="top")
        with text:
            st.markdown(
                f"""<p class="brand">StudyAI</p>
                    <p class="lede">{LEDE_EN}</p>
                    <p class="lede" lang="ta">{LEDE_TA}</p>
                    <p class="mode">{html.escape(mode)}</p>""",
                unsafe_allow_html=True,
            )
        with settings:
            reading_settings()


def process_upload(uploaded):
    """Read a newly uploaded PDF once and build its retriever."""
    file_id = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.file_id == file_id:
        return
    with st.spinner("Reading the PDF…"):
        try:
            chunks = DocumentProcessor(
                uploaded, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            ).process()
            reranker = load_reranker() if RETRIEVAL_MODE == "hybrid_rerank" else None
            retriever = build_retriever(chunks, RETRIEVAL_MODE, reranker_model=reranker)
        except StudyAIError as e:
            st.session_state.update(file_id=None, retriever=None)
            st.error(f"Couldn't read this PDF: {e}")
            return
    st.session_state.update(
        file_id=file_id, file_name=uploaded.name, passages=len(chunks),
        retriever=retriever, history=[],
    )


@st.cache_data(show_spinner=False, max_entries=50)
def answer_audio(answer, speed):
    """MP3 of the answer read aloud; cached so replaying costs nothing."""
    return synthesize(answer, speed)


def read_aloud(entry):
    """Listen button: makes an MP3 on the server and plays it with a normal audio player."""
    speed = setting("speech_rate", "Normal")
    just_made = False
    if st.button("Listen to answer", icon=":material/volume_up:", key="listen"):
        with st.spinner("Preparing audio…"):
            try:
                entry["audio"] = answer_audio(entry["answer"], speed)
                just_made = True
            except StudyAIError as e:
                st.error(f"Couldn't read the answer aloud: {e}")
    if entry.get("audio"):
        st.audio(entry["audio"], format="audio/mpeg", autoplay=just_made,
                 alt="The answer, read aloud")


def render_sources(sources):
    items = []
    for number, doc in enumerate(sources, start=1):
        text = " ".join(doc.page_content.split())
        excerpt = text if len(text) <= 320 else text[:320].rsplit(" ", 1)[0] + " …"
        page = doc.metadata.get("page")
        where = f'<span class="where">Page {page + 1}</span>' if isinstance(page, int) else ""
        items.append(
            f'<li><span class="mark" aria-label="Source {number}">{number}</span>'
            f'<div><p lang="{lang_attr(text)}">{html.escape(excerpt)}</p>{where}</div></li>'
        )
    st.markdown(
        '<p class="sources-title">Sources</p>'
        f'<ol class="sources">{"".join(items)}</ol>',
        unsafe_allow_html=True,
    )


def render_answer(entry):
    with st.container(key="answer"):
        st.markdown(
            '<p class="sr-only" role="status">Answer ready.</p>'
            f'<p class="asked" lang="{lang_attr(entry["question"])}">'
            f'{html.escape(entry["question"])}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(entry["answer"])  # Markdown keeps lists and formulas readable
        read_aloud(entry)
        if entry["sources"]:
            render_sources(entry["sources"])


def document_panel():
    with st.container(key="panel_doc"):
        st.markdown("## Your document")
        uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
        if uploaded is None:
            st.session_state.update(file_id=None, retriever=None, history=[])
            st.markdown('<p class="status">No document yet.</p>', unsafe_allow_html=True)
            return
        process_upload(uploaded)
        if st.session_state.retriever is not None:
            st.markdown(
                f'<p class="status">Ready: {st.session_state.passages} passages from '
                f"{html.escape(st.session_state.file_name)}.</p>",
                unsafe_allow_html=True,
            )


def question_panel():
    with st.container(key="panel_ask"):
        st.markdown("## Your question")
        with st.form("ask", clear_on_submit=False, border=False):
            question = st.text_input("Ask in Tamil or English", placeholder=PLACEHOLDER)
            asked = st.form_submit_button("Ask", type="primary")

        if asked:
            if st.session_state.retriever is None:
                st.warning("Upload a PDF first, then ask your question.")
            elif not question.strip():
                st.warning("Type a question, then select Ask.")
            else:
                with st.spinner("Finding the answer in your document…"):
                    try:
                        result = ChatEngine(st.session_state.retriever).get_answer(
                            question, k=TOP_K
                        )
                    except StudyAIError as e:
                        st.error(f"Couldn't get an answer: {e}")
                        return
                st.session_state.history.append(
                    {"question": question, "answer": result["answer"],
                     "sources": result["sources"]}
                )

        history = st.session_state.history
        if not history:
            return
        render_answer(history[-1])
        if len(history) > 1:
            with st.expander(f"Earlier questions ({len(history) - 1})"):
                for entry in reversed(history[:-1]):
                    st.markdown(f"**{entry['question']}**")
                    st.markdown(entry["answer"])


def main():
    init_state()
    apply_styles()
    render_masthead()
    left, right = st.columns([1, 2], gap="large")
    with left:
        document_panel()
    with right:
        question_panel()


if __name__ == "__main__":
    main()