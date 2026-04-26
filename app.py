# requirements: streamlit, pdfplumber, trafilatura, openai, langdetect, anthropic

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import anthropic
import pdfplumber
import streamlit as st
import trafilatura
from langdetect import LangDetectException, detect
from openai import OpenAI

# ── Configuratie ──────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
AUDIO_DIR  = DATA_DIR / "audio"
TEXTS_FILE = DATA_DIR / "texts.json"

DATA_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

MAX_PDF_SIZE_MB      = 25
TTS_MAX_CHARS        = 4096
TTS_PARALLEL_WORKERS = 8           # Aantal gelijktijdige TTS-calls
PRICE_PER_1M_CHARS   = 15.00       # OpenAI tts-1 prijs (USD)

VOICE_OPTIONS = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

st.set_page_config(
    page_title="Voorlezen",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Bibliotheek ───────────────────────────────────────────────────────────────

def load_library() -> list[dict]:
    if TEXTS_FILE.exists():
        try:
            return json.loads(TEXTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_library(library: list[dict]) -> None:
    TEXTS_FILE.write_text(
        json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def make_entry_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def add_to_library(title: str, source: str, text: str) -> str:
    library = load_library()
    entry_id = make_entry_id(text)
    library  = [e for e in library if e["id"] != entry_id]
    library.insert(0, {
        "id":     entry_id,
        "title":  title,
        "source": source,
        "date":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "chars":  len(text),
        "text":   text,
    })
    save_library(library)
    return entry_id


def delete_from_library(entry_id: str) -> None:
    library = [e for e in load_library() if e["id"] != entry_id]
    save_library(library)
    for audio_file in AUDIO_DIR.glob(f"{entry_id}_*.mp3"):
        audio_file.unlink(missing_ok=True)


# ── Audiocache ────────────────────────────────────────────────────────────────

def audio_cache_path(entry_id: str, voice: str) -> Path:
    safe_voice = re.sub(r"[^a-z]", "", voice.lower())
    return AUDIO_DIR / f"{entry_id}_{safe_voice}.mp3"


def load_cached_audio(entry_id: str | None, voice: str) -> bytes | None:
    if not entry_id:
        return None
    path = audio_cache_path(entry_id, voice)
    return path.read_bytes() if path.exists() else None


def save_cached_audio(entry_id: str, voice: str, audio_bytes: bytes) -> None:
    audio_cache_path(entry_id, voice).write_bytes(audio_bytes)


def has_cached_audio(entry_id: str, voice: str) -> bool:
    return audio_cache_path(entry_id, voice).exists()


def list_cached_voices(entry_id: str | None) -> list[str]:
    """Geef alle stemmen terug waarvoor audio in cache staat."""
    if not entry_id:
        return []
    return [v for v in VOICE_OPTIONS if has_cached_audio(entry_id, v)]


# ── API-clients (gecached) ────────────────────────────────────────────────────

@st.cache_resource
def get_openai_client() -> OpenAI | None:
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else None


@st.cache_resource
def get_anthropic_client() -> anthropic.Anthropic | None:
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=api_key) if api_key else None


# ── Tekstextractie ────────────────────────────────────────────────────────────

def extract_text_pdf(uploaded_file) -> str:
    if uploaded_file.size > MAX_PDF_SIZE_MB * 1024 * 1024:
        st.error(f"PDF te groot (max {MAX_PDF_SIZE_MB} MB).")
        return ""
    text_blocks = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                top    = page.height * 0.10
                bottom = page.height * 0.90
                cropped   = page.within_bbox((0, top, page.width, bottom))
                page_text = cropped.extract_text()
                if page_text:
                    text_blocks.append(page_text.strip())
    except Exception as e:
        st.error(f"PDF-extractie mislukt: {e}")
        return ""
    return "\n\n".join(text_blocks)


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def extract_text_url(url: str) -> str:
    if not is_valid_url(url):
        st.error("Ongeldige URL. Gebruik http:// of https://")
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            st.error("URL kon niet worden opgehaald.")
            return ""
        text = trafilatura.extract(downloaded)
        if not text:
            st.error("Geen hoofdtekst gevonden op de opgegeven URL.")
            return ""
        return text.strip()
    except Exception as e:
        st.error(f"URL-extractie mislukt: {e}")
        return ""


# ── Vertaling ─────────────────────────────────────────────────────────────────

def translate_to_dutch(text: str) -> str:
    client = get_anthropic_client()
    if not client:
        st.error("Anthropic API-sleutel ontbreekt.")
        return ""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8096,
            system=(
                "Je bent een professionele vertaler. Vertaal de aangeleverde tekst "
                "naar correct en natuurlijk Nederlands. Behoud de originele opmaak "
                "en alinea-indeling. Geef alleen de vertaalde tekst terug, zonder "
                "toelichting of extra tekst."
            ),
            messages=[{"role": "user", "content": text}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        st.error(f"Vertaling mislukt: {e}")
        return ""


def generate_title(text: str) -> str:
    first = text.strip().split("\n")[0][:80]
    return first if first else "Onbenoemd artikel"


# ── Taaldetectie (gecached) ───────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=50)
def detect_language(text: str) -> str:
    if not text.strip():
        return ""
    try:
        return detect(text)
    except LangDetectException:
        return ""


# ── TTS ───────────────────────────────────────────────────────────────────────

def split_text(text: str, max_chars: int = TTS_MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences, current = [], ""
    for char in text:
        current += char
        if char in ".!?\n" and len(current) > 1:
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)

    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_chars:
            chunk += sentence
        else:
            if chunk:
                chunks.append(chunk.strip())
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars])
                chunk = ""
            else:
                chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return [c for c in chunks if c]


def _tts_chunk(client: OpenAI, voice: str, chunk: str) -> bytes:
    return client.audio.speech.create(model="tts-1", voice=voice, input=chunk).content


def text_to_audio_parallel(client: OpenAI, text: str, voice: str) -> bytes:
    """Genereer TTS-audio met parallelle API-calls voor maximale snelheid."""
    chunks = split_text(text)

    if len(chunks) == 1:
        try:
            return _tts_chunk(client, voice, chunks[0])
        except Exception as e:
            st.error(f"TTS API-fout: {e}")
            return b""

    results: list[bytes | None] = [None] * len(chunks)
    progress  = st.progress(0.0, text=f"Audio genereren ({len(chunks)} chunks parallel)...")
    completed = 0

    try:
        with ThreadPoolExecutor(max_workers=TTS_PARALLEL_WORKERS) as executor:
            futures = {
                executor.submit(_tts_chunk, client, voice, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    st.error(f"TTS API-fout bij chunk {idx + 1}: {e}")
                    return b""
                completed += 1
                progress.progress(
                    completed / len(chunks),
                    text=f"Chunk {completed}/{len(chunks)} klaar...",
                )
    finally:
        progress.empty()

    if any(r is None for r in results):
        return b""
    return b"".join(results)


# ── Sidebar bibliotheek ───────────────────────────────────────────────────────

def render_sidebar():
    library = load_library()
    st.sidebar.header("📚 Bibliotheek")
    st.sidebar.caption(f"{len(library)} artikelen opgeslagen")

    if not library:
        st.sidebar.info("Nog geen artikelen opgeslagen.")
        return

    for entry in library:
        with st.sidebar.container(border=True):
            st.markdown(f"**{entry['title']}**")
            cached_voices = sum(
                1 for v in VOICE_OPTIONS if has_cached_audio(entry["id"], v)
            )
            audio_badge = f"· 🔊 {cached_voices}" if cached_voices else ""
            chars = entry.get("chars", len(entry["text"]))
            st.caption(f"{entry['date']} · {chars:,} tekens {audio_badge}".replace(",", "."))

            cols = st.columns([3, 1])
            with cols[0]:
                if st.button("Laden", key=f"load_{entry['id']}", use_container_width=True):
                    st.session_state["extracted_text"]   = entry["text"]
                    st.session_state["current_source"]   = entry["source"]
                    st.session_state["current_entry_id"] = entry["id"]
                    st.session_state["current_title"]    = entry["title"]
                    st.rerun()
            with cols[1]:
                if st.button("🗑", key=f"del_{entry['id']}", use_container_width=True,
                             help="Verwijder dit artikel en bijbehorende audio"):
                    delete_from_library(entry["id"])
                    if st.session_state.get("current_entry_id") == entry["id"]:
                        for k in ("extracted_text", "current_source",
                                  "current_entry_id", "current_title"):
                            st.session_state.pop(k, None)
                    st.rerun()


# ── Hoofdsecties ──────────────────────────────────────────────────────────────

def render_input_section():
    st.subheader("1. Tekst inladen")
    tab_pdf, tab_url = st.tabs(["📄 PDF upload", "🔗 URL invoer"])

    with tab_pdf:
        uploaded = st.file_uploader(
            "Kies een PDF", type=["pdf"], label_visibility="collapsed",
        )
        if uploaded and st.button("Tekst extraheren", key="extract_pdf"):
            with st.spinner("PDF verwerken..."):
                text = extract_text_pdf(uploaded)
            if text:
                st.session_state["extracted_text"]   = text
                st.session_state["current_source"]   = uploaded.name
                st.session_state["current_entry_id"] = None
                st.session_state["current_title"]    = uploaded.name.rsplit(".", 1)[0]
                st.rerun()

    with tab_url:
        url = st.text_input(
            "URL", placeholder="https://example.com/artikel", label_visibility="collapsed",
        )
        if st.button("Tekst extraheren", key="extract_url"):
            if not url.strip():
                st.error("Voer een URL in.")
            else:
                with st.spinner("URL ophalen..."):
                    text = extract_text_url(url.strip())
                if text:
                    st.session_state["extracted_text"]   = text
                    st.session_state["current_source"]   = url.strip()
                    st.session_state["current_entry_id"] = None
                    st.session_state["current_title"]    = generate_title(text)
                    st.rerun()


def render_text_section() -> str:
    st.subheader("2. Tekst bewerken")

    current_text  = st.session_state.get("extracted_text", "")
    editable_text = st.text_area(
        "Tekst", value=current_text, height=280,
        placeholder="Tekst verschijnt hier na extractie of bij het laden uit de bibliotheek.",
        label_visibility="collapsed",
    )

    if editable_text.strip():
        chars = len(editable_text)
        words = len(editable_text.split())
        lang  = detect_language(editable_text).upper() or "?"
        cost  = (chars / 1_000_000) * PRICE_PER_1M_CHARS

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tekens",    f"{chars:,}".replace(",", "."))
        c2.metric("Woorden",   f"{words:,}".replace(",", "."))
        c3.metric("Taal",      lang)
        c4.metric("TTS-kosten", f"${cost:.3f}")

    return editable_text


def render_save_section(editable_text: str):
    if not editable_text.strip():
        return
    with st.expander("💾 Opslaan in bibliotheek"):
        default = st.session_state.get("current_title") or generate_title(editable_text)
        title   = st.text_input("Titel", value=default[:80])
        if st.button("Opslaan", type="primary"):
            if not title.strip():
                st.error("Geef een titel op.")
                return
            entry_id = add_to_library(
                title=title.strip(),
                source=st.session_state.get("current_source", ""),
                text=editable_text.strip(),
            )
            st.session_state["current_entry_id"] = entry_id
            st.session_state["current_title"]    = title.strip()
            st.success(f"Opgeslagen als '{title.strip()}'.")
            st.rerun()


def render_translate_section(editable_text: str):
    text = editable_text.strip()
    if not text:
        return
    lang = detect_language(text)
    if lang and lang != "nl":
        if st.button(f"🌍 Vertaal naar Nederlands ({lang.upper()} gedetecteerd)"):
            with st.spinner("Tekst vertalen met Claude Sonnet..."):
                translated = translate_to_dutch(text)
            if translated:
                st.session_state["extracted_text"]   = translated
                st.session_state["current_entry_id"] = None
                st.success("Vertaald naar Nederlands.")
                st.rerun()


def render_saved_audio_section():
    """Toon alle opgeslagen MP3's voor het huidige artikel."""
    entry_id = st.session_state.get("current_entry_id")
    if not entry_id:
        return

    cached_voices = list_cached_voices(entry_id)
    if not cached_voices:
        return

    title_safe = re.sub(
        r"[^\w\-]", "_",
        st.session_state.get("current_title", "audio") or "audio",
    )[:50]

    with st.expander(f"🎵 Opgeslagen audio ({len(cached_voices)} versie{'s' if len(cached_voices) > 1 else ''})", expanded=True):
        for voice in cached_voices:
            audio_bytes = load_cached_audio(entry_id, voice)
            if not audio_bytes:
                continue

            cols = st.columns([1, 4, 1, 1])
            with cols[0]:
                st.markdown(f"**🔊 {voice}**")
            with cols[1]:
                st.audio(audio_bytes, format="audio/mp3")
            with cols[2]:
                size_kb = len(audio_bytes) / 1024
                st.caption(f"{size_kb:.0f} KB")
            with cols[3]:
                st.download_button(
                    "⬇️",
                    data=audio_bytes,
                    file_name=f"{title_safe}_{voice}.mp3",
                    mime="audio/mpeg",
                    key=f"dl_saved_{voice}",
                    help="Download MP3",
                )


def render_tts_section(editable_text: str):
    st.subheader("3. Voorlezen")

    entry_id = st.session_state.get("current_entry_id")

    cols = st.columns([2, 1])
    with cols[0]:
        if entry_id:
            labels = [
                ("🔊 " if has_cached_audio(entry_id, v) else "  ") + v
                for v in VOICE_OPTIONS
            ]
            idx   = st.selectbox(
                "Stem", range(len(VOICE_OPTIONS)),
                format_func=lambda i: labels[i], index=4,
            )
            voice = VOICE_OPTIONS[idx]
            st.caption("🔊 = audio aanwezig in cache (geen API-kosten)")
        else:
            voice = st.selectbox("Stem", VOICE_OPTIONS, index=4)

    with cols[1]:
        st.write("")
        st.write("")
        play = st.button("▶️  Voorlezen", type="primary", use_container_width=True)

    if not play:
        return

    text_to_read = editable_text.strip()
    if not text_to_read:
        st.error("Er is geen tekst om voor te lezen.")
        return

    cached = load_cached_audio(entry_id, voice)
    if cached:
        st.success("⚡ Direct uit cache geladen — geen API-kosten.")
        audio_bytes = cached
    else:
        client = get_openai_client()
        if not client:
            st.error("OpenAI API-sleutel ontbreekt.")
            return
        audio_bytes = text_to_audio_parallel(client, text_to_read, voice)
        if audio_bytes and entry_id:
            save_cached_audio(entry_id, voice, audio_bytes)
            st.success("✅ Audio gegenereerd en opgeslagen in cache.")
        elif audio_bytes:
            st.info("Tip: sla de tekst op in de bibliotheek om de audio te bewaren.")

    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
        title_safe = re.sub(
            r"[^\w\-]", "_",
            st.session_state.get("current_title", "audio") or "audio",
        )[:50]
        st.download_button(
            "⬇️  Download MP3",
            data=audio_bytes,
            file_name=f"{title_safe}_{voice}.mp3",
            mime="audio/mpeg",
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    st.title("🎧 Voorlezen")
    st.caption("Upload een PDF of voer een URL in en laat de tekst voorlezen.")

    render_input_section()
    st.divider()
    editable_text = render_text_section()
    render_save_section(editable_text)
    render_translate_section(editable_text)
    st.divider()
    render_saved_audio_section()
    render_tts_section(editable_text)


if __name__ == "__main__":
    main()
