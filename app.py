# requirements: streamlit, pdfplumber, trafilatura, openai, langdetect, anthropic

import json
import os
import re
import uuid
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
DATA_DIR     = Path(__file__).parent / "data"
AUDIO_DIR    = DATA_DIR / "audio"
LIBRARY_FILE = DATA_DIR / "texts.json"

DATA_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

MAX_PDF_SIZE_MB      = 25
TTS_MAX_CHARS        = 4096
TTS_PARALLEL_WORKERS = 8

VOICE_OPTIONS = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

st.set_page_config(
    page_title="Voorlezen",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Bibliotheek (alleen MP3 + metadata) ───────────────────────────────────────

def load_library() -> list[dict]:
    if LIBRARY_FILE.exists():
        try:
            return json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_library(library: list[dict]) -> None:
    LIBRARY_FILE.write_text(
        json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add_to_library(title: str, source_type: str, source: str,
                   voice: str, chars: int, audio_bytes: bytes) -> str:
    entry_id = uuid.uuid4().hex[:12]
    (AUDIO_DIR / f"{entry_id}.mp3").write_bytes(audio_bytes)

    library = load_library()
    library.insert(0, {
        "id":          entry_id,
        "title":       title,
        "source_type": source_type,
        "source":      source,
        "voice":       voice,
        "chars":       chars,
        "date":        datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    save_library(library)
    return entry_id


def delete_from_library(entry_id: str) -> None:
    library = [e for e in load_library() if e["id"] != entry_id]
    save_library(library)
    (AUDIO_DIR / f"{entry_id}.mp3").unlink(missing_ok=True)
    # Backward compatibility: oude bestandsnamen
    for old in AUDIO_DIR.glob(f"{entry_id}_*.mp3"):
        old.unlink(missing_ok=True)


def load_audio(entry_id: str) -> bytes | None:
    new_path = AUDIO_DIR / f"{entry_id}.mp3"
    if new_path.exists():
        return new_path.read_bytes()
    # Backward compatibility
    for old in AUDIO_DIR.glob(f"{entry_id}_*.mp3"):
        return old.read_bytes()
    return None


# ── API-clients ───────────────────────────────────────────────────────────────

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
    uploaded_file.seek(0)
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


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "nl-NL,nl;q=0.9,en;q=0.8",
}


def fetch_html(url: str) -> str | None:
    """Haal HTML op met meerdere strategieën (sommige sites blokkeren bots)."""
    # 1. Trafilatura's eigen fetcher
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded and len(downloaded) > 500:
            return downloaded
    except Exception:
        pass

    # 2. Fallback: urllib met realistische browser-headers
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=BROWSER_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def extract_text_url(url: str) -> str:
    if not is_valid_url(url):
        st.error("Ongeldige URL. Gebruik http:// of https://")
        return ""

    html = fetch_html(url)
    if not html:
        st.error("URL kon niet worden opgehaald.")
        return ""

    # Probeer meerdere extractiestrategieën
    for kwargs in (
        {},
        {"favor_recall": True},
        {"favor_recall": True, "include_comments": False, "include_tables": True},
    ):
        try:
            text = trafilatura.extract(html, **kwargs)
            if text and len(text.strip()) > 100:
                return text.strip()
        except Exception:
            continue

    st.error(
        "Geen hoofdtekst gevonden. De pagina gebruikt mogelijk JavaScript "
        "om de tekst te tonen, of blokkeert geautomatiseerde toegang."
    )
    return ""


# ── Vertaling & samenvatting ──────────────────────────────────────────────────

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
    """Genereer korte beschrijvende titel via Claude (fallback: eerste regel)."""
    client = get_anthropic_client()
    snippet = text.strip()[:2000]
    if not client:
        return text.strip().split("\n")[0][:80] or "Onbenoemd artikel"
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            system=(
                "Je krijgt een artikel of document. Geef één korte, beschrijvende "
                "titel terug (max 70 tekens) die de inhoud samenvat. Geef alleen "
                "de titel zelf, zonder aanhalingstekens of toelichting."
            ),
            messages=[{"role": "user", "content": snippet}],
        )
        return response.content[0].text.strip()[:80]
    except Exception:
        return text.strip().split("\n")[0][:80] or "Onbenoemd artikel"


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


def text_to_audio(client: OpenAI, text: str, voice: str,
                  progress_bar, base: float, span: float) -> bytes:
    """Parallelle TTS. Update progress_bar tussen `base` en `base + span`."""
    chunks = split_text(text)

    if len(chunks) == 1:
        try:
            return _tts_chunk(client, voice, chunks[0])
        except Exception as e:
            st.error(f"TTS API-fout: {e}")
            return b""

    results: list[bytes | None] = [None] * len(chunks)
    completed = 0

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
            progress_bar.progress(
                base + (completed / len(chunks)) * span,
                text=f"🎤 Audio genereren... ({completed}/{len(chunks)} chunks)",
            )

    if any(r is None for r in results):
        return b""
    return b"".join(results)


# ── Hoofdpipeline ─────────────────────────────────────────────────────────────

def process_to_audio(source_type: str, source: str, source_data,
                     voice: str, auto_translate: bool) -> dict | None:
    """Doorloop de hele pipeline: extractie → vertaling → titel → TTS."""
    progress = st.progress(0.0, text="🚀 Verwerken starten...")

    try:
        # 1. Tekstextractie (0% → 20%)
        progress.progress(0.05, text="📄 Tekst extraheren...")
        if source_type == "pdf":
            text = extract_text_pdf(source_data)
        else:
            text = extract_text_url(source)

        if not text:
            return None

        progress.progress(0.20, text=f"📄 Tekst gelezen ({len(text):,} tekens)".replace(",", "."))

        # 2. Vertaling (20% → 35%)
        if auto_translate:
            try:
                lang = detect(text)
            except LangDetectException:
                lang = "nl"
            if lang != "nl":
                progress.progress(0.25, text=f"🌍 Vertalen vanuit {lang.upper()} naar Nederlands...")
                translated = translate_to_dutch(text)
                if translated:
                    text = translated
                progress.progress(0.35, text="🌍 Vertaling klaar.")

        # 3. Titel (35% → 40%)
        progress.progress(0.37, text="📝 Titel genereren...")
        title = generate_title(text)
        progress.progress(0.40, text=f"📝 Titel: {title[:60]}")

        # 4. TTS (40% → 100%)
        client = get_openai_client()
        if not client:
            st.error("OpenAI API-sleutel ontbreekt.")
            return None

        audio_bytes = text_to_audio(
            client, text, voice, progress, base=0.40, span=0.60,
        )
        if not audio_bytes:
            return None

        progress.progress(1.0, text="✅ Klaar!")

        return {
            "title":       title,
            "source":      source,
            "source_type": source_type,
            "voice":       voice,
            "chars":       len(text),
            "audio_bytes": audio_bytes,
        }
    finally:
        progress.empty()


# ── Sidebar bibliotheek ───────────────────────────────────────────────────────

def source_icon(entry: dict) -> str:
    """Bepaal pictogram op basis van source_type (met fallback voor oude entries)."""
    st_type = entry.get("source_type")
    if st_type == "pdf":
        return "📄"
    if st_type == "url":
        return "🔗"
    src = entry.get("source", "")
    return "🔗" if src.startswith(("http://", "https://")) else "📄"


def render_sidebar():
    library = load_library()
    st.sidebar.header("📚 Bibliotheek")
    n = len(library)
    st.sidebar.caption(f"{n} artikel{'en' if n != 1 else ''} opgeslagen")

    if not library:
        st.sidebar.info("Nog geen artikelen opgeslagen.")
        return

    for entry in library:
        with st.sidebar.container(border=True):
            icon = source_icon(entry)
            st.markdown(f"{icon} **{entry['title']}**")

            voice = entry.get("voice", "?")
            chars = entry.get("chars", 0)
            st.caption(
                f"{entry['date']} · 🎤 {voice} · "
                f"{chars:,} tekens".replace(",", ".")
            )

            src = entry.get("source", "")
            if src:
                src_display = src if len(src) <= 50 else src[:47] + "..."
                st.caption(f"📍 {src_display}")

            audio_bytes = load_audio(entry["id"])
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mpeg")

            cols = st.columns([3, 1])
            with cols[0]:
                if audio_bytes:
                    title_safe = re.sub(r"[^\w\-]", "_", entry["title"])[:50]
                    st.download_button(
                        "⬇️ MP3",
                        data=audio_bytes,
                        file_name=f"{title_safe}.mp3",
                        mime="audio/mpeg",
                        key=f"dl_{entry['id']}",
                        use_container_width=True,
                    )
            with cols[1]:
                if st.button(
                    "🗑", key=f"del_{entry['id']}",
                    use_container_width=True, help="Verwijderen",
                ):
                    delete_from_library(entry["id"])
                    st.rerun()


# ── Hoofdscherm ───────────────────────────────────────────────────────────────

def render_main():
    st.title("🎧 Voorlezen")
    st.caption("Upload een PDF of voer een URL in en laat het direct voorlezen.")

    # Instellingen
    cols = st.columns([2, 3])
    with cols[0]:
        voice = st.selectbox("Stem", VOICE_OPTIONS, index=4)
    with cols[1]:
        st.write("")
        st.write("")
        auto_translate = st.checkbox(
            "Vertaal automatisch naar Nederlands",
            value=True,
            help="Niet-Nederlandse tekst wordt automatisch vertaald via Claude Sonnet.",
        )

    st.divider()

    # Input
    tab_pdf, tab_url = st.tabs(["📄 PDF upload", "🔗 URL invoer"])

    source_type = None
    source      = None
    source_data = None

    with tab_pdf:
        uploaded = st.file_uploader(
            "Kies een PDF", type=["pdf"], label_visibility="collapsed",
        )
        if uploaded:
            source_type = "pdf"
            source      = uploaded.name
            source_data = uploaded
            st.info(f"📄 Klaar om te verwerken: **{uploaded.name}**")

    with tab_url:
        url = st.text_input(
            "URL", placeholder="https://example.com/artikel",
            label_visibility="collapsed",
        )
        if url.strip() and not source_type:
            source_type = "url"
            source      = url.strip()
            source_data = None
            preview = source if len(source) <= 60 else source[:57] + "..."
            st.info(f"🔗 Klaar om te verwerken: **{preview}**")

    ready = source_type is not None

    if st.button(
        "▶️  Voorlezen",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    ) and ready:
        st.session_state.pop("pending_result", None)
        result = process_to_audio(source_type, source, source_data, voice, auto_translate)
        if result:
            st.session_state["pending_result"] = result

    # Resultaat
    pending = st.session_state.get("pending_result")
    if not pending:
        return

    st.divider()
    icon = "📄" if pending["source_type"] == "pdf" else "🔗"
    st.subheader(f"{icon} {pending['title']}")
    st.caption(
        f"🎤 {pending['voice']} · "
        f"{pending['chars']:,} tekens".replace(",", ".")
    )

    st.audio(pending["audio_bytes"], format="audio/mpeg")

    st.divider()

    cols = st.columns([3, 2])
    with cols[0]:
        title_input = st.text_input(
            "Titel voor in de bibliotheek",
            value=pending["title"],
        )
        if st.button("💾 Opslaan in bibliotheek", type="primary", use_container_width=True):
            if not title_input.strip():
                st.error("Geef een titel op.")
            else:
                add_to_library(
                    title       = title_input.strip(),
                    source_type = pending["source_type"],
                    source      = pending["source"],
                    voice       = pending["voice"],
                    chars       = pending["chars"],
                    audio_bytes = pending["audio_bytes"],
                )
                st.session_state.pop("pending_result", None)
                st.success(f"Opgeslagen: '{title_input.strip()}'")
                st.rerun()

    with cols[1]:
        title_safe = re.sub(r"[^\w\-]", "_", pending["title"])[:50]
        st.write("")
        st.write("")
        st.download_button(
            "⬇️ Download MP3",
            data=pending["audio_bytes"],
            file_name=f"{title_safe}_{pending['voice']}.mp3",
            mime="audio/mpeg",
            use_container_width=True,
        )


def main():
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
