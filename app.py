# requirements: streamlit, pdfplumber, trafilatura, openai, langdetect, anthropic

import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path

import anthropic
import pdfplumber
import streamlit as st
import trafilatura
from langdetect import LangDetectException, detect
from openai import OpenAI

# ── Opslagmappen ──────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).parent / "data"
AUDIO_DIR = DATA_DIR / "audio"
TEXTS_FILE = DATA_DIR / "texts.json"

DATA_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)


# ── Bibliotheekfuncties (teksten) ─────────────────────────────────────────────

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


def add_to_library(title: str, source: str, text: str) -> str:
    library = load_library()
    entry_id = hashlib.md5(text.encode()).hexdigest()[:10]
    # Verwijder bestaand item met zelfde id
    library = [e for e in library if e["id"] != entry_id]
    library.insert(0, {
        "id":     entry_id,
        "title":  title,
        "source": source,
        "date":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "text":   text,
    })
    save_library(library)
    return entry_id


def delete_from_library(entry_id: str) -> None:
    library = [e for e in load_library() if e["id"] != entry_id]
    save_library(library)
    # Verwijder eventuele audio
    for audio_file in AUDIO_DIR.glob(f"{entry_id}_*.mp3"):
        audio_file.unlink(missing_ok=True)


# ── Audiocache ────────────────────────────────────────────────────────────────

def audio_cache_path(entry_id: str, voice: str) -> Path:
    return AUDIO_DIR / f"{entry_id}_{voice}.mp3"


def load_cached_audio(entry_id: str, voice: str) -> bytes | None:
    path = audio_cache_path(entry_id, voice)
    if path.exists():
        return path.read_bytes()
    return None


def save_cached_audio(entry_id: str, voice: str, audio_bytes: bytes) -> None:
    audio_cache_path(entry_id, voice).write_bytes(audio_bytes)


# ── API-clients ───────────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API-sleutel ontbreekt. Voeg OPENAI_API_KEY toe aan st.secrets of als omgevingsvariabele.")
        st.stop()
    return OpenAI(api_key=api_key)


def get_anthropic_client() -> anthropic.Anthropic:
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Anthropic API-sleutel ontbreekt. Voeg ANTHROPIC_API_KEY toe aan st.secrets of als omgevingsvariabele.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# ── Tekstextractie ────────────────────────────────────────────────────────────

def extract_text_pdf(uploaded_file) -> str:
    text_blocks = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_height   = page.height
                top_margin    = page_height * 0.10
                bottom_margin = page_height * 0.90
                cropped   = page.within_bbox((0, top_margin, page.width, bottom_margin))
                page_text = cropped.extract_text()
                if page_text:
                    text_blocks.append(page_text.strip())
    except Exception as e:
        st.error(f"PDF-extractie mislukt: {e}")
        return ""
    return "\n\n".join(text_blocks)


def extract_text_url(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            st.error("URL kon niet worden opgehaald. Controleer de URL en probeer opnieuw.")
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
    try:
        client = get_anthropic_client()
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


# ── TTS ───────────────────────────────────────────────────────────────────────

def split_text(text: str, max_chars: int = 4096) -> list[str]:
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


def text_to_audio(client: OpenAI, text: str, voice: str) -> bytes:
    chunks = split_text(text)
    if len(chunks) == 1:
        try:
            return client.audio.speech.create(model="tts-1", voice=voice, input=chunks[0]).content
        except Exception as e:
            st.error(f"TTS API-fout: {e}")
            return b""

    audio_parts = []
    progress = st.progress(0, text="Audio genereren...")
    for i, chunk in enumerate(chunks):
        try:
            audio_parts.append(
                client.audio.speech.create(model="tts-1", voice=voice, input=chunk).content
            )
        except Exception as e:
            st.error(f"TTS API-fout bij chunk {i + 1}: {e}")
            return b""
        progress.progress((i + 1) / len(chunks), text=f"Chunk {i + 1}/{len(chunks)} verwerkt...")
    progress.empty()
    return b"".join(audio_parts)


# ── Hoofdapp ──────────────────────────────────────────────────────────────────

def main():
    st.title("Voorlezen")
    st.caption("Upload een PDF of voer een URL in om tekst voor te laten lezen.")

    # ── Tabbladen: nieuw / bibliotheek ────────────────────────────────────────
    tab_nieuw, tab_bibliotheek = st.tabs(["Nieuw artikel", "Bibliotheek"])

    # ── Tab: Nieuw artikel ────────────────────────────────────────────────────
    with tab_nieuw:
        sub_pdf, sub_url = st.tabs(["PDF upload", "URL invoer"])

        with sub_pdf:
            uploaded_file = st.file_uploader("Kies een PDF-bestand", type=["pdf"])
            if uploaded_file is not None:
                if st.button("Tekst extraheren uit PDF"):
                    with st.spinner("PDF verwerken..."):
                        tekst = extract_text_pdf(uploaded_file)
                    if tekst:
                        st.session_state["extracted_text"]  = tekst
                        st.session_state["current_source"]  = uploaded_file.name
                        st.session_state["current_entry_id"] = None
                        st.success(f"Tekst geëxtraheerd ({len(tekst)} tekens).")
                    else:
                        st.error("Geen tekst gevonden in de PDF.")

        with sub_url:
            url_input = st.text_input("Voer een URL in", placeholder="https://example.com/artikel")
            if st.button("Tekst extraheren uit URL"):
                if not url_input.strip():
                    st.error("Voer een geldige URL in.")
                else:
                    with st.spinner("URL ophalen en verwerken..."):
                        tekst = extract_text_url(url_input.strip())
                    if tekst:
                        st.session_state["extracted_text"]   = tekst
                        st.session_state["current_source"]   = url_input.strip()
                        st.session_state["current_entry_id"] = None
                        st.success(f"Tekst geëxtraheerd ({len(tekst)} tekens).")

    # ── Tab: Bibliotheek ──────────────────────────────────────────────────────
    with tab_bibliotheek:
        library = load_library()
        if not library:
            st.info("De bibliotheek is nog leeg. Sla eerst een artikel op.")
        else:
            labels = [f"{e['date']}  —  {e['title']}" for e in library]
            keuze  = st.selectbox("Kies een opgeslagen artikel", labels)
            idx    = labels.index(keuze)
            entry  = library[idx]

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Laad artikel", use_container_width=True):
                    st.session_state["extracted_text"]   = entry["text"]
                    st.session_state["current_source"]   = entry["source"]
                    st.session_state["current_entry_id"] = entry["id"]
                    st.success(f"'{entry['title']}' geladen.")
                    st.rerun()
            with col2:
                if st.button("Verwijder artikel", use_container_width=True):
                    delete_from_library(entry["id"])
                    if st.session_state.get("current_entry_id") == entry["id"]:
                        st.session_state.pop("extracted_text", None)
                        st.session_state.pop("current_entry_id", None)
                    st.success(f"'{entry['title']}' verwijderd.")
                    st.rerun()

    st.divider()

    # ── Tekstgebied ───────────────────────────────────────────────────────────
    current_text  = st.session_state.get("extracted_text", "")
    editable_text = st.text_area(
        "Geëxtraheerde tekst (bewerkbaar)",
        value=current_text,
        height=300,
        placeholder="Tekst verschijnt hier na extractie. U kunt de tekst aanpassen voor het voorlezen.",
    )

    # ── Opslaan in bibliotheek ────────────────────────────────────────────────
    if editable_text.strip():
        with st.expander("Opslaan in bibliotheek"):
            default_title = st.session_state.get("current_source", "")[:60]
            save_title = st.text_input("Titel", value=default_title)
            if st.button("Opslaan"):
                if not save_title.strip():
                    st.error("Geef een titel op.")
                else:
                    entry_id = add_to_library(
                        title=save_title.strip(),
                        source=st.session_state.get("current_source", ""),
                        text=editable_text.strip(),
                    )
                    st.session_state["current_entry_id"] = entry_id
                    st.success(f"Opgeslagen als '{save_title.strip()}'.")

    st.divider()

    # ── Taaldetectie & vertaling ──────────────────────────────────────────────
    current_editable = editable_text.strip()
    detected_lang    = ""
    if current_editable:
        try:
            detected_lang = detect(current_editable)
        except LangDetectException:
            detected_lang = ""

    if detected_lang and detected_lang != "nl":
        if st.button(f"Vertaal naar Nederlands ({detected_lang.upper()} gedetecteerd)"):
            with st.spinner("Tekst vertalen met Claude Sonnet..."):
                translated = translate_to_dutch(current_editable)
            if translated:
                st.session_state["extracted_text"]   = translated
                st.session_state["current_entry_id"] = None
                st.success("Tekst vertaald naar Nederlands.")
                st.rerun()

    st.divider()

    # ── Stemkeuze & voorlezen ─────────────────────────────────────────────────
    voice_options  = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    selected_voice = st.selectbox("Stem", options=voice_options, index=4)

    if st.button("Voorlezen", type="primary"):
        text_to_read = editable_text.strip()
        if not text_to_read:
            st.error("Er is geen tekst om voor te lezen. Extraheer eerst tekst via PDF of URL.")
        else:
            entry_id = st.session_state.get("current_entry_id")

            # Controleer audiocache
            cached = load_cached_audio(entry_id, selected_voice) if entry_id else None

            if cached:
                st.info("Audio geladen uit lokale cache (geen API-kosten).")
                audio_bytes = cached
            else:
                try:
                    lang = detect(text_to_read)
                except LangDetectException:
                    lang = "onbekend"

                openai_client = get_openai_client()
                with st.spinner(f"Audio genereren (taal: {lang})..."):
                    audio_bytes = text_to_audio(openai_client, text_to_read, selected_voice)

                # Sla op in cache als het artikel in de bibliotheek staat
                if audio_bytes and entry_id:
                    save_cached_audio(entry_id, selected_voice, audio_bytes)
                    st.success("Audio opgeslagen — volgende keer gratis afspelen.")

            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

                # Downloadknop
                st.download_button(
                    label="Download MP3",
                    data=audio_bytes,
                    file_name=f"{entry_id or 'audio'}_{selected_voice}.mp3",
                    mime="audio/mpeg",
                )
            else:
                st.error("Audio kon niet worden gegenereerd.")


if __name__ == "__main__":
    main()
