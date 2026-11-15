# requirements: streamlit, pdfplumber, trafilatura, openai, langdetect, anthropic

import hashlib
import json
import os
import re
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import anthropic
import openai
import pdfplumber
import streamlit as st
import trafilatura
from langdetect import DetectorFactory, LangDetectException, detect
from openai import OpenAI

DetectorFactory.seed = 0  # zonder seed kan langdetect per run een andere taal kiezen

# ── Configuratie ──────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent / "data"
AUDIO_DIR    = DATA_DIR / "audio"
LIBRARY_FILE = DATA_DIR / "texts.json"

DATA_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

MAX_PDF_SIZE_MB = 25

CLAUDE_MODEL   = "claude-sonnet-4-6"  # vertaling, titel en nood-extractie uit HTML
TTS_MODEL      = "tts-1"
TTS_HARD_LIMIT = 4096  # maximum aantal tekens per OpenAI TTS-aanvraag

# De tekst wordt in stukken geknipt die tegelijk vertaald en ingesproken worden.
# De wachttijd hangt daardoor af van het traagste stuk, niet van de totale
# lengte. 2000 tekens houdt elk stuk snel en laat ruimte voor een vertaling die
# langer uitvalt dan het origineel (die moet onder TTS_HARD_LIMIT blijven).
CHUNK_CHARS = 2000
MAX_WORKERS = 16

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
    # Eerst naar een tijdelijk bestand: een onderbroken schrijfactie laat dan
    # de bestaande bibliotheek heel.
    tmp = LIBRARY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(LIBRARY_FILE)


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


def audio_path(entry_id: str) -> Path | None:
    """Pad naar de MP3 van een artikel, of None als die ontbreekt."""
    path = AUDIO_DIR / f"{entry_id}.mp3"
    if path.exists():
        return path
    # Backward compatibility: oude bestandsnamen
    return next(AUDIO_DIR.glob(f"{entry_id}_*.mp3"), None)


def load_audio(entry_id: str) -> bytes | None:
    path = audio_path(entry_id)
    return path.read_bytes() if path else None


def get_library_entry(entry_id: str) -> dict | None:
    for entry in load_library():
        if entry.get("id") == entry_id:
            return entry
    return None


# ── API-clients ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_openai_client() -> OpenAI | None:
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY")
    # max_retries=4: OpenAI SDK doet retries op 429/5xx/connection errors
    return OpenAI(api_key=api_key, max_retries=4, timeout=90.0) if api_key else None


@st.cache_resource
def get_anthropic_client() -> anthropic.Anthropic | None:
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=api_key, max_retries=4) if api_key else None


def _reply_text(response) -> str:
    """Alle tekstblokken uit een Claude-antwoord, aan elkaar."""
    return "".join(b.text for b in response.content if b.type == "text").strip()


# ── Tekstextractie ────────────────────────────────────────────────────────────

def clean_pdf_text(text: str) -> str:
    """Herstel regels die de PDF-opmaak midden in een zin afbreekt.

    Zonder dit leest de stem "voor- lezen" en pauzeert hij op elk regeleinde.
    Regels die met een hoofdletter beginnen (koppen, nieuwe zinnen) blijven staan.
    """
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Afbreekstreepje aan het regeleinde: "voor-\nlezen" → "voorlezen"
    text = re.sub(r"(?<=[a-zà-ÿ])-\n+(?=[a-zà-ÿ])", "", text)
    # Regel eindigt zonder leesteken en gaat klein verder → dezelfde zin
    text = re.sub(r"(?<![.!?:;…\n])\n+(?=[a-zà-ÿ])", " ", text)
    return text.strip()


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

    text = clean_pdf_text("\n\n".join(text_blocks))
    if not text:
        st.error(
            "Geen tekst gevonden in deze PDF. Waarschijnlijk is het een scan "
            "(afbeeldingen zonder tekstlaag)."
        )
    return text


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
        req = urllib.request.Request(url, headers=BROWSER_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def extract_jsonld(html: str) -> str:
    """Zoek de artikeltekst in Schema.org JSON-LD (veld `articleBody`).

    Bewust niet `description`: dat is een samenvatting van een paar regels, en
    die zou dan in plaats van het artikel worden voorgelezen.
    """
    pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    matches = re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)
    candidates = []
    for match in matches:
        try:
            data = json.loads(match.strip())
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        # Voeg @graph-items toe (veelgebruikt patroon)
        for item in list(items):
            if isinstance(item, dict) and isinstance(item.get("@graph"), list):
                items.extend(item["@graph"])
        for item in items:
            if not isinstance(item, dict):
                continue
            body = item.get("articleBody")
            if body and isinstance(body, str):
                candidates.append(body.strip())
    return max(candidates, key=len) if candidates else ""


def extract_with_claude(html: str, notes: list[str]) -> str:
    """Laatste redmiddel: Claude extraheert hoofdtekst uit ruwe HTML."""
    client = get_anthropic_client()
    if not client:
        return ""

    # Strip scripts en stijl om tokens te besparen
    cleaned = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style[^>]*>.*?</style>",   "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<!--.*?-->",                "", cleaned, flags=re.DOTALL)
    if len(cleaned) > 80_000:
        cleaned = cleaned[:80_000]

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=(
                "Je krijgt ruwe HTML van een webpagina. Extraheer ALLEEN de "
                "hoofdtekst van het artikel of nieuwsbericht. Negeer menu's, "
                "advertenties, footers, comments, gerelateerde artikelen, "
                "navigatielinks en auteur/datum-info. Geef alleen de leesbare "
                "tekst terug, zonder HTML-tags en zonder toelichting. "
                "Als er geen duidelijk artikel is, antwoord met 'GEEN_ARTIKEL'."
            ),
            messages=[{"role": "user", "content": cleaned}],
        )
    except Exception:
        return ""

    result = _reply_text(response)
    if result == "GEEN_ARTIKEL" or len(result) < 100:
        return ""
    if response.stop_reason == "max_tokens":
        notes.append("Het artikel was te lang om volledig op te halen: het einde ontbreekt.")
    return result


def extract_text_url(url: str, notes: list[str]) -> str:
    if not is_valid_url(url):
        st.error("Ongeldige URL. Gebruik http:// of https://")
        return ""

    html = fetch_html(url)
    if not html:
        st.error("URL kon niet worden opgehaald.")
        return ""

    # Strategie 1: trafilatura met verschillende settings
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

    # Strategie 2: JSON-LD (Schema.org Article)
    jsonld_text = extract_jsonld(html)
    if len(jsonld_text) > 100:
        return jsonld_text

    # Strategie 3: Claude als laatste redmiddel
    st.info("⚙️ Standaard extractie lukte niet — Claude probeert het nu...")
    claude_text = extract_with_claude(html, notes)
    if claude_text:
        return claude_text

    st.error(
        "Geen hoofdtekst gevonden. De pagina gebruikt mogelijk JavaScript "
        "om de tekst te tonen, of blokkeert geautomatiseerde toegang."
    )
    return ""


# ── Tekst in stukken knippen ──────────────────────────────────────────────────

def _wrap_words(text: str, max_chars: int):
    """Knip één te lange zin tussen woorden in delen van hooguit max_chars."""
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        yield text[:cut]
        text = text[cut:].lstrip()
    if text:
        yield text


def split_text(text: str, max_chars: int) -> list[str]:
    """Verdeel tekst in stukken van hooguit max_chars.

    Knipt bij voorkeur tussen alinea's, anders tussen zinnen (leesteken gevolgd
    door witruimte, zodat getallen als "4.7" heel blijven) en pas als één zin
    te lang is tussen woorden.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    # (scheiding ervoor, tekst): alinea's houden hun regelovergang, zinnen uit
    # dezelfde alinea een spatie.
    pieces: list[tuple[str, str]] = []
    sep = ""
    for paragraph, newlines in re.findall(r"([^\n]+)(\n*)", text):
        paragraph = paragraph.strip()
        if len(paragraph) <= max_chars:
            if paragraph:
                pieces.append((sep, paragraph))
        else:
            inner_sep = sep
            for sentence in re.split(r"(?<=[.!?…])\s+", paragraph):
                for part in _wrap_words(sentence, max_chars):
                    pieces.append((inner_sep, part))
                    inner_sep = " "
        sep = "\n\n" if len(newlines) > 1 or not paragraph else "\n"

    chunks, current = [], ""
    for sep, piece in pieces:
        if current and len(current) + len(sep) + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = current + sep + piece if current else piece
    if current:
        chunks.append(current)
    return chunks


# ── Vertaling & titel ─────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Taalcode van de tekst ('nl', 'en', ...), of '' als die niet te bepalen is."""
    try:
        return detect(text)
    except LangDetectException:
        return ""


def translate_chunk(client: anthropic.Anthropic, text: str) -> str:
    """Vertaal één stuk tekst naar het Nederlands. Gooit een fout als dat niet lukt."""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=(
            "Je bent een professionele vertaler. Je krijgt een fragment uit een "
            "langere tekst. Vertaal het naar correct en natuurlijk Nederlands. "
            "Behoud de alinea-indeling. Geef alleen de vertaling terug, zonder "
            "toelichting of extra tekst. Is het fragment al Nederlands, geef het "
            "dan ongewijzigd terug."
        ),
        messages=[{"role": "user", "content": text}],
    )
    translated = _reply_text(response)
    if response.stop_reason != "end_turn" or not translated:
        raise RuntimeError(f"vertaling onvolledig ({response.stop_reason})")
    return translated


def _fallback_title(text: str) -> str:
    return text.strip().split("\n")[0][:80] or "Onbenoemd artikel"


def generate_title(client: anthropic.Anthropic | None, text: str) -> str:
    """Korte Nederlandse titel via Claude (fallback: eerste regel van de tekst)."""
    if not client:
        return _fallback_title(text)
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=100,
            system=(
                "Je krijgt het begin van een artikel of document. Geef één korte, "
                "beschrijvende Nederlandse titel terug (max 70 tekens) die de "
                "inhoud samenvat. Geef alleen de titel zelf, zonder "
                "aanhalingstekens of toelichting."
            ),
            messages=[{"role": "user", "content": text.strip()[:2000]}],
        )
        return _reply_text(response)[:80] or _fallback_title(text)
    except Exception:
        return _fallback_title(text)


# ── TTS ───────────────────────────────────────────────────────────────────────

TTS_MAX_ATTEMPTS = 4  # extra bovenop de retries die de OpenAI-SDK zelf al doet


def _tts_chunk(client: OpenAI, voice: str, chunk: str) -> bytes:
    """TTS voor één chunk, met eigen retry+backoff bovenop de SDK-retries.

    OpenAI's tts-1 geeft af en toe een tijdelijke 500 (server_error) terug.
    De SDK retryt zelf al op 429/5xx, maar bij aanhoudende drukte is een
    extra, tragere retry-laag nodig om chunks niet voortijdig te laten falen.
    """
    last_error: Exception | None = None
    for attempt in range(TTS_MAX_ATTEMPTS):
        try:
            return client.audio.speech.create(
                model=TTS_MODEL, voice=voice, input=chunk
            ).content
        except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as e:
            last_error = e
            is_retryable = (
                isinstance(e, (openai.APIConnectionError, openai.APITimeoutError))
                or getattr(e, "status_code", 0) >= 500
                or getattr(e, "status_code", 0) == 429
            )
            if not is_retryable or attempt == TTS_MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise last_error  # onbereikbaar, maar houdt type-checkers tevreden


def _convert_chunk(chunk: str, voice: str, tts_client: OpenAI,
                   translator: anthropic.Anthropic | None) -> tuple[bytes, int, str | None]:
    """Vertaal (als er een `translator` is) en spreek één stuk tekst in.

    Draait in een werkthread en doet daarom geen Streamlit-aanroepen. Geeft
    (audio, aantal ingesproken tekens, vertaalfout) terug. Lukt de vertaling
    niet, dan wordt het stuk onvertaald ingesproken en staat de reden in het
    derde veld. Een TTS-fout wordt doorgegooid.
    """
    translate_error = None
    if translator:
        try:
            chunk = translate_chunk(translator, chunk)
        except Exception as e:
            translate_error = str(e)

    audio = b"".join(
        _tts_chunk(tts_client, voice, part)
        for part in split_text(chunk, TTS_HARD_LIMIT)
    )
    return audio, len(chunk), translate_error


# ── Hoofdpipeline ─────────────────────────────────────────────────────────────

def process_to_audio(source_type: str, source: str, source_data,
                     voice: str, auto_translate: bool) -> dict | None:
    """Tekst ophalen, daarna per stuk tegelijk vertalen en inspreken."""
    progress = st.progress(0.0, text="📄 Tekst ophalen...")
    notes: list[str] = []  # meldingen die na afloop bij het resultaat horen

    try:
        if source_type == "pdf":
            text = extract_text_pdf(source_data)
        elif source_type == "url":
            text = extract_text_url(source, notes)
        else:
            text = (source_data or "").strip()
            if not text:
                st.error("Geen tekst ingevoerd.")
        if not text:
            return None

        tts_client = get_openai_client()
        if not tts_client:
            st.error("OpenAI API-sleutel ontbreekt.")
            return None
        claude = get_anthropic_client()

        lang = detect_language(text)
        translate = auto_translate and lang not in ("", "nl")
        if translate and not claude:
            notes.append("Niet vertaald: de Anthropic API-sleutel ontbreekt.")
            translate = False

        chunks = split_text(text, CHUNK_CHARS)
        total  = len(chunks)
        action = f"Vertalen ({lang.upper()} → NL) en inspreken" if translate else "Inspreken"
        progress.progress(0.10, text=f"🎤 {action}... (0/{total} delen klaar)")

        audio_parts: list[bytes] = [b""] * total
        spoken_chars = 0
        translate_errors: list[str] = []

        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        try:
            # De titel komt uit het begin van de brontekst en hoeft dus niet
            # op de vertaling te wachten.
            title_future = executor.submit(generate_title, claude, text)
            futures = {
                executor.submit(
                    _convert_chunk, chunk, voice, tts_client, claude if translate else None
                ): i
                for i, chunk in enumerate(chunks)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                try:
                    audio, chars, translate_error = future.result()
                except Exception as e:
                    st.error(
                        f"Audio genereren mislukt bij deel {idx + 1} van {total}, "
                        f"ook na herhaalde pogingen: {e}"
                    )
                    return None
                audio_parts[idx] = audio
                spoken_chars += chars
                if translate_error:
                    translate_errors.append(translate_error)
                progress.progress(
                    0.10 + 0.90 * done / total,
                    text=f"🎤 {action}... ({done}/{total} delen klaar)",
                )
            title = title_future.result()
        finally:
            # Na een fout: stukken die nog in de wachtrij staan niet meer
            # uitvoeren (en dus niet betalen).
            executor.shutdown(wait=False, cancel_futures=True)

        if translate_errors:
            notes.append(
                f"{len(translate_errors)} van de {total} delen konden niet vertaald "
                f"worden en zijn in de oorspronkelijke taal voorgelezen "
                f"({translate_errors[0]})."
            )

        return {
            "title":       title,
            "source":      source,
            "source_type": source_type,
            "voice":       voice,
            "chars":       spoken_chars,
            "audio_bytes": b"".join(audio_parts),
            "notes":       notes,
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
    if st_type == "text":
        return "✍️"
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

            # Alleen kijken of het bestand er is: de MP3 zelf wordt pas
            # ingelezen als het artikel wordt afgespeeld.
            has_audio = audio_path(entry["id"]) is not None
            if not has_audio:
                st.warning("MP3-bestand niet gevonden.")

            cols = st.columns([3, 1])
            with cols[0]:
                if st.button(
                    "Voorlezen",
                    key=f"play_saved_{entry['id']}",
                    disabled=not has_audio,
                    use_container_width=True,
                ):
                    st.session_state["active_library_entry_id"] = entry["id"]
                    st.session_state.pop("pending_result", None)
                    st.session_state.pop("show_pending_audio", None)
                    st.rerun()
            with cols[1]:
                if st.button(
                    "🗑", key=f"del_{entry['id']}",
                    use_container_width=True, help="Verwijderen",
                ):
                    delete_from_library(entry["id"])
                    if st.session_state.get("active_library_entry_id") == entry["id"]:
                        st.session_state.pop("active_library_entry_id", None)
                    st.rerun()


def render_saved_player():
    entry_id = st.session_state.get("active_library_entry_id")
    if not entry_id:
        return False

    entry = get_library_entry(entry_id)
    audio_bytes = load_audio(entry_id)
    if not entry or not audio_bytes:
        st.session_state.pop("active_library_entry_id", None)
        return False

    icon = source_icon(entry)
    st.subheader(f"{icon} {entry['title']}")
    st.caption(
        f"Opgeslagen op {entry.get('date', '?')} · "
        f"🎤 {entry.get('voice', '?')} · "
        f"{entry.get('chars', 0):,} tekens".replace(",", ".")
    )

    source = entry.get("source", "")
    if source:
        st.caption(f"Bron: {source}")

    st.audio(audio_bytes, format="audio/mpeg")

    cols = st.columns([2, 1])
    with cols[0]:
        title_safe = re.sub(r"[^\w\-]", "_", entry["title"])[:50]
        st.download_button(
            "⬇️ Download MP3",
            data=audio_bytes,
            file_name=f"{title_safe}.mp3",
            mime="audio/mpeg",
            use_container_width=True,
            key=f"main_dl_{entry_id}",
        )
    with cols[1]:
        if st.button("Sluiten", use_container_width=True):
            st.session_state.pop("active_library_entry_id", None)
            st.rerun()

    st.divider()
    return True


# ── Hoofdscherm ───────────────────────────────────────────────────────────────

def input_signature(source_type: str, source: str, source_data,
                    voice: str, auto_translate: bool) -> str:
    """Kenmerk van invoer + instellingen. Verandert er iets, dan hoort een
    eerder resultaat er niet meer bij en is een nieuwe conversie nodig."""
    if source_type == "pdf":
        content = f"{source_data.size}:{getattr(source_data, 'file_id', '')}"
    elif source_type == "text":
        content = hashlib.md5(source_data.strip().encode("utf-8")).hexdigest()
    else:
        content = ""
    return f"{source_type}:{source}:{content}:{voice}:{auto_translate}"


def render_main():
    st.title("🎧 Voorlezen")
    st.caption("Upload een PDF, voer een URL in of plak tekst, en laat het direct voorlezen.")

    # Melding van vóór de laatste st.rerun() (die wist alles wat op het scherm stond)
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    render_saved_player()

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
    tab_pdf, tab_url, tab_text = st.tabs(
        ["📄 PDF upload", "🔗 URL invoer", "✍️ Tekst invoer"]
    )

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

    with tab_text:
        pasted = st.text_area(
            "Plak of typ hier de tekst",
            height=200,
            placeholder="Plak hier de tekst die je wilt laten voorlezen...",
            label_visibility="collapsed",
            key="pasted_text",
        )
        if pasted.strip() and not source_type:
            source_type = "text"
            source      = "Ingevoerde tekst"
            source_data = pasted
            chars = len(pasted.strip())
            st.info(f"✍️ Klaar om te verwerken: **{chars:,} tekens ingevoerd**".replace(",", "."))

    ready = source_type is not None
    current_input_signature = (
        input_signature(source_type, source, source_data, voice, auto_translate)
        if ready else None
    )
    pending = st.session_state.get("pending_result")

    if pending and pending.get("input_signature") != current_input_signature:
        st.session_state.pop("pending_result", None)
        st.session_state.pop("show_pending_audio", None)
        pending = None

    primary_label = "Voorlezen" if pending else "Start conversie"

    if st.button(
        primary_label,
        type="primary",
        use_container_width=True,
        disabled=not ready and not pending,
    ):
        if pending:
            st.session_state["show_pending_audio"] = True
        elif ready:
            st.session_state.pop("pending_result", None)
            st.session_state.pop("show_pending_audio", None)
            st.session_state.pop("active_library_entry_id", None)
            result = process_to_audio(source_type, source, source_data, voice, auto_translate)
            if result:
                result["input_signature"] = current_input_signature
                st.session_state["pending_result"] = result
                st.rerun()

    # Resultaat
    pending = st.session_state.get("pending_result")
    if not pending:
        return

    st.divider()
    icons = {"pdf": "📄", "url": "🔗", "text": "✍️"}
    icon  = icons.get(pending["source_type"], "📄")
    st.subheader(f"{icon} {pending['title']}")
    st.caption(
        f"🎤 {pending['voice']} · "
        f"{pending['chars']:,} tekens".replace(",", ".")
    )
    for note in pending.get("notes", []):
        st.warning(note)

    if st.session_state.get("show_pending_audio"):
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
                st.session_state.pop("show_pending_audio", None)
                st.session_state["flash"] = f"Opgeslagen in de bibliotheek: '{title_input.strip()}'"
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
