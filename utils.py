"""
utils.py — Pure logic / utility functions.

Zero Streamlit dependency.  Everything here is testable with plain pytest.
"""

import re
import numpy as np
from urllib.parse import parse_qs, urlparse
from pytubefix import YouTube

from config import Document


# ── YouTube URL helpers ──────────────────────────────────────────────────────

def extract_video_id(url):
    """Extract an 11-character YouTube video ID from any common URL format."""
    if not url:
        return None
    normalized = url.strip()
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", normalized):
        return normalized

    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    allowed_hosts = {
        "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
        "youtu.be", "www.youtu.be",
    }
    if host not in allowed_hosts:
        return None

    video_id = None
    if "youtu.be" in host:
        video_id = parsed.path.lstrip("/").split("/")[0]
    else:
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id:
            video_id = query_id
        else:
            path_parts = [part for part in parsed.path.split("/") if part]
            if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
                video_id = path_parts[1]

    return video_id if video_id and re.fullmatch(r"[0-9A-Za-z_-]{11}", video_id) else None


def format_timestamp(seconds):
    """Convert seconds → readable timestamp like '1:05:30' or '5:30'."""
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"


# ── Transcript fetching & processing ────────────────────────────────────────

def get_video_info_and_transcript(video_id):
    """
    Fetch video metadata and captions via pytubefix.

    Returns:
        (meta_dict, transcript_list_or_None, error_string_or_None)
    """
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        meta = {
            "title": yt.title or "Unknown Title",
            "author": yt.author or "Unknown Channel",
            "description": (yt.description or "")[:1000],
        }
        caption = None
        for lang_code in ["en", "a.en"]:
            caption = yt.captions.get(lang_code)
            if caption:
                break
        if not caption and yt.captions:
            caption = list(yt.captions.values())[0]
        if not caption:
            return meta, None, "No captions found on this video."

        raw_srt = caption.generate_srt_captions()
        transcript = []
        for block in raw_srt.strip().split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                time_line = lines[1]
                text = " ".join(lines[2:])
                start_str = time_line.split(" --> ")[0].strip()
                parts = start_str.replace(",", ".").split(":")
                if len(parts) != 3:
                    continue
                try:
                    start_secs = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                except ValueError:
                    continue
                transcript.append({"start": start_secs, "text": text})
        return meta, transcript, None
    except Exception as e:
        return {}, None, str(e)


def process_transcript(raw_transcript, meta, target_chunk_chars=600):
    """
    Convert a raw transcript list into LangChain Documents.

    Creates a special 'identity' document at index 0 containing video metadata
    (title, host, guest) so the LLM can answer 'who is speaking?' questions.
    """
    all_docs = []

    # --- Doc #0: Video Identity ---
    title = meta.get("title", "")
    host = meta.get("author", "Unknown Host")
    guest = "Unknown Guest"
    for pattern in [
        r" with (.+?)(?:\s*[-|,]|$)",
        r'feat(?:uring)?\.?\s+(.+?)(?:\s*[-|,]|$)',
        r":\s*(.+?) on ",
    ]:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            guest = m.group(1).strip()
            break

    identity_text = (
        f"VIDEO TITLE: {title}\n"
        f"HOST / INTERVIEWER: {host}\n"
        f"GUEST / INTERVIEWEE: {guest}\n"
        f"DESCRIPTION: {meta.get('description', '')}\n\n"
        f"IMPORTANT: {host} is the HOST who interviews {guest}."
    )
    all_docs.append(
        Document(page_content=identity_text, metadata={"start": 0.0, "timestamp": "0:00", "type": "metadata"})
    )

    # --- Paragraph-level chunking ---
    current_text, current_start = "", 0.0
    for entry in raw_transcript:
        start_time = entry.get("start", 0.0)
        text_val = entry.get("text", "").strip()
        if not text_val:
            continue
        if not current_text:
            current_start = start_time
        current_text += text_val + " "
        if len(current_text) >= target_chunk_chars:
            all_docs.append(
                Document(
                    page_content=current_text.strip(),
                    metadata={"start": current_start, "timestamp": format_timestamp(current_start)},
                )
            )
            current_text, current_start = "", 0.0
    if current_text.strip():
        all_docs.append(
            Document(
                page_content=current_text.strip(),
                metadata={"start": current_start, "timestamp": format_timestamp(current_start)},
            )
        )
    return all_docs


# ── Relevance scoring ───────────────────────────────────────────────────────

def compute_relevance(question, retrieved_docs, embeddings_model):
    """Cosine similarity between the question and the best retrieved chunk."""
    if not retrieved_docs:
        return 0.0
    try:
        q_vec = np.array(embeddings_model.embed_query(question))
        doc_vecs = np.array(embeddings_model.embed_documents([d.page_content for d in retrieved_docs]))
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        sims = doc_vecs @ q_norm / (np.linalg.norm(doc_vecs, axis=1) + 1e-9)
        return float(np.max(sims))
    except Exception:
        return 0.5


def normalize_doc_metadata(doc):
    """Extract (timestamp_str, start_seconds) from a Document's metadata."""
    start_val = doc.metadata.get("start", 0.0)
    try:
        start_val = float(start_val)
    except (TypeError, ValueError):
        start_val = 0.0
    timestamp_val = doc.metadata.get("timestamp") or format_timestamp(start_val)
    return timestamp_val, start_val


def is_meta_question(prompt, meta_patterns):
    """Return True if the prompt matches any broad/meta question pattern."""
    return any(p in prompt.lower() for p in meta_patterns)
