import streamlit as st
import re
import os
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv
from pytubefix import YouTube
from pytubefix.exceptions import VideoUnavailable
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

# ── Standard LangChain Imports (0.2+) ──
from langchain.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    ParentDocumentRetriever,
)
from langchain_core.stores import InMemoryStore
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
# ───────────────────────────────────────────────────────────────────────────

try:
    from . import audio_utils
except ImportError:
    import audio_utils

load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# --- Page Config ---
st.set_page_config(
    page_title="YT Chatbot V2 - Beast Mode",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0a0b10;
        color: #e2e8f0;
    }
    .main-header {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.25rem;
    }
    .video-card {
        background: rgba(255,255,255,0.04); padding: 16px;
        border-radius: 16px; border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.5rem;
    }
    .video-intelligence-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.1));
        padding: 16px 20px; border-radius: 14px;
        border: 1px solid rgba(99,102,241,0.3); margin-bottom: 1.2rem;
    }
    .confidence-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 700; margin-left: 8px;
    }
    .badge-high   { background: rgba(34,197,94,0.2);  color: #4ade80; border: 1px solid #4ade80; }
    .badge-medium { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid #fbbf24; }
    .badge-low    { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid #f87171; }
    .timestamp-link {
        color: #818cf8; text-decoration: none; font-weight: 600;
        font-size: 0.82rem; background: rgba(99,102,241,0.12);
        padding: 2px 7px; border-radius: 5px; margin-right: 6px;
        transition: background 0.2s;
    }
    .timestamp-link:hover { background: rgba(99,102,241,0.25); }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    explanation_type = st.selectbox(
        "Explanation Style",
        ["Quick Summary", "Moderate Detail", "Deep Dive"],
        index=1
    )
    presets = {
        "Quick Summary":   {"top_k": 3, "ensemble_weight": 0.4},
        "Moderate Detail": {"top_k": 5, "ensemble_weight": 0.6},
        "Deep Dive":       {"top_k": 7, "ensemble_weight": 0.8}
    }
    current_preset = presets[explanation_type]

    st.markdown("---")
    manual_tuning = st.toggle("🛠️ Enable Manual Tuning", value=False)
    if manual_tuning:
        with st.expander("🔬 Manual Engine Tuning", expanded=True):
            child_size = st.slider("Child Chunk Size", 100, 600, 300)
            parent_size = st.slider("Parent Chunk Size", 600, 3000, 1200)
            top_k = st.slider("Top K Retrieved", 2, 10, current_preset["top_k"])
            ensemble_weight = st.slider("Ensemble Weight", 0.0, 1.0, current_preset["ensemble_weight"])
    else:
        child_size, parent_size = 300, 1200
        top_k = current_preset["top_k"]
        ensemble_weight = current_preset["ensemble_weight"]
        st.info(f"✨ **Smart Autopilot**: Optimal for *{explanation_type}*.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Answer Quality Monitor")
    st.caption("Tracks in-scope confidence (blocked queries excluded).")
    graded = [s for s in st.session_state.get("quality_scores", []) if s is not None]
    if graded:
        avg = sum(graded) / len(graded)
        col_a, col_b = st.columns(2)
        col_a.metric("Avg Confidence", f"{avg:.0%}")
        col_b.metric("Answers Graded", len(graded))
        st.progress(min(avg, 1.0))
    else:
        st.caption("Ask a question to start tracking quality.")

# --- Helper Functions ---
def extract_video_id(url):
    if not url:
        return None
    normalized = url.strip()
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", normalized):
        return normalized

    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    allowed_hosts = {
        "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
        "youtu.be", "www.youtu.be"
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
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

@st.cache_data(show_spinner=False)
def get_video_info_and_transcript(video_id):
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        meta = {
            "title": yt.title or "Unknown Title",
            "author": yt.author or "Unknown Channel",
            "description": (yt.description or "")[:1000],
        }
        caption = None
        for lang_code in ['en', 'a.en']:
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
                parts = start_str.replace(',', '.').split(':')
                if len(parts) != 3:
                    continue
                try:
                    start_secs = float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
                except ValueError:
                    continue
                transcript.append({'start': start_secs, 'text': text})
        return meta, transcript, None
    except Exception as e:
        return {}, None, str(e)

def process_transcript(raw_transcript, meta, target_chunk_chars=600):
    all_docs = []
    
    title = meta.get('title', '')
    host = meta.get('author', 'Unknown Host')
    guest = "Unknown Guest"
    for pattern in [r' with (.+?)(?:\s*[-|,]|$)', r'feat(?:uring)?\.?\s+(.+?)(?:\s*[-|,]|$)', r':\s*(.+?) on ']:
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
    all_docs.append(Document(
        page_content=identity_text,
        metadata={"start": 0.0, "timestamp": "0:00", "type": "metadata"}
    ))
    
    current_text, current_start = "", 0.0
    for entry in raw_transcript:
        start_time = entry.get('start', 0.0)
        text_val = entry.get('text', '').strip()
        if not text_val:
            continue
        if not current_text:
            current_start = start_time
        current_text += text_val + " "
        if len(current_text) >= target_chunk_chars:
            all_docs.append(Document(
                page_content=current_text.strip(),
                metadata={"start": current_start, "timestamp": format_timestamp(current_start)}
            ))
            current_text, current_start = "", 0.0
    if current_text.strip():
        all_docs.append(Document(
            page_content=current_text.strip(),
            metadata={"start": current_start, "timestamp": format_timestamp(current_start)}
        ))
    return all_docs

def compute_relevance(question, retrieved_docs, embeddings_model):
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
    start_val = doc.metadata.get("start", 0.0)
    try:
        start_val = float(start_val)
    except (TypeError, ValueError):
        start_val = 0.0
    timestamp_val = doc.metadata.get("timestamp") or format_timestamp(start_val)
    return timestamp_val, start_val

# ── FIX 2: Cache the LLM in session state — no longer called per message ──
def get_llm():
    """
    Returns a working LLM. Priority: Grok (xAI) -> Gemini -> Mistral.
    Now logs the actual error so silent failures are visible.
    """
    llm_errors = []

    # 1) Try Grok (xAI)
    if XAI_API_KEY:
        try:
            llm = ChatXAI(
                model="grok-3-mini-fast",
                temperature=0.1,
                max_tokens=512,
                xai_api_key=XAI_API_KEY
            )
            llm.invoke("Say OK")
            return llm, "grok", []
        except Exception as e:
            llm_errors.append(f"Grok failed: {e}")

    # 2) Try Gemini
    if GOOGLE_API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1,
                max_output_tokens=512,
                google_api_key=GOOGLE_API_KEY
            )
            llm.invoke("Say OK")
            return llm, "gemini", []
        except Exception as e:
            llm_errors.append(f"Gemini failed: {e}")

    # 3) Fallback to Mistral via Hugging Face Inference API
    if HF_API_TOKEN:
        try:
            hf_llm = HuggingFaceEndpoint(
                repo_id='mistralai/Mistral-7B-Instruct-v0.2',
                task='text-generation',
                temperature=0.1
            )
            return ChatHuggingFace(llm=hf_llm), "mistral", []
        except Exception as e:
            llm_errors.append(f"Mistral failed: {e}")

    return None, "none", llm_errors
# ─────────────────────────────────────────────────────────────────────────

# --- Main Application ---
st.markdown('<div class="main-header">YT Chatbot <span style="font-size:0.6em">v2</span></div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#64748b; font-size:0.95rem;'>Parent-Document Retrieval • BM25 Hybrid • Reranking • Deep Linking • Live Quality Monitor</p>", unsafe_allow_html=True)

# Session State
# ── FIX 2 (cont): added 'llm' and 'llm_name' keys ──
for key, default in [("messages", []), ("memory", None), ("current_video", None),
                      ("retriever", None), ("video_meta", {}), ("quality_scores", []),
                      ("embeddings", None), ("llm", None), ("llm_name", None)]:
    if key not in st.session_state:
        st.session_state[key] = default
if st.session_state.memory is None:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, return_messages=True)

video_url = st.text_input("🔗 Enter YouTube Video Link", placeholder="https://www.youtube.com/watch?v=...")
video_id = extract_video_id(video_url)

if video_id:
    if st.session_state.current_video != video_id:
        st.session_state.current_video = video_id
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.video_meta = {}
        st.session_state.quality_scores = []
        st.session_state.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        # ── FIX 2 (cont): reset cached LLM on video change ──
        st.session_state.llm = None
        st.session_state.llm_name = None

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="video-card">', unsafe_allow_html=True)
        st.video(f"https://www.youtube.com/watch?v={video_id}")
        st.markdown('</div>', unsafe_allow_html=True)
        if st.session_state.video_meta:
            meta = st.session_state.video_meta
            st.markdown('<div class="video-intelligence-card">', unsafe_allow_html=True)
            st.markdown("**🎯 Video Intelligence**")
            st.markdown(f"📺 **Title**: {meta.get('title', 'N/A')}")
            st.markdown(f"👤 **Channel**: {meta.get('author', 'N/A')}")
            if meta.get('description'):
                with st.expander("📄 Description"):
                    st.caption(meta['description'][:500])
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if not st.session_state.retriever:
            with st.status("🚀 Initializing AI Engine...", expanded=True) as status:
                st.write("📂 Fetching video data & captions...")
                meta, raw_transcript, error_msg = get_video_info_and_transcript(video_id)
                if meta:
                    st.session_state.video_meta = meta

                if not raw_transcript:
                    st.write("⚠️ No captions found. Trying Whisper fallback...")
                    # ── FIX 5: use tempfile so cleanup is guaranteed even on crash ──
                    import tempfile
                    try:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            with st.spinner("🎵 Downloading audio..."):
                                audio_path = audio_utils.download_audio(
                                    f"https://www.youtube.com/watch?v={video_id}",
                                    output_dir=tmp_dir
                                )
                            with st.spinner("🧠 Transcribing with Whisper..."):
                                raw_transcript = audio_utils.transcribe_audio(audio_path)
                            st.success("✅ Whisper Transcription Complete!")
                    except Exception as we:
                        error_msg = f"Captions: {error_msg} | Whisper: {we}"
                        raw_transcript = None
                    # ── No manual cleanup needed — TemporaryDirectory handles it ──

                if raw_transcript:
                    st.write("🧪 Building PDR + BM25 Hybrid Index...")
                    docs = process_transcript(raw_transcript, meta)

                    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size)
                    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size)

                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    st.session_state.embeddings = embeddings

                    vectorstore = FAISS.from_documents([docs[0]], embeddings)
                    store = InMemoryStore()
                    pdr_retriever = ParentDocumentRetriever(
                        vectorstore=vectorstore, docstore=store,
                        child_splitter=child_splitter, parent_splitter=parent_splitter,
                    )
                    pdr_retriever.add_documents(docs)

                    bm25_retriever = BM25Retriever.from_documents(docs)
                    bm25_retriever.k = top_k

                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[pdr_retriever, bm25_retriever],
                        weights=[ensemble_weight, 1 - ensemble_weight]
                    )

                    compressor = FlashrankRerank()
                    st.session_state.retriever = ContextualCompressionRetriever(
                        base_compressor=compressor, base_retriever=ensemble_retriever
                    )

                    # ── FIX 2 (cont): load LLM once here, store in session state ──
                    st.write("🔌 Connecting to AI backend...")
                    model, model_name, errors = get_llm()
                    st.session_state.llm = model
                    st.session_state.llm_name = model_name
                    if errors:
                        for err in errors:
                            st.warning(f"⚠️ {err}")
                    # ────────────────────────────────────────────────────────────

                    status.update(label="✨ Analysis Ready!", state="complete", expanded=False)
                else:
                    status.update(label="💨 Failed to load transcript.", state="error")
                    st.error(f"Error: {error_msg}")

        # --- Chat History ---
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and msg.get("confidence") is not None:
                    conf = msg["confidence"]
                    bc = "badge-high" if conf >= 0.50 else ("badge-medium" if conf >= 0.30 else "badge-low")
                    st.markdown(f'<span class="confidence-badge {bc}">{conf:.0%} In-Scope</span>', unsafe_allow_html=True)
                st.markdown(msg["content"])
                if msg.get("citations"):
                    with st.expander("📍 Source Citations"):
                        for cite in msg["citations"]:
                            if cite.get("start", 0) > 0:
                                st.markdown(
                                    f'<a href="https://youtu.be/{video_id}?t={int(cite["start"])}" target="_blank" class="timestamp-link">[{cite["timestamp"]}]</a> {cite["text"][:130]}...',
                                    unsafe_allow_html=True
                                )

        # --- Chat Input ---
        if prompt := st.chat_input("Ask anything about this video..."):
            if not st.session_state.retriever:
                st.warning("Please wait for analysis to complete.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("⏳ Searching transcript..."):
                        retrieved_docs = st.session_state.retriever.invoke(prompt)
                        context = "\n\n".join([
                            f"[{normalize_doc_metadata(d)[0]}] {d.page_content}"
                            for d in retrieved_docs
                        ])
                        relevance = compute_relevance(prompt, retrieved_docs, st.session_state.embeddings)

                    META_PATTERNS = [
                        "summary", "summarize", "overview", "key point", "main point",
                        "what is this video", "what are they talk", "what did they discuss",
                        "what is the video about", "tell me about", "topics covered",
                        "what happened", "highlight", "takeaway", "conclusion", "key insight",
                        "who are", "who is", "what is the topic", "explain", "describe"
                    ]
                    is_meta = any(p in prompt.lower() for p in META_PATTERNS)
                    is_blocked = (not is_meta) and (relevance < 0.20)

                    if is_blocked:
                        ai_text = "❌ **This question is not covered in this video.** I can only answer questions based on the video transcript."
                        st.warning(ai_text)
                        st.session_state.memory.save_context({"input": prompt}, {"output": ai_text})
                        st.session_state.messages.append({
                            "role": "assistant", "content": ai_text,
                            "citations": [], "confidence": None
                        })
                    else:
                        bc = "badge-high" if relevance >= 0.50 else ("badge-medium" if relevance >= 0.30 else "badge-low")
                        st.markdown(f'<span class="confidence-badge {bc}">{relevance:.0%} In-Scope</span>', unsafe_allow_html=True)

                        # ── FIX 2 (cont): use cached LLM from session state ──
                        model = st.session_state.llm
                        model_name = st.session_state.llm_name
                        # ────────────────────────────────────────────────────

                        if model is None:
                            ai_text = (
                                "⚠️ No LLM backend is configured. Add one of these API keys in `.env`: "
                                "`XAI_API_KEY`, `GOOGLE_API_KEY`, or `HUGGINGFACEHUB_API_TOKEN`."
                            )
                            st.error(ai_text)
                        else:
                            model_labels = {
                                "grok": "⚡ Grok (xAI)",
                                "gemini": "✨ Gemini",
                                "mistral": "🔄 Mistral (fallback)"
                            }
                            st.caption(f"Using: {model_labels.get(model_name, model_name)}")

                            instructions = {
                                "Quick Summary":   "Answer in exactly 2-3 bullet points.",
                                "Moderate Detail": "Answer in 1-2 short paragraphs. Be concise.",
                                "Deep Dive":       "Answer in up to 4 paragraphs with technical depth."
                            }

                            prompt_tpl = ChatPromptTemplate.from_messages([
                                ("system", f"""You are a strict video-transcript Q&A assistant.

RULES — follow without exception:
1. Answer ONLY from the transcript context below. ZERO outside knowledge.
2. If the answer is NOT in the context, say: "This is not covered in the video."
3. Do NOT use your pre-trained knowledge, even if you know the answer.
4. {instructions[explanation_type]}
5. Use the VIDEO IDENTITY section for speaker/host questions.
6. Never repeat the same information.

Context:
{context}"""),
                                MessagesPlaceholder(variable_name="history"),
                                ("human", "{question}")
                            ])

                            chain = prompt_tpl | model
                            history = st.session_state.memory.load_memory_variables({})['history']

                            try:
                                def stream_generator():
                                    full_text = ""
                                    for chunk in chain.stream({"history": history, "question": prompt}):
                                        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                                        if not full_text and "Assistant:" in content:
                                            content = content.split("Assistant:")[-1].lstrip()
                                        full_text += content
                                        yield content

                                ai_text = st.write_stream(stream_generator())
                            except Exception as llm_err:
                                ai_text = f"⚠️ LLM error: {str(llm_err)[:200]}. Please try again in a moment."
                                st.error(ai_text)

                        st.session_state.memory.save_context({"input": prompt}, {"output": ai_text})
                        st.session_state.quality_scores.append(relevance)

                        citations = []
                        for d in retrieved_docs:
                            timestamp_val, start_val = normalize_doc_metadata(d)
                            citations.append({
                                "timestamp": timestamp_val,
                                "start": start_val,
                                "text": d.page_content
                            })
                        st.session_state.messages.append({
                            "role": "assistant", "content": ai_text,
                            "citations": citations, "confidence": relevance
                        })

                        if citations:
                            with st.expander("📍 Source Citations"):
                                for cite in citations:
                                    if cite.get("start", 0) > 0:
                                        st.markdown(
                                            f'<a href="https://youtu.be/{video_id}?t={int(cite["start"])}" target="_blank" class="timestamp-link">[{cite["timestamp"]}]</a> {cite["text"][:130]}...',
                                            unsafe_allow_html=True
                                        )

else:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.info("📄 **Parent-Document Retrieval**\nPrecise timestamp search + full context for the LLM.")
    c2.info("🔍 **BM25 + Semantic Hybrid**\nKeyword + embedding search for maximum recall.")
    c3.info("📊 **Live Quality Monitor**\nConfidence badge on every answer — transparency no other tool offers.")
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#475569;'>Paste a YouTube link above to get started</p>", unsafe_allow_html=True)
