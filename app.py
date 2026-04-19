"""
app.py — Streamlit frontend.

This file handles ONLY the UI: page config, CSS, sidebar widgets,
video display, chat rendering, and user input.  All business logic
lives in utils.py, rag_engine.py, and config.py.
"""

import streamlit as st
import tempfile

from config import (
    PRESETS,
    META_PATTERNS,
    DEFAULT_CHILD_SIZE,
    DEFAULT_PARENT_SIZE,
    ConversationBufferWindowMemory,
)
from utils import (
    extract_video_id,
    get_video_info_and_transcript,
    process_transcript,
    compute_relevance,
    normalize_doc_metadata,
    is_meta_question,
)
from rag_engine import get_embeddings, build_retriever, get_llm, build_qa_chain, generate_video_summary
import audio_utils


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="YT Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════

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
    .executive-summary-card {
        background: linear-gradient(135deg, rgba(236,72,153,0.1), rgba(168,85,247,0.1));
        padding: 20px; border-radius: 16px;
        border: 1px solid rgba(236,72,153,0.3);
        margin-top: 1rem;
    }
    .summary-hook {
        font-size: 1.1rem; font-weight: 600; color: #f472b6;
        margin-bottom: 0.8rem; line-height: 1.4;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    explanation_type = st.selectbox(
        "Explanation Style",
        list(PRESETS.keys()),
        index=1,
    )
    current_preset = PRESETS[explanation_type]

    st.markdown("---")
    manual_tuning = st.toggle("🛠️ Enable Manual Tuning", value=False)
    if manual_tuning:
        with st.expander("🔬 Manual Engine Tuning", expanded=True):
            child_size = st.slider("Child Chunk Size", 100, 600, DEFAULT_CHILD_SIZE)
            parent_size = st.slider("Parent Chunk Size", 600, 3000, DEFAULT_PARENT_SIZE)
            top_k = st.slider("Top K Retrieved", 2, 10, current_preset["top_k"])
            ensemble_weight = st.slider("Ensemble Weight", 0.0, 1.0, current_preset["ensemble_weight"])
    else:
        child_size, parent_size = DEFAULT_CHILD_SIZE, DEFAULT_PARENT_SIZE
        top_k = current_preset["top_k"]
        ensemble_weight = current_preset["ensemble_weight"]
        st.info(f"✨ **Smart Autopilot**: Optimal for *{explanation_type}*.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔬 RAG Observability")
    st.caption("Detailed diagnostic trace of the most recent retrieval.")
    
    trace = st.session_state.get("rerank_trace", [])
    if trace:
        import pandas as pd
        df = pd.DataFrame(trace)
        # Style the dataframe for a more premium look
        st.dataframe(
            df,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Score": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=1),
                "Source": "Source Chunk",
                "Delta": "Delta"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("Ask a question to see the search log.")


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="main-header">YT Chatbot </div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:0.95rem;'>"
    "Parent-Document Retrieval • BM25 Hybrid • FlashRank Reranking • RAG Observability"
    "</p>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("messages", []),
    ("memory", None),
    ("current_video", None),
    ("retriever", None),
    ("video_meta", {}),
    ("rerank_trace", []),
    ("video_summary", None),
    ("embeddings", None),
    ("llm", None),
    ("llm_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default
if st.session_state.memory is None:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, return_messages=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED TRANSCRIPT FETCH  (wraps the pure function with Streamlit caching)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _cached_video_info(video_id):
    return get_video_info_and_transcript(video_id)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

video_url = st.text_input("🔗 Enter YouTube Video Link", placeholder="https://www.youtube.com/watch?v=...")
video_id = extract_video_id(video_url)

if video_id:
    # Reset state when the video changes
    if st.session_state.current_video != video_id:
        st.session_state.current_video = video_id
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.video_meta = {}
        st.session_state.rerank_trace = []
        st.session_state.video_summary = None
        st.session_state.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        st.session_state.llm = None
        st.session_state.llm_name = None

    col1, col2 = st.columns([1, 1], gap="large")

    # ── Left column: video player & metadata ──
    with col1:
        with st.container(border=True):
            st.video(f"https://www.youtube.com/watch?v={video_id}")
        if st.session_state.video_meta:
            meta = st.session_state.video_meta
            st.markdown('<div class="video-intelligence-card">', unsafe_allow_html=True)
            st.markdown("**🎯 Video Intelligence**")
            st.markdown(f"📺 **Title**: {meta.get('title', 'N/A')}")
            st.markdown(f"👤 **Channel**: {meta.get('author', 'N/A')}")
            if meta.get("description"):
                with st.expander("📄 Description"):
                    st.caption(meta["description"][:500])
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.video_summary:
            summary = st.session_state.video_summary
            st.markdown('<div class="executive-summary-card">', unsafe_allow_html=True)
            st.markdown("### 🏆 Executive Summary")
            
            # Simple parsing for our custom format
            hook = ""
            takeaways = ""
            suggestion = ""
            
            if "HOOK:" in summary:
                hook = summary.split("HOOK:")[1].split("TAKEAWAYS:")[0].strip()
            if "TAKEAWAYS:" in summary:
                takeaways = summary.split("TAKEAWAYS:")[1].split("SUGGESTION:")[0].strip()
            if "SUGGESTION:" in summary:
                suggestion = summary.split("SUGGESTION:")[1].strip()
            
            if hook:
                st.markdown(f'<p class="summary-hook">{hook}</p>', unsafe_allow_html=True)
            if takeaways:
                st.markdown("**Key Takeaways:**")
                st.markdown(takeaways)
            if suggestion:
                st.info(f"💡 **Try asking:** {suggestion}")
            else:
                # Fallback if parsing fails
                st.markdown(summary)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Right column: AI engine + chat ──
    with col2:
        # --- Initialization pipeline ---
        if not st.session_state.retriever:
            with st.status("🚀 Initializing AI Engine...", expanded=True) as status:
                st.write("📂 Fetching video data & captions...")
                meta, raw_transcript, error_msg = _cached_video_info(video_id)
                if meta:
                    st.session_state.video_meta = meta

                # Whisper fallback
                if not raw_transcript:
                    st.write("⚠️ No captions found. Trying Whisper fallback...")
                    try:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            with st.spinner("🎵 Downloading audio..."):
                                audio_path = audio_utils.download_audio(
                                    f"https://www.youtube.com/watch?v={video_id}",
                                    output_dir=tmp_dir,
                                )
                            with st.spinner("🧠 Transcribing with Whisper..."):
                                raw_transcript = audio_utils.transcribe_audio(audio_path)
                            st.success("✅ Whisper Transcription Complete!")
                    except Exception as we:
                        error_msg = f"Captions: {error_msg} | Whisper: {we}"
                        raw_transcript = None

                if raw_transcript:
                    st.write("🧪 Building PDR + BM25 Hybrid Index...")
                    docs = process_transcript(raw_transcript, meta)

                    embeddings = get_embeddings()
                    st.session_state.embeddings = embeddings

                    st.session_state.retriever = build_retriever(
                        docs, embeddings, child_size, parent_size, top_k, ensemble_weight,
                    )

                    st.write("🔌 Connecting to AI backend...")
                    model, model_name, errors = get_llm()
                    st.session_state.llm = model
                    st.session_state.llm_name = model_name
                    if errors:
                        for err in errors:
                            st.warning(f"⚠️ {err}")
                    
                    if not st.session_state.video_summary:
                        st.write("📝 Generating Executive Summary...")
                        st.session_state.video_summary = generate_video_summary(
                            st.session_state.llm, docs, meta
                        )

                    status.update(label="✨ Analysis Ready!", state="complete", expanded=False)
                else:
                    status.update(label="💨 Failed to load transcript.", state="error")
                    st.error(f"Error: {error_msg}")

        # --- Chat history ---
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("citations"):
                        with st.expander("📍 Source Citations"):
                            for cite in msg["citations"]:
                                if cite.get("start", 0) > 0:
                                    st.markdown(
                                        f'<a href="https://youtu.be/{video_id}?t={int(cite["start"])}" '
                                        f'target="_blank" class="timestamp-link">[{cite["timestamp"]}]</a> '
                                        f'{cite["text"][:130]}...',
                                        unsafe_allow_html=True,
                                    )

        # --- Chat input ---
        if prompt := st.chat_input("Ask anything about this video..."):
            with chat_container:
                if not st.session_state.retriever:
                    st.warning("Please wait for analysis to complete.")
                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        # --- Stage 1: Initial Hybrid Search ---
                        with st.spinner("🔍 Initial search (FAISS + BM25)..."):
                            # Get the underlying ensemble retriever from our ContextualCompressionRetriever
                            base_retriever = st.session_state.retriever.base_retriever
                            initial_docs = base_retriever.invoke(prompt)
                            
                        # --- Stage 2: FlashRank Reranking ---
                        with st.spinner("🧠 FlashRank Reranking..."):
                            reranker = st.session_state.retriever.base_compressor
                            retrieved_docs = reranker.compress_documents(initial_docs, prompt)
                            
                        # --- Compute Rerank Trace for Observability ---
                        trace = []
                        for i, doc in enumerate(retrieved_docs):
                            # Try to find the document's original position in initial_docs
                            # Flashrank usually returns a 'relevance_score' in metadata
                            score = doc.metadata.get("relevance_score", 0)
                            
                            # Find original rank by content matching (best effort)
                            orig_rank = "N/A"
                            for j, orig_doc in enumerate(initial_docs):
                                if orig_doc.page_content == doc.page_content:
                                    orig_rank = j + 1
                                    break
                            
                            delta = ""
                            if isinstance(orig_rank, int):
                                diff = orig_rank - (i + 1)
                                if diff > 0: delta = f"⬆️ {diff}"
                                elif diff < 0: delta = f"⬇️ {abs(diff)}"
                                else: delta = "⏺️"
                            
                            trace.append({
                                "Rank": i + 1,
                                "Delta": delta,
                                "Source": doc.page_content[:60] + "...",
                                "Score": float(score)
                            })
                        
                        st.session_state.rerank_trace = trace
                        
                        # --- Final Response Generation ---
                        context = "\n\n".join(
                            [f"[{normalize_doc_metadata(d)[0]}] {d.page_content}" for d in retrieved_docs]
                        )
                        
                        model = st.session_state.llm
                        model_name = st.session_state.llm_name

                        if model is None:
                            ai_text = (
                                "⚠️ No LLM backend is configured. "
                                "Add `HUGGINGFACEHUB_API_TOKEN` to your `.env` file."
                            )
                            st.error(ai_text)
                        else:
                            model_labels = {
                                "grok": "⚡ Grok (xAI)",
                                "gemini": "✨ Gemini",
                                "mistral": "🔄 Mistral",
                            }
                            st.caption(f"Using: {model_labels.get(model_name, model_name)}")

                            chain = build_qa_chain(
                                model, context, st.session_state.video_meta, explanation_type,
                            )
                            history = st.session_state.memory.load_memory_variables({})["history"]

                            try:
                                def stream_generator():
                                    full_text = ""
                                    for chunk in chain.stream({"history": history, "question": prompt}):
                                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                                        if not full_text and "Assistant:" in content:
                                            # Strip accidental prefixes
                                            content = content.split("Assistant:")[-1].lstrip()
                                        full_text += content
                                        yield content

                                ai_text = st.write_stream(stream_generator())
                            except Exception as llm_err:
                                ai_text = f"⚠️ LLM error: {str(llm_err)[:200]}. Please try again."
                                st.error(ai_text)

                        st.session_state.memory.save_context({"input": prompt}, {"output": ai_text})
                        
                        citations = []
                        for d in retrieved_docs:
                            timestamp_val, start_val = normalize_doc_metadata(d)
                            citations.append({
                                "timestamp": timestamp_val,
                                "start": start_val,
                                "text": d.page_content,
                            })
                        st.session_state.messages.append({
                            "role": "assistant", "content": ai_text,
                            "citations": citations
                        })

                        if citations:
                            with st.expander("📍 Source Citations"):
                                for cite in citations:
                                    if cite.get("start", 0) > 0:
                                        st.markdown(
                                            f'<a href="https://youtu.be/{video_id}?t={int(cite["start"])}" '
                                            f'target="_blank" class="timestamp-link">[{cite["timestamp"]}]</a> '
                                            f'{cite["text"][:130]}...',
                                            unsafe_allow_html=True,
                                        )

else:
    # Landing page
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.info("📄 **Parent-Document Retrieval**\nPrecise timestamp search + full context for the LLM.")
    c2.info("🔍 **BM25 + Semantic Hybrid**\nKeyword + embedding search for maximum recall.")
    c3.info("🔬 **RAG Observability**\nLive Rerank Trace shows exactly how the AI selects sources.")
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#475569;'>Paste a YouTube link above to get started</p>",
        unsafe_allow_html=True,
    )
