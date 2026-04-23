"""
rag_engine.py — RAG pipeline construction and LLM orchestration.

Builds the full PDR + BM25 + Flashrank Rerank retriever and handles
LLM selection.  No Streamlit dependency.
"""

from config import (
    HF_API_TOKEN,
    EMBEDDING_MODEL_NAME,
    LLM_REPO_ID,
    EXPLANATION_INSTRUCTIONS,
    # LangChain classes
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
    FAISS,
    BM25Retriever,
    RecursiveCharacterTextSplitter,
    EnsembleRetriever,
    ContextualCompressionRetriever,
    ParentDocumentRetriever,
    InMemoryStore,
    FlashrankRerank,
)


# ── Retriever construction ───────────────────────────────────────────────────

def get_embeddings():
    """Return the shared HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_retriever(docs, embeddings, child_size=300, parent_size=1200, top_k=10, ensemble_weight=0.6):
    """
    Build the full PDR + BM25 + Flashrank Rerank retriever.

    Args:
        docs:             List of LangChain Documents (from utils.process_transcript).
        embeddings:       HuggingFaceEmbeddings instance.
        child_size:       Chunk size for the child splitter (granular search).
        parent_size:      Chunk size for the parent splitter (context for LLM).
        top_k:            Number of documents BM25 returns.
        ensemble_weight:  Weight for the PDR retriever (1 − weight goes to BM25).

    Returns:
        A ContextualCompressionRetriever backed by the hybrid ensemble.
    """
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size)

    vectorstore = FAISS.from_documents([docs[0]], embeddings)
    store = InMemoryStore()
    pdr_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    pdr_retriever.add_documents(docs)

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[pdr_retriever, bm25_retriever],
        weights=[ensemble_weight, 1 - ensemble_weight],
    )

    compressor = FlashrankRerank()
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )
    return retriever


# ── LLM selection ────────────────────────────────────────────────────────────

def get_llm():
    """
    Return (model, model_name, errors).

    Currently uses Mistral-7B via the HuggingFace Inference API.
    Extend this function to add Grok / Gemini / other providers.
    """
    if not HF_API_TOKEN:
        return None, "none", ["HUGGINGFACEHUB_API_TOKEN is missing in .env"]

    try:
        hf_llm = HuggingFaceEndpoint(
            repo_id=LLM_REPO_ID,
            task="text-generation",
            temperature=0.1,
            huggingfacehub_api_token=HF_API_TOKEN,
        )
        return ChatHuggingFace(llm=hf_llm), "mistral", []
    except Exception as e:
        return None, "none", [f"Mistral connection failed: {e}"]


# ── QA chain construction ────────────────────────────────────────────────────

def build_qa_chain(model, context, video_meta, explanation_type):
    """
    Build and return a prompt_template | model chain.

    Args:
        model:             The Chat LLM instance.
        context:           Formatted context string from the retriever.
        video_meta:        Dict with 'title' and 'author' keys.
        explanation_type:  One of 'Quick Summary', 'Moderate Detail', 'Deep Dive'.

    Returns:
        A LangChain chain (prompt | model) that accepts
        {"history": ..., "question": ...}.
    """
    instruction = EXPLANATION_INSTRUCTIONS.get(explanation_type, EXPLANATION_INSTRUCTIONS["Moderate Detail"])
    title = video_meta.get("title", "Unknown")
    author = video_meta.get("author", "Unknown")

    prompt_tpl = ChatPromptTemplate.from_messages([
        ("system", f"""You are a Video Intelligence Agent. Your specialized world is defined ONLY by the video metadata and transcript provided below.

INSTRUCTIONS:
1. RELIANCE: Use the provided Context to answer specific factual questions.
2. IDENTITY: You know the video title and speaker from the 'VIDEO IDENTITY' section. Use this for 'Who is...' or 'What is this video about?' questions.
3. GLOBAL UNDERSTANDING: If the user asks for a summary, key points, or a general overview, use your intelligence to synthesize the provided Context chunks into a coherent answer.
4. BOUNDARIES: If the user asks about topics completely unrelated to the video (e.g., general world news, sexuality, unrelated celebrities, or general trivia), you must politely decline by saying: "I am specialized in analyzing this specific video, and it does not discuss that topic."
5. BRAIN: Do not say you are an AI or that you cannot watch the video. You have 'seen' the content through the provided transcript.
6. {instruction}

VIDEO IDENTITY:
{title} | {author}

Context:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    return prompt_tpl | model


def generate_video_summary(llm, docs, meta):
    """
    Generate a high-level executive summary of the video transcript.
    """
    if not docs:
        return "No transcript available for summary."

    title = meta.get("title", "this video")
    
    # Take a representative sample of chunks to stay within context limits
    # (First 4 and Last 4 chunks usually cover the intro and conclusion well)
    if len(docs) > 8:
        sample_docs = docs[:4] + docs[-4:]
    else:
        sample_docs = docs
        
    context_text = "\n\n".join([d.page_content for d in sample_docs])

    prompt = PromptTemplate.from_template(
        "You are a professional video analyst. Based on the following transcript segments from '{title}', "
        "provide a concise executive summary formatted exactly as follows:\n\n"
        "HOOK: [One catchy sentence about the video's essence]\n"
        "TAKEAWAYS:\n"
        "- [Key point 1]\n"
        "- [Key point 2]\n"
        "- [Key point 3]\n\n"
        "SUGGESTION: [One high-value question the user could ask to learn more]\n\n"
        "Transcript segments:\n{context}"
    )

    try:
        # We use invoke here for a non-streaming, immediate response
        response = llm.invoke(prompt.format(title=title, context=context_text))
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"⚠️ Summary generation failed: {str(e)}"

# End of file
