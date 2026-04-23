"""
config.py — Central configuration, constants, and robust LangChain imports.

All environment variables, LangChain version-safe imports, and application
constants are defined here so every other module can do:
    from config import <whatever it needs>
"""

import os
from dotenv import load_dotenv

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv(override=True)

HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# ── Robust LangChain Imports (supports 0.1 → 0.3+ and langchain-classic) ────
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# Retrievers — try standard langchain first, fall back to langchain-classic
try:
    from langchain.retrievers import (
        EnsembleRetriever,
        ContextualCompressionRetriever,
        ParentDocumentRetriever,
    )
except ImportError:
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
    except ImportError:
        try:
            from langchain_classic.retrievers import (
                EnsembleRetriever,
                ContextualCompressionRetriever,
                ParentDocumentRetriever,
            )
        except ImportError:
            raise ImportError(
                "Could not find EnsembleRetriever / ContextualCompressionRetriever / "
                "ParentDocumentRetriever. Install langchain or langchain-classic."
            )

# InMemoryStore
try:
    from langchain_core.stores import InMemoryStore
except ImportError:
    from langchain.storage import InMemoryStore

# ConversationBufferWindowMemory
try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    try:
        from langchain_community.memory import ConversationBufferWindowMemory
    except ImportError:
        try:
            from langchain_classic.memory import ConversationBufferWindowMemory
        except ImportError:
            raise ImportError(
                "Could not find ConversationBufferWindowMemory. "
                "Install langchain or langchain-classic."
            )

# ── Application Constants ────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"

PRESETS = {
    "Quick Summary":   {"top_k": 3, "ensemble_weight": 0.4},
    "Moderate Detail": {"top_k": 5, "ensemble_weight": 0.6},
    "Deep Dive":       {"top_k": 7, "ensemble_weight": 0.8},
}

EXPLANATION_INSTRUCTIONS = {
    "Quick Summary":   "Answer in exactly 2-3 bullet points.",
    "Moderate Detail": "Answer in 1-2 short paragraphs. Be concise.",
    "Deep Dive":       "Answer in up to 4 paragraphs with technical depth.",
}

META_PATTERNS = [
    "summary", "summarize", "overview", "key point", "main point",
    "what is this video", "what are they talk", "what did they discuss",
    "what is the video about", "tell me about", "topics covered",
    "what happened", "highlight", "takeaway", "conclusion", "key insight",
    "who are", "who is", "what is the topic", "explain", "describe",
    "who", "host", "guest", "all about", "everyone", "speaker", "person",
]

DEFAULT_CHILD_SIZE = 300
DEFAULT_PARENT_SIZE = 1200

# End of file
