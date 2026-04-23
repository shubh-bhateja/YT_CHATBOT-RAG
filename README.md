# YouTube AI Chatbot: Advanced RAG with Video Intelligence

A production-grade RAG (Retrieval-Augmented Generation) pipeline for extracting high-context insights from YouTube transcripts. This isn't just a basic prompt-wrapper; it's a multi-stage search architecture designed to handle long-form content with precise citations and observability.

## 🚀 The "Beast Mode" Architecture

The system is built to solve the "context window" problem by using a two-stage retrieval process that prioritizes accuracy and speed.

### 1. The RAG Engine (PDR + Hybrid Search)
Standard RAG often fails because chunks are too small to be meaningful or too large to be precise. I implemented a **Parent-Document Retrieval (PDR)** strategy:
*   **Search Layer**: Granular 300-char "child" chunks for high-precision semantic matching.
*   **Context Layer**: When a match is found, the system retrieves the full 1200-char "parent" paragraph to provide the LLM with enough context to avoid hallucinations.
*   **Hybrid Search**: Combines **BM25 (Keyword)** and **FAISS (Semantic)** search via an Ensemble Retriever. This ensures specific technical terms or names aren't lost in the embedding space.
*   **FlashRank Reranking**: A cross-encoder model re-scores the top 10 results, moving the most relevant context to the top of the list before it hits the LLM.

### 2. Audio Engineering (Whisper Integration)
For videos without captions, the app implements a local transcription fallback:
*   **Faster-Whisper**: A CTranslate2 implementation of OpenAI’s Whisper, which is ~4x faster than the original.
*   **Int8 Quantization**: The model is quantized to 8-bit integers, allowing 2-hour videos to be transcribed in minutes on standard CPUs without massive RAM overhead.
*   **VAD Filtering**: Voice Activity Detection skips silences, ensuring the transcript remains concise and focused on speech.

### 3. RAG Observability
Built-in **Rerank Trace** dashboard shows you the "under the hood" logic for every query. You can see the initial retrieval scores vs. the final reranked scores, making it easy to debug why the AI chose specific parts of the video as context.

---

## 🛠️ Tech Stack
*   **Orchestration**: LangChain
*   **Frontend**: Streamlit (with deep-linked timestamp citations)
*   **Vector DB**: FAISS
*   **Models**: Mistral-7B (HuggingFace), Gemini-Pro, Grok
*   **Reranker**: FlashRank
*   **Transcription**: Faster-Whisper + Pytubefix

---

## 🏁 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Add your keys to a `.env` file:
```env
HUGGINGFACEHUB_API_TOKEN=your_token
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## 📊 Scale & Deployment
The app is stateless and container-ready. By using FAISS for local search and offloading inference to HuggingFace, it scales horizontally easily. For persistence, the `InMemoryStore` can be swapped for a Redis or MongoDB backend.
