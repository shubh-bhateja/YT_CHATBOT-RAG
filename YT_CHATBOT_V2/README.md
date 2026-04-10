# YouTube Chatbot V2: Beast Mode RAG

This is an advanced version of the YouTube Chatbot, designed to showcase production-grade RAG (Retrieval-Augmented Generation) capabilities. It moves beyond naive RAG by implementing hybrid search, contextual memory, and deep linking for a true "BEAST" portfolio project.

## 🌟 Advanced Features

- **Hybrid Search (Ensemble Retrieval)**: 
  - Combines **Dense Retrieval** (semantic search via FAISS) with **Sparse Retrieval** (keyword search via BM25).
  - This ensures that specific technical terms ("AlphaFold", "DeepMind") are caught even if semantic embeddings are slightly off.
- **Reranking (FlashRank)**:
  - Uses a lightweight reranker to re-score the top retrieved chunks, ensuring the LLM receives only the most relevant context.
- **Conversational Memory**:
  - Implements a history-aware chain that remembers the last 5 turns of conversation.
  - Supports follow-up questions like "Explain the second point further" or "Why did he say that?".
- **Deep Linking Citations**:
  - Automatically extracts timestamps from the YouTube transcript.
  - Every answer includes clickable links that jump the video to the exact moment the information was discussed.
- **Premium UI/UX**:
  - Built with a modern dark-themed interface in Streamlit.
  - Uses `st.chat_message` for a professional chat feel and simulated streaming responses.

## 🛠️ Tech Stack

- **Framework**: LangChain
- **UI**: Streamlit
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS (CPU)
- **Keyword Search**: BM25
- **Reranker**: FlashRank
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.2` (via HuggingFace)
- **Evaluation**: Ragas

## 🚀 Getting Started

1.  Navigate to this folder:
    ```bash
    cd YT_CHATBOT_V2
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up your `.env`:
    ```env
    HUGGINGFACEHUB_API_TOKEN=your_token_here
    ```
4.  Run the app:
    ```bash
    streamlit run app.py
    ```

## 📊 Evaluation

Check out `evaluate_rag.py` for a demonstration of how to evaluate your RAG pipeline using metrics like **Faithfulness**, **Answer Relevance**, and **Context Recall**.
