import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever, ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()


# ── FIX 3: Build the REAL retriever from transcript docs ──────────────────
def build_retriever(docs, child_size=300, parent_size=1200, top_k=5, ensemble_weight=0.6):
    """
    Builds the same PDR + BM25 + reranking pipeline used in app.py.
    This is what makes the evaluation meaningful — it tests the actual system.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size)

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
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    return retriever
# ──────────────────────────────────────────────────────────────────────────


def generate_synthetic_data(transcript_docs, llm, num_questions=3):
    """
    Generate synthetic question-answer pairs from the provided context.
    """
    print(f"🧠 Generating {num_questions} synthetic test cases...")

    gen_prompt = PromptTemplate.from_template(
        "Based on the following video transcript context, generate {num_questions} "
        "challenging factual questions and their ground truth answers. "
        "Format as: Q: <question> | A: <answer>\n\nContext:\n{context}"
    )

    context_text = "\n".join([doc.page_content for doc in transcript_docs[:5]])
    response = llm.invoke(gen_prompt.format(num_questions=num_questions, context=context_text))

    lines = response.content.split("\n")
    questions, answers = [], []
    for line in lines:
        if "|" in line and ("Q:" in line or "?" in line):
            parts = line.split("|")
            questions.append(parts[0].replace("Q:", "").strip())
            answers.append(parts[1].replace("A:", "").strip())

    return questions[:num_questions], answers[:num_questions]


def run_evaluation(video_id="dQw4w9WgXcQ"):
    """
    Runs a REAL RAGAS evaluation using the actual RAG retrieval pipeline.
    """
    print(f"🚀 Starting RAG Evaluation for video: {video_id}...")

    # 1. Fetch transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        docs = [Document(page_content=entry['text']) for entry in transcript]
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        return

    # 2. Build the REAL retriever (same pipeline as app.py)
    print("🔧 Building retrieval pipeline...")
    retriever = build_retriever(docs)

    # 3. Setup Evaluator LLM
    llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        task='text-generation',
        temperature=0.1
    )
    evaluator_llm = ChatHuggingFace(llm=llm)

    # 4. Generate Questions & Ground Truth
    questions, ground_truths = generate_synthetic_data(docs, evaluator_llm)
    if not questions:
        print("❌ Failed to generate test cases.")
        return

    # ── FIX 3 (cont): retrieve contexts using the REAL pipeline per question ──
    print("🔍 Running real retrieval for each question...")
    real_contexts = []
    for q in questions:
        retrieved = retriever.invoke(q)
        real_contexts.append([d.page_content for d in retrieved])
    # ──────────────────────────────────────────────────────────────────────────

    # 5. Generate answers from the LLM conditioned on retrieved context
    print("💬 Generating answers from LLM...")
    sample_answers = []
    for q, ctx_list in zip(questions, real_contexts):
        context_str = "\n\n".join(ctx_list)
        resp = evaluator_llm.invoke(
            f"Answer the question ONLY using the context below.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {q}"
        )
        sample_answers.append(resp.content)

    # 6. Prepare Dataset and run RAGAS
    data = {
        "question": questions,
        "answer": sample_answers,
        "contexts": real_contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    print("📊 Running RAGAS metrics...")
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevance],
            llm=evaluator_llm,
            raise_exceptions=False
        )
        print("\n✅ Evaluation Results:")
        print(result)

        # Save results to CSV for your README / portfolio evidence
        df = result.to_pandas()
        df.to_csv(f"ragas_results_{video_id}.csv", index=False)
        print(f"\n💾 Results saved to ragas_results_{video_id}.csv")

    except Exception as e:
        print(f"⚠️ RAGAS execution failed: {e}")
        print("Tip: RAGAS works best with GPT-4 class models. "
              "Consider swapping evaluator_llm for ChatGoogleGenerativeAI with Gemini.")


if __name__ == "__main__":
    run_evaluation()
