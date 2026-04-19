"""
evaluate_rag.py — RAGAS evaluation harness.

Uses the SAME retrieval pipeline as the main app (imported from rag_engine)
so the evaluation scores reflect real-world performance.
"""

import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from rag_engine import build_retriever, get_embeddings

load_dotenv()


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
        docs = [Document(page_content=entry["text"]) for entry in transcript]
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        return

    # 2. Build the REAL retriever (same pipeline as app.py)
    print("🔧 Building retrieval pipeline...")
    embeddings = get_embeddings()
    retriever = build_retriever(docs, embeddings)

    # 3. Setup Evaluator LLM
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        temperature=0.1,
    )
    evaluator_llm = ChatHuggingFace(llm=llm)

    # 4. Generate Questions & Ground Truth
    questions, ground_truths = generate_synthetic_data(docs, evaluator_llm)
    if not questions:
        print("❌ Failed to generate test cases.")
        return

    # 5. Retrieve contexts using the REAL pipeline per question
    print("🔍 Running real retrieval for each question...")
    real_contexts = []
    for q in questions:
        retrieved = retriever.invoke(q)
        real_contexts.append([d.page_content for d in retrieved])

    # 6. Generate answers from the LLM conditioned on retrieved context
    print("💬 Generating answers from LLM...")
    sample_answers = []
    for q, ctx_list in zip(questions, real_contexts):
        context_str = "\n\n".join(ctx_list)
        resp = evaluator_llm.invoke(
            f"Answer the question ONLY using the context below.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {q}"
        )
        sample_answers.append(resp.content)

    # 7. Prepare Dataset and run RAGAS
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
            metrics=[faithfulness, answer_relevancy],
            llm=evaluator_llm,
            raise_exceptions=False,
        )
        print("\n✅ Evaluation Results:")
        print(result)

        # Save results to CSV for your README / portfolio evidence
        df = result.to_pandas()
        df.to_csv(f"ragas_results_{video_id}.csv", index=False)
        print(f"\n💾 Results saved to ragas_results_{video_id}.csv")

    except Exception as e:
        print(f"⚠️ RAGAS execution failed: {e}")
        print(
            "Tip: RAGAS works best with GPT-4 class models. "
            "Consider swapping evaluator_llm for ChatGoogleGenerativeAI with Gemini."
        )


if __name__ == "__main__":
    run_evaluation()
