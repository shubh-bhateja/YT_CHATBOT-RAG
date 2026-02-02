import streamlit as st
import re
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader

# --- Custom CSS for background, video, and style ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #e0eafc 0%, #cfdef3 100%) !important;
        }
        .main {
            background: transparent !important;
        }
        .video-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1.5em;
        }
        .video-container iframe, .video-container video {
            border-radius: 18px;
            box-shadow: 0 4px 24px rgba(60,60,120,0.12);
            max-width: 600px;
            width: 100%;
            height: 340px;
        }
        .stButton>button {
            color: white;
            background: #6366f1;
            border-radius: 8px;
            font-size: 18px;
            padding: 0.5em 2em;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
        }
        .stSelectbox>div>div>div {
            border-radius: 8px;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="YouTube Chatbot - RAG Arch.", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align:center; color:#6366f1;'>🤖 YouTube Chatbot <span style='font-size:0.6em;'>- RAG Arch.</span></h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Ask questions about any YouTube video transcript!</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter a YouTube video link and your question below. The chatbot will answer using the video transcript. 🎥💬</p>", unsafe_allow_html=True)

with st.expander("ℹ️ How does it work?"):
    st.write(
        "This chatbot fetches the transcript of the YouTube video, splits it into chunks, "
        "embeds them, and retrieves the most relevant parts to answer your question using a powerful language model."
    )

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

video_url = st.text_input("🔗 Paste YouTube Video URL", placeholder="e.g. https://www.youtube.com/watch?v=Gfr50f6ZBvo")

video_id = extract_video_id(video_url)
if video_id:
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    st.video(f"https://www.youtube.com/watch?v={video_id}")
    st.markdown('</div>', unsafe_allow_html=True)

question = st.text_input("❓ Ask a question about the video", placeholder="Type your question here...")

explanation_type = st.selectbox(
    "Choose explanation type:",
    ["Brief", "Medium", "In-depth"],
    index=1
)

explanation_instructions = {
    "Brief": "Give a concise answer (1-2 sentences).",
    "Medium": "Give a moderately detailed answer (3-5 sentences).",
    "In-depth": "Give a thorough, detailed answer with explanations and examples if possible (at least 6 sentences)."
}

if st.button("🚀 Get Answer") and video_url and question:
    with st.spinner("⏳ Processing video and generating answer..."):
        try:
            loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
            docs = loader.load()
            transcript = docs[0].page_content

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

            embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embedding)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            retrieved_docs = retriever.invoke(question)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}

                Explanation style: {explanation_instruction}
                """,
                input_variables=['context', 'question', 'explanation_instruction']
            )

            llm = HuggingFaceEndpoint(
                repo_id='mistralai/Mistral-7B-Instruct-v0.2',
                task='text-generation'
            )
            model = ChatHuggingFace(llm=llm)

            final_prompt = prompt.invoke({
                "context": context_text,
                "question": question,
                "explanation_instruction": explanation_instructions[explanation_type]
            })
            answer = model.invoke(final_prompt)

            st.success("✅ Here is your answer:")
            st.markdown(f"### 💡 {answer.content}")

            with st.expander("📄 Show retrieved transcript context"):
                st.write(context_text)

            with st.expander("📝 Show full transcript"):
                st.write(transcript)

        except Exception as e:
            st.error(f"❌ Error: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color: #6366f1;'>"
    "Made with ❤️ using LangChain, HuggingFace, and Streamlit"
    "</div>",
    unsafe_allow_html=True
)