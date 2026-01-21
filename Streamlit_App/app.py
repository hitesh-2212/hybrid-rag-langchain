import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    WikipediaLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

# ----------------------------------
# Environment
# ----------------------------------
load_dotenv()
st.set_page_config(page_title="Hybrid RAG", layout="wide")

# ----------------------------------
# Custom UI Styling
# ----------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #22d3ee;
}
.subtitle {
    color: #94a3b8;
    margin-bottom: 25px;
}
.chat-card {
    background-color: #020617;
    border-radius: 14px;
    padding: 20px;
    margin-top: 20px;
    border: 1px solid #1e293b;
}
.answer-card {
    background-color: #020617;
    border-radius: 14px;
    padding: 20px;
    border-left: 6px solid #22d3ee;
}
.source-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 10px;
}
.pdf {
    background-color: #064e3b;
    color: #a7f3d0;
}
.wiki {
    background-color: #1e3a8a;
    color: #bfdbfe;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Header
# ----------------------------------
st.markdown('<div class="main-title">📄 Hybrid RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions from your PDF with intelligent Wikipedia fallback</div>',
    unsafe_allow_html=True
)

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.header("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file", type=["pdf"]
)

query = st.sidebar.text_input(
    "❓ Ask a question",
    placeholder="e.g. What problem does BERT solve?"
)

ask_button = st.sidebar.button("🚀 Ask")

# ----------------------------------
# Helpers
# ----------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parser = StrOutputParser()

# ----------------------------------
# Main Logic (UNCHANGED)
# ----------------------------------
if uploaded_file and query and ask_button:

    with st.spinner("🤔 Thinking..."):

        # Save uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            FILE_PATH = tmp.name

        # Load PDF
        documents = PyPDFLoader(FILE_PATH).load()

        # Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Embeddings + Vector Store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # LLM
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

        # Prompts
        rag_prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using ONLY the context below.
            If the answer is not present, respond with:
            "Not found in context"

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        wiki_prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using the Wikipedia context below.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        # Chains
        pdf_chain = (
            RunnableParallel(
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                }
            )
            | rag_prompt
            | llm
            | parser
        )

        wiki_chain = (
            RunnableParallel(
                {
                    "context": RunnableLambda(
                        lambda q: WikipediaLoader(
                            query=q,
                            load_max_docs=2
                        ).load()
                    )
                    | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                }
            )
            | wiki_prompt
            | llm
            | parser
        )

        # Execution Logic
        answer = pdf_chain.invoke(query)
        source = "pdf"

        if answer.strip().lower() == "not found in context":
            answer = wiki_chain.invoke(query)
            source = "wiki"

    # ----------------------------------
    # UI Output
    # ----------------------------------
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown(f"**🧑 You:** {query}")
    st.markdown("</div>", unsafe_allow_html=True)

    badge = "📄 PDF" if source == "pdf" else "🌐 Wikipedia"
    badge_class = "pdf" if source == "pdf" else "wiki"

    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.markdown(
        f'<span class="source-badge {badge_class}">{badge}</span>',
        unsafe_allow_html=True
    )
    st.markdown(answer)
    st.markdown("</div>", unsafe_allow_html=True)

elif ask_button:
    st.warning("⚠️ Please upload a PDF and enter a question.")
