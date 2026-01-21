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
# Custom Styling (UI ONLY)
# ----------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(90deg, #4ade80, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 16px;
        color: #94a3b8;
        margin-bottom: 25px;
    }
    .answer-box {
        background-color: #0f172a;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #22d3ee;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">📄 Hybrid RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions from your PDF with automatic Wikipedia fallback</div>',
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
    placeholder="e.g. What is BERT?"
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

    with st.spinner("🔍 Reading and analyzing your document..."):

        # Save uploaded file
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

        # Execution
        answer = pdf_chain.invoke(query)

        if answer.strip().lower() == "not found in context":
            answer = wiki_chain.invoke(query)

    # ----------------------------------
    # Final Answer UI
    # ----------------------------------
    st.markdown("### ✅ Answer")
    st.markdown(
        f'<div class="answer-box">{answer}</div>',
        unsafe_allow_html=True
    )

elif ask_button:
    st.warning("⚠️ Please upload a PDF and enter a question.")
