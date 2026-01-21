import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------
# Environment
# -------------------------------------------------
load_dotenv()

st.set_page_config(page_title="Hybrid RAG: PDF + Wikipedia", layout="wide")

st.title("📄 Hybrid RAG: PDF + Wikipedia")
st.caption("Search uploaded PDF first, fallback to Wikipedia only if needed")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF file", type=["pdf"]
)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def format_docs(docs):
    """Safely format retrieved documents for the LLM"""
    text = "\n\n".join(doc.page_content for doc in docs)
    return text[:3000]  # token-safe for Groq

parser = StrOutputParser()

# -------------------------------------------------
# PDF Processing (cached)
# -------------------------------------------------
@st.cache_resource
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        file_path = tmp.name

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    return retriever


# -------------------------------------------------
# Main App Logic
# -------------------------------------------------
if uploaded_file:
    with st.spinner("Processing PDF..."):
        retriever = process_pdf(uploaded_file)

    st.success("PDF processed successfully!")

    query = st.text_input("Ask a question from the document:")

    if query:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

        # ---------------- PDF Prompt ----------------
        pdf_prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using the context below.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        # ---------------- Wikipedia Prompt ----------------
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

        # ---------------- PDF RAG Chain ----------------
        pdf_chain = (
            RunnableParallel(
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                }
            )
            | pdf_prompt
            | llm
            | parser
        )

        # ---------------- Wikipedia Chain ----------------
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

        # ---------------- Hybrid Decision Logic ----------------
        with st.spinner("Searching PDF..."):
            retrieved_docs = retriever.invoke(query)

        if len(retrieved_docs) == 0:
            st.info("No relevant content found in PDF. Searching Wikipedia...")
            with st.spinner("Searching Wikipedia..."):
                answer = wiki_chain.invoke(query)
        else:
            with st.spinner("Answering from PDF..."):
                answer = pdf_chain.invoke(query)

        st.subheader("Answer")
        st.write(answer)

else:
    st.info("Please upload a PDF to begin.")
