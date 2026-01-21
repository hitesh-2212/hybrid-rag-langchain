import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
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

st.title("📄 Hybrid RAG: PDF + Wikipedia")
st.caption("Ask questions from your PDF, with Wikipedia fallback")

# ----------------------------------
# Sidebar (UI ONLY)
# ----------------------------------
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF document", type=["pdf"]
)

query = st.sidebar.text_input(
    "Enter your question",
    placeholder="e.g. What is BERT?"
)

run_button = st.sidebar.button("Ask")

# ----------------------------------
# Run Logic (UNCHANGED)
# ----------------------------------
if uploaded_file and query and run_button:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        FILE_PATH = tmp.name

    # 1. Load document
    documents = PyPDFLoader(FILE_PATH).load()

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # 3. Embeddings + Vector Store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 4. LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # 5. Prompts
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

    # 6. Context formatter
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parser = StrOutputParser()

    # 7. PDF RAG Chain
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

    # 8. Wikipedia Chain
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

    # 9. Execution Logic (same as original)
    answer = pdf_chain.invoke(query)

    if answer.strip().lower() == "not found in context":
        answer = wiki_chain.invoke(query)

    # ----------------------------------
    # Final Output (UI CLEAN)
    # ----------------------------------
    st.subheader("✅ Answer")
    st.write(answer)

elif run_button:
    st.warning("Please upload a PDF and enter a question.")
