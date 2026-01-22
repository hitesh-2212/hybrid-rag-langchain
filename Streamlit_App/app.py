import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (PyPDFLoader,TextLoader, WikipediaLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (RunnableParallel,RunnablePassthrough,RunnableLambda)
from langchain_core.output_parsers import StrOutputParser


# Environment

load_dotenv()
st.set_page_config(page_title="Hybrid RAG", layout="wide")


# Styling

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
.answer-card {
    background-color: #020617;
    padding: 20px;
    border-radius: 14px;
    border-left: 6px solid #22d3ee;
    margin-top: 20px;
}
.badge {
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


# Header

st.markdown('<div class="main-title">📄 Hybrid RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions from your PDF with automatic Wikipedia fallback</div>',
    unsafe_allow_html=True
)


# Sidebar Inputs

st.sidebar.header("📂 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

query = st.sidebar.text_input(
    "❓ Ask a question",
    placeholder="e.g. What is BERT? / What is Delhi?"
)

run_button = st.sidebar.button("🚀 Ask")


# Helper

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parser = StrOutputParser()


# Main Execution

if uploaded_file and query and run_button:

    with st.spinner("🤔 Thinking..."):

        
        # Save uploaded file
        
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            FILE_PATH = tmp.name

        DOCUMENT_SOURCE = "pdf"
        documents = []

        
        # Load Documents 
       
        try:
            ext = os.path.splitext(FILE_PATH)[1].lower()

            if ext == ".pdf":
                documents = PyPDFLoader(FILE_PATH).load()

            elif ext == ".txt":
                documents = TextLoader(FILE_PATH, encoding="utf-8").load()

            else:
                raise ValueError("Unsupported file type")

        except Exception:
            documents = []

       
        # Document-level Wikipedia fallback
        
        if not documents:
            documents = WikipediaLoader(
                query=query,
                load_max_docs=2
            ).load()
            DOCUMENT_SOURCE = "wikipedia"

        
        # Chunking
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        
        # Embeddings + Vector Store
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        
        # LLM
        
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

        
        # Prompt
        
        rag_prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using ONLY the context below.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        
        # LCEL RAG Chain
        
        rag_chain = RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
        )

        final_chain = rag_chain | rag_prompt | llm | parser

        
        # Run Query (PDF first)
        
        answer = final_chain.invoke(query)

        
        # Query-level fallback 
        
        fallback_triggers = [
            "no information",
            "not present",
            "not found",
            "does not contain",
            "cannot find"
        ]

        if any(trigger in answer.lower() for trigger in fallback_triggers):

            wiki_docs = WikipediaLoader(
                query=query,
                load_max_docs=2
            ).load()

            wiki_chunks = splitter.split_documents(wiki_docs)
            wiki_vectorstore = FAISS.from_documents(wiki_chunks, embeddings)
            wiki_retriever = wiki_vectorstore.as_retriever(search_kwargs={"k": 3})

            wiki_chain = (
                RunnableParallel(
                    {
                        "context": wiki_retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough()
                    }
                ) | rag_prompt | llm | parser
            )

            answer = wiki_chain.invoke(query)
            DOCUMENT_SOURCE = "wikipedia"

    
    # Output UI
    
    badge_class = "pdf" if DOCUMENT_SOURCE == "pdf" else "wiki"
    badge_text = "📄 PDF" if DOCUMENT_SOURCE == "pdf" else "🌐 Wikipedia"

    st.markdown(f'<span class="badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.write(answer)
    st.markdown('</div>', unsafe_allow_html=True)

elif run_button:
    st.warning("Please upload a document and enter a question.")
