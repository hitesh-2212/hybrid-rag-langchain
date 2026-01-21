import os
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
st.set_page_config(page_title="Hybrid RAG App", layout="wide")

st.title("📄 Hybrid RAG: PDF + Wikipedia")
st.caption("Your original RAG logic with a Streamlit interface")

# ----------------------------------
# Sidebar Inputs (UI added)
# ----------------------------------
st.sidebar.header("Inputs")

FILE_PATH = st.sidebar.text_input(
    "Enter file path (PDF or TXT)",
    value="data/sample.pdf"
)

QUERY = st.sidebar.text_input(
    "Enter your question",
    value="what is oops?"
)

run_button = st.sidebar.button("Run Query")

# ----------------------------------
# Run only when button is clicked
# ----------------------------------
if run_button:

    # 1. Load documents
    documents = []

    try:
        ext = os.path.splitext(FILE_PATH)[1].lower()

        if ext == ".pdf":
            st.write("📄 Loading PDF...")
            documents = PyPDFLoader(FILE_PATH).load()

        elif ext == ".txt":
            st.write("📄 Loading TXT...")
            documents = TextLoader(FILE_PATH).load()

        else:
            st.error("Unsupported file type")
            st.stop()

    except Exception as e:
        st.error(f"Document loading failed: {e}")
        st.stop()

    # 2. Chunking
    st.write("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    st.write(f"Total chunks created: {len(chunks)}")

    # 3. Embeddings + Vector Store
    st.write("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.write("📦 Building vector store...")
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

    # 9. Execution Logic (UNCHANGED)
    st.subheader("🔍 Result")

    with st.spinner("Searching PDF..."):
        answer = pdf_chain.invoke(QUERY)

    if answer.strip().lower() == "not found in context":
        st.info("Answer not found in PDF → Falling back to Wikipedia...")
        with st.spinner("Searching Wikipedia..."):
            answer = wiki_chain.invoke(QUERY)

    st.success("Final Answer")
    st.write(answer)

else:
    st.info("Enter inputs from the sidebar and click **Run Query**")
