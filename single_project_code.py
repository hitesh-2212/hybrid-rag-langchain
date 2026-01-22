# 1. Imports

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (PyPDFLoader,TextLoader,WikipediaLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (RunnableParallel,RunnablePassthrough,RunnableLambda)
from langchain_core.output_parsers import StrOutputParser


# 2. Environment & Inputs

load_dotenv()

FILE_PATH = "data/sample.pdf"        # .pdf or .txt
QUERY = "What is delhi?"             


# 3. Load Documents 

documents = []
DOCUMENT_SOURCE = "pdf"

try:
    ext = os.path.splitext(FILE_PATH)[1].lower()

    if ext == ".pdf":
        print("Loading PDF document...")
        documents = PyPDFLoader(FILE_PATH).load()

    elif ext == ".txt":
        print("Loading text document...")
        documents = TextLoader(FILE_PATH, encoding="utf-8").load()

    else:
        raise ValueError("Unsupported file type")

except Exception as e:
    print("Document loading failed:", e)
    documents = []


# 4. Wikipedia Fallback (NO DOCUMENT FOUND)

if not documents:
    print("No local document found → Falling back to Wikipedia...")
    documents = WikipediaLoader(
        query=QUERY,
        load_max_docs=2
    ).load()
    DOCUMENT_SOURCE = "wikipedia"


# 5. Chunking

print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# 6. Embeddings + Vector Store

print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building vector store...")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 7. LLM

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# 8. Prompt

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


# 9. Helpers

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parser = StrOutputParser()


# 10. LCEL RAG Chain

rag_chain =RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

final_chain=rag_chain|rag_prompt|llm|parser

# 11. Run Query (PDF FIRST)

print("\nQuery:", QUERY)
print("Searching PDF...\n")

answer = final_chain.invoke(QUERY)


# 12. QUERY-LEVEL FALLBACK 

fallback_triggers = [
    "no information",
    "not present",
    "not found",
    "does not contain",
    "cannot find"
]

if any(trigger in answer.lower() for trigger in fallback_triggers):
    print("Query not answered by PDF → Falling back to Wikipedia...\n")

    # Load Wikipedia docs
    wiki_docs = WikipediaLoader(
        query=QUERY,
        load_max_docs=2
    ).load()

    # Chunk Wikipedia docs
    wiki_chunks = splitter.split_documents(wiki_docs)

    # Build Wiki vector store
    wiki_vectorstore = FAISS.from_documents(wiki_chunks, embeddings)
    wiki_retriever = wiki_vectorstore.as_retriever(search_kwargs={"k": 3})

    # Wiki RAG Chain
    wiki_chain = (
        RunnableParallel(
            {
                "context": wiki_retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
        )| rag_prompt| llm| parser )

    answer = wiki_chain.invoke(QUERY)
    DOCUMENT_SOURCE = "wikipedia"


# 13. Final Output

print(f"Answer Source: {DOCUMENT_SOURCE.upper()}\n")
print("Final Answer:\n")
print(answer)
