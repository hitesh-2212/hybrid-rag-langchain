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

load_dotenv()


# Inputs

FILE_PATH = "data/sample.pdf"
QUERY = "what is oops?"

# 1. Load documents

documents = []

try:
    ext = os.path.splitext(FILE_PATH)[1].lower()

    if ext == ".pdf":
        print("Loading PDF...")
        documents = PyPDFLoader(FILE_PATH).load()

    elif ext == ".txt":
        print("Loading TXT...")
        documents = TextLoader(FILE_PATH).load()

except Exception as e:
    print("Document loading failed:", e)
    documents = []


# 2. Chunking

print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# 3. Embeddings + Vector Store

print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building vector store...")
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
    text = "\n\n".join(doc.page_content for doc in docs)
    return text

parser = StrOutputParser()


# 7. PDF RAG Chain

pdf_chain = (
    RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
    | rag_prompt| llm | parser
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
            )| RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })| wiki_prompt| llm| parser)

# 9.Execution Logic

print("\nSearching PDF...\n")
answer = pdf_chain.invoke(QUERY)

if answer.strip().lower() == "not found in context":
    print("Answer not found in PDF → Falling back to Wikipedia...\n")
    answer = wiki_chain.invoke(QUERY)

print("Final Answer:\n")
print(answer)
