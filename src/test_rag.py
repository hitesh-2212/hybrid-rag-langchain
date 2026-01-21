from pathlib import Path

from loaders import load_pdfs
from chunker import chunk_documents
from embeddings import get_embeddings
from vectorstore import index_documents, load_faiss_index
from rag_pipeline import generate_answer
from web_search import search_wikipedia



# 1️ Resolve base directory

BASE_DIR = Path(__file__).resolve().parent.parent


# 2️ Load PDF documents

pdf_dir = BASE_DIR / "data"          

docs = load_pdfs(pdf_dir)
print("Loaded docs:", len(docs))

# Add metadata
for d in docs:
    d.metadata["source_type"] = "pdf"
    d.metadata["title"] = d.metadata.get("source", "PDF Document")


# 3️ Chunk documents

chunks = chunk_documents(docs)
print("Total chunks:", len(chunks))


# 4️ Build FAISS index

embeddings = get_embeddings()
index_documents(chunks, embeddings)


# 5️ Load FAISS index

db = load_faiss_index(embeddings)


# 6️ Test document-based RAG

query = "What is object oriented programming?"
doc_results = db.similarity_search(query, k=3)

doc_answer, doc_sources = generate_answer(query, doc_results)

print("\n📄 DOCUMENT ANSWER")
print(doc_answer)


# 7️ Test Wikipedia-based RAG

wiki_docs = search_wikipedia("Object Oriented Programming")
wiki_answer, wiki_sources = generate_answer(query, wiki_docs)

print("\n🌐 WIKIPEDIA ANSWER")
print(wiki_answer)
