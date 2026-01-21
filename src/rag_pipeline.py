from typing import List
from langchain_core.documents import Document

from langchain_groq import ChatGroq


def get_llm():
    return ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)




def build_context(documents: List[Document]) -> str:
    """
    Combine retrieved document chunks into a single context string.
    """
    return "\n\n".join(doc.page_content for doc in documents)


def generate_answer(
    query: str,
    documents: List[Document]
):
    """
    Core RAG function:
    - Builds context
    - Sends context + question to Groq LLM
    - Returns answer + sources
    """

    llm = get_llm()
    context = build_context(documents)

    prompt = f"""
You are an AI assistant.
Answer the question strictly using the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    sources = []
    for doc in documents:
        sources.append({
            "source_type": doc.metadata.get("source_type", "document"),
            "title": doc.metadata.get("title", "Unknown"),
            "snippet": doc.page_content[:200]
        })

    return response.content, sources


