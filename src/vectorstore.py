from langchain_community.vectorstores import FAISS


def index_documents(chunks, embeddings, path="index/faiss_index"):
    if not chunks:
        raise RuntimeError(
            "No document chunks found. "
            "Check PDF path and PDF content."
        )
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(path)


def load_faiss_index(embeddings, path="index/faiss_index"):
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


