from langchain_community.document_loaders import PyPDFLoader, TextLoader
from typing import List


def load_pdfs(path: str):
    docs = []
    for file in path.glob("*.pdf"):
        loader = PyPDFLoader(str(file))
        docs.extend(loader.load())
    return docs


def load_texts(path: str):
    docs = []
    for file in path.glob("*.txt"):
        loader = TextLoader(str(file))
        docs.extend(loader.load())
    return docs
