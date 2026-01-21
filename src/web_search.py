from langchain_community.document_loaders import WikipediaLoader


def search_wikipedia(query: str, max_docs: int = 3):
    loader = WikipediaLoader(
        query=query,
        load_max_docs=max_docs,
        doc_content_chars_max=2000
    )
    docs = loader.load()

    for d in docs:
        d.metadata["source_type"] = "wikipedia"

    return docs
