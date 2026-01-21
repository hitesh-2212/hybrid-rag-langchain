def route_query(query: str):
    keywords = ["latest", "current", "recent", "today"]
    if any(k in query.lower() for k in keywords):
        return "wikipedia"
    return "document"
