from langchain.tools import tool


vector_store = None


def set_vector_store(store):
    global vector_store
    vector_store = store


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    return f"{minutes:02d}:{seconds:02d}"


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the uploaded documents, audio, or YouTube
    transcription for information relevant to the user's question.
    """

    if vector_store is None:
        return "Knowledge base is not available."

    documents = vector_store.similarity_search(
        query,
        k=5
    )

    if not documents:
        return "No relevant information found."

    results = []

    for i, doc in enumerate(documents):

        metadata = doc.metadata

        start = metadata.get("start")
        end = metadata.get("end")
        source = metadata.get("source", "Unknown")
        chunk_id = metadata.get("chunk_id", "Unknown")

        if start is not None and end is not None:
            timestamp = (
                f"{format_time(start)} - "
                f"{format_time(end)}"
            )
        else:
            timestamp = "Timestamp unavailable"

        results.append(
            f"""
SOURCE {i + 1}

Text:
{doc.page_content}

Source:
{source}

Timestamp:
{timestamp}

Chunk:
{chunk_id}
"""
        )

    return "\n".join(results)