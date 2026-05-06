from rag.pipeline import Query, RetrievedDocument, Retriever


def retrieve(retriever: Retriever, query: Query, k_pool: int) -> list[RetrievedDocument]:
    return retriever.retrieve(query, top_k=k_pool)
