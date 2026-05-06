from rag.pipeline import Query, load_hotpotqa_queries, load_nq_queries


def load_query_examples(
    dataset: str,
    start: int = 0,
    limit: int = 1000,
    split: str = "validation",
) -> list[Query]:
    if dataset == "hotpotqa":
        return load_hotpotqa_queries(start=start, limit=limit, split=split)
    if dataset == "nq":
        return load_nq_queries(start=start, limit=limit, split=split)
    raise ValueError(f"Unknown dataset: {dataset}")
