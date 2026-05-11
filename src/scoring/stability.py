from rag.pipeline import (
    Generator,
    Query,
    RetrievedDocument,
    build_diagnostic_perturbations,
    compute_anchoring_consistency,
)

from data.schemas import StabilityResult


def build_perturbations(
    query: Query,
    docs: list[RetrievedDocument],
    pool: list[RetrievedDocument],
) -> list[list[RetrievedDocument]]:
    _ = query
    candidate_ids = {doc.doc_id for doc in docs}
    replacement_candidates = [doc for doc in pool if doc.doc_id not in candidate_ids]
    return build_diagnostic_perturbations(docs, replacement_candidates)


def score_stability(
    query: Query,
    docs: list[RetrievedDocument],
    generator: Generator,
    perturbations: list[list[RetrievedDocument]],
    tail_level: float = 1.0,
) -> StabilityResult:
    consistency, diagnostic_generations, base_answer = compute_anchoring_consistency(
        query,
        docs,
        generator,
        perturbations,
        tail_level=tail_level,
    )
    return StabilityResult(
        consistency=consistency,
        variance=1.0 - consistency,
        base_answer=base_answer,
        diagnostic_generations=diagnostic_generations,
    )
