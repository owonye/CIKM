from data.schemas import CandidateScore
from rag.pipeline import (
    Generator,
    Query,
    RetrievedDocument,
    SufficiencyEstimator,
    build_fixed_candidate_perturbations,
    compute_anchoring_consistency,
    compute_lexical_redundancy,
    extract_evidence_features,
)


def score_candidate(
    query: Query,
    docs: list[RetrievedDocument],
    candidate: RetrievedDocument,
    generator: Generator,
    estimator: SufficiencyEstimator,
    base_f_score: float,
    base_c_score: float,
    alpha: float,
    beta: float,
    rho: float,
    aspect_model: str = "BAAI/bge-small-en-v1.5",
    replacement_candidates: list[RetrievedDocument] | None = None,
) -> CandidateScore:
    selected_docs = docs + [candidate]
    features = extract_evidence_features(query, selected_docs, aspect_model=aspect_model)
    post_f_score = estimator.predict(features).sufficiency_score
    perturbations = build_fixed_candidate_perturbations(
        selected_docs,
        replacement_candidates=replacement_candidates,
    )
    post_c_score, _, _ = compute_anchoring_consistency(query, selected_docs, generator, perturbations)
    delta_f = post_f_score - base_f_score
    delta_c = post_c_score - base_c_score
    redundancy = compute_lexical_redundancy(candidate, docs)
    utility = alpha * delta_f + beta * delta_c - rho * redundancy
    return CandidateScore(
        doc_id=candidate.doc_id,
        delta_f=delta_f,
        delta_c=delta_c,
        redundancy=redundancy,
        utility=utility,
        post_consistency=post_c_score,
    )
