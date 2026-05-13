from data.schemas import CandidateScore
from rag.pipeline import (
    Generator,
    Query,
    RetrievedDocument,
    SufficiencyEstimator,
    anchor_deficit,
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
    stability_threshold: float = 0.8,
    tail_level: float = 1.0,
    sufficiency_tolerance: float = 0.0,
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
    post_c_score, _, _ = compute_anchoring_consistency(
        query,
        selected_docs,
        generator,
        perturbations,
        tail_level=tail_level,
    )
    delta_f = post_f_score - base_f_score
    delta_c = post_c_score - base_c_score
    redundancy = compute_lexical_redundancy(candidate, docs)
    base_deficit = anchor_deficit(base_c_score, stability_threshold)
    post_deficit = anchor_deficit(post_c_score, stability_threshold)
    deficit_reduction = base_deficit - post_deficit
    feasible = post_f_score >= estimator.threshold - sufficiency_tolerance
    _ = (alpha, beta)
    utility = deficit_reduction - rho * redundancy
    return CandidateScore(
        doc_id=candidate.doc_id,
        delta_f=delta_f,
        delta_c=delta_c,
        redundancy=redundancy,
        utility=utility,
        post_consistency=post_c_score,
        anchor_deficit_reduction=deficit_reduction,
        base_anchor_deficit=base_deficit,
        post_anchor_deficit=post_deficit,
        feasible=feasible,
    )
