from typing import Any

from rag.pipeline import Query, RetrievedDocument, SufficiencyEstimator, extract_evidence_features


def score_sufficiency(
    query: Query,
    docs: list[RetrievedDocument],
    estimator: SufficiencyEstimator,
    aspect_model: str = "BAAI/bge-small-en-v1.5",
) -> float:
    features = extract_evidence_features(query, docs, aspect_model=aspect_model)
    return estimator.predict(features).sufficiency_score


def score_sufficiency_with_features(
    query: Query,
    docs: list[RetrievedDocument],
    estimator: SufficiencyEstimator,
    aspect_model: str = "BAAI/bge-small-en-v1.5",
) -> tuple[float, Any]:
    features = extract_evidence_features(query, docs, aspect_model=aspect_model)
    decision = estimator.predict(features)
    return decision.sufficiency_score, features
