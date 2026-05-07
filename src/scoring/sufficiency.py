from dataclasses import dataclass
from typing import Any

from rag.pipeline import (
    EvidenceFeatures,
    Query,
    RetrievedDocument,
    SufficiencyEstimator,
    clamp_unit,
    compute_lexical_redundancy,
    compute_pairwise_similarities,
    estimate_coverage,
    estimate_supportiveness,
    min_max_normalize,
)


@dataclass
class SufficiencyComponents:
    relevance: float
    coverage: float
    supportiveness: float
    redundancy: float

    def to_features(self) -> EvidenceFeatures:
        return EvidenceFeatures(
            relevance=self.relevance,
            redundancy=self.redundancy,
            coverage=self.coverage,
            supportiveness=self.supportiveness,
        )


@dataclass
class SufficiencyWeights:
    w_rel: float
    w_cov: float
    w_sup: float
    w_red: float


class ScoreNormalizer:
    def __init__(self, stats: dict[str, tuple[float, float]] | None = None) -> None:
        self.stats = stats or {}

    def normalize(self, name: str, value: float) -> float:
        if name not in self.stats:
            return clamp_unit(value)
        min_v, max_v = self.stats[name]
        if max_v <= min_v:
            return 0.0
        return clamp_unit((value - min_v) / (max_v - min_v))


class LightweightSufficiencyScorer:
    """
    Deterministic STAR-style scorer shared by all policies.
    It performs no LLM judge calls and returns component scores in [0, 1].
    """

    def __init__(
        self,
        weights: SufficiencyWeights,
        normalizer: ScoreNormalizer | None = None,
        aspect_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.weights = weights
        self.normalizer = normalizer or ScoreNormalizer()
        self.aspect_model = aspect_model

    @classmethod
    def from_estimator(
        cls,
        estimator: SufficiencyEstimator,
        normalizer: ScoreNormalizer | None = None,
        aspect_model: str = "BAAI/bge-small-en-v1.5",
    ) -> "LightweightSufficiencyScorer":
        return cls(
            SufficiencyWeights(
                w_rel=estimator.relevance_weight,
                w_cov=estimator.coverage_weight,
                w_sup=estimator.supportiveness_weight,
                w_red=estimator.redundancy_weight,
            ),
            normalizer=normalizer,
            aspect_model=aspect_model,
        )

    def score_relevance(self, query: Query, docs: list[RetrievedDocument]) -> float:
        _ = query
        raw_scores = [doc.retrieval_score for doc in docs]
        normalized_scores = min_max_normalize(raw_scores)
        return self.normalizer.normalize("relevance", sum(normalized_scores) / max(len(normalized_scores), 1))

    def score_coverage(self, query: Query, docs: list[RetrievedDocument]) -> float:
        return self.normalizer.normalize("coverage", estimate_coverage(query, docs, model_name=self.aspect_model))

    def score_supportiveness(self, query: Query, docs: list[RetrievedDocument]) -> float:
        raw_scores = [doc.retrieval_score for doc in docs]
        normalized_scores = min_max_normalize(raw_scores)
        if not normalized_scores:
            return 0.0
        return self.normalizer.normalize("supportiveness", estimate_supportiveness(query, docs, normalized_scores))

    def score_redundancy(self, docs: list[RetrievedDocument]) -> float:
        if len(docs) < 2:
            return 0.0
        pairwise_sims = compute_pairwise_similarities(docs)
        semantic_redundancy = (sum(pairwise_sims) / len(pairwise_sims) + 1.0) / 2.0 if pairwise_sims else 0.0
        lexical_redundancy = max(compute_lexical_redundancy(doc, docs[:idx] + docs[idx + 1 :]) for idx, doc in enumerate(docs))
        return self.normalizer.normalize("redundancy", max(semantic_redundancy, lexical_redundancy))

    def score_components(self, query: Query, docs: list[RetrievedDocument]) -> SufficiencyComponents:
        if not docs:
            return SufficiencyComponents(0.0, 0.0, 0.0, 1.0)
        return SufficiencyComponents(
            relevance=self.score_relevance(query, docs),
            coverage=self.score_coverage(query, docs),
            supportiveness=self.score_supportiveness(query, docs),
            redundancy=self.score_redundancy(docs),
        )

    def score(self, query: Query, docs: list[RetrievedDocument]) -> float:
        components = self.score_components(query, docs)
        return (
            self.weights.w_rel * components.relevance
            + self.weights.w_cov * components.coverage
            + self.weights.w_sup * components.supportiveness
            - self.weights.w_red * components.redundancy
        )


def score_sufficiency(
    query: Query,
    docs: list[RetrievedDocument],
    estimator: SufficiencyEstimator,
    aspect_model: str = "BAAI/bge-small-en-v1.5",
) -> float:
    scorer = LightweightSufficiencyScorer.from_estimator(estimator, aspect_model=aspect_model)
    return scorer.score(query, docs)


def score_sufficiency_with_features(
    query: Query,
    docs: list[RetrievedDocument],
    estimator: SufficiencyEstimator,
    aspect_model: str = "BAAI/bge-small-en-v1.5",
) -> tuple[float, Any]:
    scorer = LightweightSufficiencyScorer.from_estimator(estimator, aspect_model=aspect_model)
    features = scorer.score_components(query, docs).to_features()
    decision = estimator.predict(features)
    return decision.sufficiency_score, features
