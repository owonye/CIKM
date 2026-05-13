from __future__ import annotations

from dataclasses import dataclass

from data.schemas import PolicyDecision
from rag.pipeline import Generator, Query, RetrievedDocument, SufficiencyEstimator


@dataclass
class PolicyContext:
    generator: Generator
    estimator: SufficiencyEstimator
    initial_k: int = 3
    expanded_k: int = 8
    stability_threshold: float = 0.8
    utility_rho: float = 0.1
    aspect_model: str = "BAAI/bge-small-en-v1.5"


class EvidencePolicy:
    name = "base_policy"

    def __init__(self, context: PolicyContext) -> None:
        self.context = context

    def act(
        self,
        query: Query,
        docs: list[RetrievedDocument],
        pool: list[RetrievedDocument],
    ) -> PolicyDecision:
        raise NotImplementedError
