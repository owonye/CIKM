from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QueryExample:
    query_id: str
    dataset: str
    question: str
    gold_answers: list[str] = field(default_factory=list)


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    rank: int


@dataclass
class CandidateScore:
    doc_id: str
    delta_f: float
    delta_c: float
    redundancy: float
    utility: float
    post_consistency: float | None = None


@dataclass
class StabilityResult:
    consistency: float
    variance: float
    base_answer: str
    perturbation_answers: list[str] = field(default_factory=list)
    diagnostic_generations: int = 0


@dataclass
class PolicyDecision:
    action: str
    reason: str
    final_doc_ids: list[str]
    chosen_doc_id: str | None = None
    f_score: float | None = None
    c_score: float | None = None
    is_sbu: bool = False
    candidate_scores: list[CandidateScore] = field(default_factory=list)
    overhead_gens: int = 0


@dataclass
class RunRecord:
    query_id: str
    dataset: str
    gold_answer: str
    initial_doc_ids: list[str]
    pool_doc_ids: list[str]
    f_score: float | None
    c_score: float | None
    stop_by_sufficiency: bool
    is_sbu: bool
    action: str
    chosen_doc_id: str | None
    final_doc_ids: list[str]
    prediction: str
    em: float | None
    f1: float | None
    consistency_pre: float | None
    consistency_post: float | None
    variance_post: float | None
    overhead_gens: int
    avg_docs: float
    candidate_scores: list[CandidateScore] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_scores"] = [asdict(score) for score in self.candidate_scores]
        return payload
