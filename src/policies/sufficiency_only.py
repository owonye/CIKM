from data.schemas import PolicyDecision
from rag.pipeline import Query, RetrievedDocument, extract_evidence_features

from .base import EvidencePolicy


class SufficiencyOnlyPolicy(EvidencePolicy):
    name = "sufficiency_only"

    def act(
        self,
        query: Query,
        docs: list[RetrievedDocument],
        pool: list[RetrievedDocument],
    ) -> PolicyDecision:
        _ = pool
        features = extract_evidence_features(query, docs, aspect_model=self.context.aspect_model)
        decision = self.context.estimator.predict(features)
        if decision.sufficient:
            return PolicyDecision(
                action="STOP",
                reason=decision.reason,
                final_doc_ids=[doc.doc_id for doc in docs],
                f_score=decision.sufficiency_score,
            )
        expanded = pool[: self.context.expanded_k]
        return PolicyDecision(
            action="EXPAND",
            reason=decision.reason,
            final_doc_ids=[doc.doc_id for doc in expanded],
            f_score=decision.sufficiency_score,
        )
