from data.schemas import PolicyDecision
from rag.pipeline import Query, RetrievedDocument, extract_evidence_features
from scoring.stability import build_perturbations, score_stability

from .base import EvidencePolicy


class DiagnoseExpandPolicy(EvidencePolicy):
    name = "diagnose_expand"

    def act(
        self,
        query: Query,
        docs: list[RetrievedDocument],
        pool: list[RetrievedDocument],
    ) -> PolicyDecision:
        features = extract_evidence_features(query, docs, aspect_model=self.context.aspect_model)
        decision = self.context.estimator.predict(features)
        if not decision.sufficient:
            expanded = pool[: self.context.expanded_k]
            return PolicyDecision(
                action="EXPAND",
                reason=decision.reason,
                final_doc_ids=[doc.doc_id for doc in expanded],
                f_score=decision.sufficiency_score,
            )

        stability = score_stability(
            query,
            docs,
            self.context.generator,
            build_perturbations(query, docs, pool),
        )
        if stability.consistency >= self.context.stability_threshold:
            return PolicyDecision(
                action="STOP",
                reason="answer_sufficient_stable",
                final_doc_ids=[doc.doc_id for doc in docs],
                f_score=decision.sufficiency_score,
                c_score=stability.consistency,
                overhead_gens=stability.diagnostic_generations,
            )
        expanded = pool[: self.context.expanded_k]
        return PolicyDecision(
            action="EXPAND",
            reason="sufficient_but_unstable",
            final_doc_ids=[doc.doc_id for doc in expanded],
            f_score=decision.sufficiency_score,
            c_score=stability.consistency,
            is_sbu=True,
            overhead_gens=stability.diagnostic_generations,
        )
