from data.schemas import CandidateScore, PolicyDecision
from rag.pipeline import Query, RetrievedDocument, extract_evidence_features
from scoring.stability import build_perturbations, score_stability
from scoring.utility import score_candidate

from .base import EvidencePolicy


class ProposedSelectPolicy(EvidencePolicy):
    name = "proposed_select"

    def score_one(
        self,
        query: Query,
        docs: list[RetrievedDocument],
        candidate: RetrievedDocument,
        f_score: float,
        c_score: float,
        replacement_candidates: list[RetrievedDocument] | None = None,
    ) -> CandidateScore:
        return score_candidate(
            query,
            docs,
            candidate,
            self.context.generator,
            self.context.estimator,
            base_f_score=f_score,
            base_c_score=c_score,
            alpha=self.context.utility_alpha,
            beta=self.context.utility_beta,
            rho=self.context.utility_rho,
            aspect_model=self.context.aspect_model,
            replacement_candidates=replacement_candidates,
        )

    def choose_candidate(
        self,
        query: Query,
        docs: list[RetrievedDocument],
        candidates: list[RetrievedDocument],
        f_score: float,
        c_score: float,
    ) -> PolicyDecision:
        scores = [
            score_candidate(
                query,
                docs,
                candidate,
                self.context.generator,
                self.context.estimator,
                base_f_score=f_score,
                base_c_score=c_score,
                alpha=self.context.utility_alpha,
                beta=self.context.utility_beta,
                rho=self.context.utility_rho,
                aspect_model=self.context.aspect_model,
                replacement_candidates=[other for other in candidates if other.doc_id != candidate.doc_id],
            )
            for candidate in candidates
        ]
        selected = max(scores, key=lambda item: item.utility)
        return PolicyDecision(
            action="SELECT",
            reason="sufficient_but_unstable",
            final_doc_ids=[doc.doc_id for doc in docs] + [selected.doc_id],
            chosen_doc_id=selected.doc_id,
            f_score=f_score,
            c_score=c_score,
            is_sbu=True,
            candidate_scores=scores,
            overhead_gens=2 * len(scores),
        )

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

        doc_ids = {doc.doc_id for doc in docs}
        candidates = [doc for doc in pool if doc.doc_id not in doc_ids]
        if not candidates:
            return PolicyDecision(
                action="STOP",
                reason="answer_sufficient_no_candidates",
                final_doc_ids=[doc.doc_id for doc in docs],
                f_score=decision.sufficiency_score,
                c_score=stability.consistency,
                overhead_gens=stability.diagnostic_generations,
            )
        selected = self.choose_candidate(query, docs, candidates, decision.sufficiency_score, stability.consistency)
        selected.overhead_gens += stability.diagnostic_generations
        return selected
