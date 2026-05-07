from data.schemas import PolicyDecision
from rag.pipeline import Query, RetrievedDocument

from .proposed_select import ProposedSelectPolicy


class NextRankedSelectPolicy(ProposedSelectPolicy):
    name = "next_ranked"

    def choose_candidate(
        self,
        query: Query,
        docs: list[RetrievedDocument],
        candidates: list[RetrievedDocument],
        f_score: float,
        c_score: float,
    ) -> PolicyDecision:
        selected = candidates[0]
        score = self.score_one(
            query,
            docs,
            selected,
            f_score,
            c_score,
            replacement_candidates=[candidate for candidate in candidates if candidate.doc_id != selected.doc_id],
        )
        return PolicyDecision(
            action="SELECT",
            reason="sufficient_but_unstable",
            final_doc_ids=[doc.doc_id for doc in docs] + [selected.doc_id],
            chosen_doc_id=selected.doc_id,
            f_score=f_score,
            c_score=c_score,
            is_sbu=True,
            candidate_scores=[score],
            overhead_gens=2,
        )
