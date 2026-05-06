import hashlib

from data.schemas import PolicyDecision
from rag.pipeline import Query, RetrievedDocument

from .proposed_select import ProposedSelectPolicy


class RandomSelectPolicy(ProposedSelectPolicy):
    name = "random_select"

    def choose_candidate(
        self,
        query: Query,
        docs: list[RetrievedDocument],
        candidates: list[RetrievedDocument],
        f_score: float,
        c_score: float,
    ) -> PolicyDecision:
        digest = hashlib.sha1(query.text.encode("utf-8")).hexdigest()
        selected = candidates[int(digest, 16) % len(candidates)]
        score = self.score_one(query, docs, selected, f_score, c_score)
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
