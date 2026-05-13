import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from experiment_utils import load_manifest, set_global_seed, write_run_config
from rag.pipeline import (
    FaissRetriever,
    GENERATOR_PROMPT_VERSION,
    HFGenerator,
    OpenAIGenerator,
    Query,
    SimpleGenerator,
    SimpleRetriever,
    StabilityAwareEvidenceSelector,
    StructureAwareAdaptiveRAG,
    SufficiencyEstimator,
    build_demo_corpus,
    compute_query_overlap,
    compute_retrieval_confidence,
    embed_corpus_texts,
    load_hotpotqa_queries,
    load_hotpotqa_sample,
    load_musique_queries,
    load_musique_sample,
    load_nq_queries,
    load_nq_sample,
    load_triviaqa_queries,
    load_triviaqa_sample,
)
from scoring.sufficiency import LightweightSufficiencyScorer


VALID_BASELINES = {
    "vanilla_rag",
    "fixed_large_k_rag",
    "confidence_adaptive_rag",
    "structure_aware_adaptive_rag",
    "diagnose_then_expand",
    "random_selection",
    "next_ranked_selection",
    "stability_aware_selection",
    "oracle_best_candidate",
    "selection_no_redundancy",
    "selection_mean_consistency",
    "selection_no_filter",
}


def _json_safe(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def parse_baselines(raw: str) -> list[str]:
    baselines = [token.strip() for token in raw.split(",") if token.strip()]
    if not baselines:
        raise ValueError("At least one baseline must be provided.")
    unknown = [name for name in baselines if name not in VALID_BASELINES]
    if unknown:
        raise ValueError(f"Unknown baseline(s): {unknown}")
    return baselines


def build_resources(args: argparse.Namespace):
    if args.mode == "demo":
        corpus = build_demo_corpus()
        queries = [Query("When is the birthday of Michael Phelps?", answer="June 30, 1985", answers=["June 30, 1985"])]
        simple_retriever = SimpleRetriever(corpus)
        faiss_retriever = simple_retriever
    elif args.mode in {"hotpotqa", "musique", "triviaqa"}:
        sample_loaders = {
            "hotpotqa": load_hotpotqa_sample,
            "musique": load_musique_sample,
            "triviaqa": load_triviaqa_sample,
        }
        query_loaders = {
            "hotpotqa": load_hotpotqa_queries,
            "musique": load_musique_queries,
            "triviaqa": load_triviaqa_queries,
        }
        raw_docs = sample_loaders[args.mode](start=args.doc_start, limit=args.doc_limit, split=args.corpus_split)
        cache_namespace = f"{args.mode}::{args.corpus_split}::{args.doc_start}:{args.doc_start + args.doc_limit}"
        corpus = embed_corpus_texts(
            raw_docs,
            model_name=args.embedding_model,
            cache_dir=args.retrieval_cache_dir,
            cache_namespace=cache_namespace,
        )
        queries = query_loaders[args.mode](start=args.query_start, limit=args.query_limit, split=args.query_split)
        simple_retriever = FaissRetriever(
            corpus,
            model_name=args.embedding_model,
            cache_dir=args.retrieval_cache_dir,
            cache_namespace=cache_namespace,
        )
        faiss_retriever = simple_retriever
    else:
        raw_docs = load_nq_sample(
            start=args.doc_start,
            limit=args.doc_limit,
            split=args.corpus_split,
            max_tokens=args.nq_max_tokens,
            stride=args.nq_stride,
        )
        cache_namespace = (
            f"nq::{args.corpus_split}::{args.doc_start}:{args.doc_start + args.doc_limit}"
            f"::chunk_{args.nq_max_tokens}_{args.nq_stride}"
        )
        corpus = embed_corpus_texts(
            raw_docs,
            model_name=args.embedding_model,
            cache_dir=args.retrieval_cache_dir,
            cache_namespace=cache_namespace,
        )
        queries = load_nq_queries(start=args.query_start, limit=args.query_limit, split=args.query_split)
        simple_retriever = FaissRetriever(
            corpus,
            model_name=args.embedding_model,
            cache_dir=args.retrieval_cache_dir,
            cache_namespace=cache_namespace,
        )
        faiss_retriever = simple_retriever

    if args.mode != "demo" and not args.use_openai and not args.hf_model_id and not args.allow_simple_generator:
        raise ValueError(
            "Non-demo evaluation requires real QA generation. Use --use-openai, --hf-model-id, or pass --allow-simple-generator explicitly."
        )

    if args.use_openai:
        generator = OpenAIGenerator(model=args.openai_model, cache_path=args.openai_cache_path)
    elif args.hf_model_id:
        generator = HFGenerator(model_id=args.hf_model_id, max_new_tokens=args.hf_max_new_tokens)
    else:
        generator = SimpleGenerator()

    return corpus, queries, simple_retriever, faiss_retriever, generator


def resolve_manifest_overrides(args: argparse.Namespace) -> argparse.Namespace:
    manifest = load_manifest(args.manifest_path)
    if not manifest:
        args.manifest_id = None
        return args
    args.manifest_id = manifest.get("manifest_id")
    args.doc_start = int(manifest["doc_start"])
    args.doc_limit = int(manifest["doc_limit"])
    args.corpus_split = str(manifest["corpus_split"])
    args.query_start = int(manifest["eval_query_start"])
    args.query_limit = int(manifest["eval_query_limit"])
    args.query_split = str(manifest["query_split"])
    args.initial_k = int(manifest["initial_k"])
    args.expanded_k = int(manifest["expanded_k"])
    args.candidate_pool_k = int(manifest.get("candidate_pool_k", args.candidate_pool_k))
    args.stability_threshold = float(manifest.get("stability_threshold", args.stability_threshold))
    args.tail_level = float(manifest.get("tail_level", args.tail_level))
    args.sufficiency_tolerance = float(manifest.get("sufficiency_tolerance", args.sufficiency_tolerance))
    args.utility_rho = float(manifest.get("utility_rho", getattr(args, "utility_rho", 0.1)))
    args.embedding_model = str(manifest["embedding_model"])
    args.seed = int(manifest["seed"])
    args.retrieval_cache_dir = str(manifest.get("retrieval_cache_dir", args.retrieval_cache_dir))
    args.nq_max_tokens = int(manifest.get("nq_max_tokens", args.nq_max_tokens))
    args.nq_stride = int(manifest.get("nq_stride", args.nq_stride))
    return args


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def exact_match_score(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evidence_has_weak_support(query: Query, docs, overlap_threshold: float) -> bool:
    gold_answers = query.answers if query.answers else ([query.answer] if query.answer else [])
    if not gold_answers:
        return False
    normalized_gold = [normalize_answer(answer) for answer in gold_answers if answer]
    normalized_gold = [answer for answer in normalized_gold if answer]
    if not normalized_gold:
        return False
    for doc in docs:
        doc_text = normalize_answer(doc.text)
        has_match = any(gold in doc_text for gold in normalized_gold)
        if has_match and compute_query_overlap(query, doc) >= overlap_threshold:
            return True
    return False


def get_oracle_support(
    query: Query,
    retriever,
    initial_k: int,
    expanded_k: int,
    overlap_threshold: float,
) -> dict[str, Any]:
    initial_docs = retriever.retrieve(query, top_k=initial_k)
    expanded_docs = retriever.retrieve(query, top_k=expanded_k)
    initial_support = evidence_has_weak_support(query, initial_docs, overlap_threshold=overlap_threshold)
    expanded_support = evidence_has_weak_support(query, expanded_docs, overlap_threshold=overlap_threshold)
    oracle_should_expand = (not initial_support) and expanded_support
    oracle_has_signal = initial_support or expanded_support
    return {
        "oracle_initial_support": initial_support,
        "oracle_expanded_support": expanded_support,
        "oracle_should_expand": oracle_should_expand,
        "oracle_has_signal": oracle_has_signal,
    }


def add_oracle_metrics(row: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    row = row.copy()
    row.update(oracle)
    if not oracle["oracle_has_signal"]:
        row["decision_correct"] = None
        row["decision_error_type"] = "no_oracle_support"
        return row

    if row["decision"] in {"answer_now", "fixed_retrieve"}:
        decision_expand = False
    elif row["decision"] in {"retrieve_more", "fixed_retrieve_more", "select_evidence"}:
        decision_expand = True
    else:
        decision_expand = None

    if decision_expand is None:
        row["decision_correct"] = None
        row["decision_error_type"] = "fixed_policy"
    elif decision_expand == oracle["oracle_should_expand"]:
        row["decision_correct"] = 1
        row["decision_error_type"] = "correct"
    elif decision_expand:
        row["decision_correct"] = 0
        row["decision_error_type"] = "unnecessary_expand"
    else:
        row["decision_correct"] = 0
        row["decision_error_type"] = "premature_stop"
    return row


def add_metrics(row: dict[str, Any], query: Query, generator_type: str) -> dict[str, Any]:
    row = row.copy()
    gold_answers = query.answers if query.answers else ([query.answer] if query.answer else [])
    row["gold_answer"] = " || ".join(gold_answers) if gold_answers else None
    if generator_type == "simple_placeholder":
        row["exact_match"] = None
        row["f1"] = None
    elif gold_answers:
        row["exact_match"] = max(exact_match_score(row["answer"], gold) for gold in gold_answers)
        row["f1"] = max(f1_score(row["answer"], gold) for gold in gold_answers)
    else:
        row["exact_match"] = None
        row["f1"] = None
    return row


def load_estimator(args: argparse.Namespace) -> tuple[SufficiencyEstimator, str, dict[str, Any]]:
    estimator = SufficiencyEstimator()
    if not args.calibration_file:
        return estimator, args.structure_aware_label or "structure_aware_adaptive_rag", {}

    calibration_path = Path(args.calibration_file).resolve()
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    config = json.loads(calibration_path.read_text(encoding="utf-8"))
    estimator.update_parameters(
        relevance_weight=config["relevance_weight"],
        coverage_weight=config["coverage_weight"],
        supportiveness_weight=config["supportiveness_weight"],
        redundancy_weight=config["redundancy_weight"],
        threshold=config["threshold"],
    )
    if args.structure_aware_label:
        return estimator, args.structure_aware_label, config

    ablate_signal = config.get("ablate_signal", "none")
    if ablate_signal != "none":
        return estimator, f"structure_aware_wo_{ablate_signal}", config
    return estimator, "structure_aware_adaptive_rag", config


def load_confidence_threshold(args: argparse.Namespace) -> float:
    if not args.confidence_calibration_file:
        return args.confidence_threshold

    confidence_path = Path(args.confidence_calibration_file).resolve()
    if not confidence_path.exists():
        raise FileNotFoundError(f"Confidence calibration file not found: {confidence_path}")
    config = json.loads(confidence_path.read_text(encoding="utf-8"))
    threshold = config.get("threshold")
    if threshold is None:
        raise ValueError("Confidence calibration file missing 'threshold'.")
    return float(threshold)


def apply_stability_calibration(args: argparse.Namespace) -> argparse.Namespace:
    if not args.stability_calibration_file:
        return args
    calibration_path = Path(args.stability_calibration_file).resolve()
    if not calibration_path.exists():
        raise FileNotFoundError(f"Stability calibration file not found: {calibration_path}")
    config = json.loads(calibration_path.read_text(encoding="utf-8"))
    best = config.get("best", config)
    args.stability_threshold = float(best["stability_threshold"])
    args.tail_level = float(best.get("tail_level", args.tail_level))
    args.sufficiency_tolerance = float(best.get("sufficiency_tolerance", args.sufficiency_tolerance))
    args.utility_rho = float(best.get("utility_rho", getattr(args, "utility_rho", 0.1)))
    return args


def run_vanilla(query: Query, retriever, generator, top_k: int = 3) -> dict[str, Any]:
    docs = retriever.retrieve(query, top_k=top_k)
    answer = generator.generate(query, docs)
    return {
        "baseline": "vanilla_rag",
        "query": query.text,
        "decision": "fixed_retrieve",
        "reason": "fixed_policy",
        "used_docs": [doc.doc_id for doc in docs],
        "initial_doc_ids": [doc.doc_id for doc in docs],
        "final_doc_ids": [doc.doc_id for doc in docs],
        "expanded": False,
        "retrieval_calls": 1,
        "initial_doc_count": len(docs),
        "doc_count": len(docs),
        "final_k": len(docs),
        "sufficiency_score": None,
        "relevance": None,
        "redundancy": None,
        "coverage": None,
        "supportiveness": None,
        "answer": answer,
    }


def run_fixed_large_k(query: Query, retriever, generator, initial_k: int = 3, expanded_k: int = 5) -> dict[str, Any]:
    final_docs = retriever.retrieve(query, top_k=expanded_k)
    initial_docs = final_docs[:initial_k]
    answer = generator.generate(query, final_docs)
    return {
        "baseline": "fixed_large_k_rag",
        "query": query.text,
        "decision": "fixed_retrieve_more",
        "reason": "fixed_policy",
        "used_docs": [doc.doc_id for doc in final_docs],
        "initial_doc_ids": [doc.doc_id for doc in initial_docs],
        "final_doc_ids": [doc.doc_id for doc in final_docs],
        "expanded": True,
        "retrieval_calls": 1,
        "initial_doc_count": len(initial_docs),
        "doc_count": len(final_docs),
        "final_k": len(final_docs),
        "sufficiency_score": None,
        "relevance": None,
        "redundancy": None,
        "coverage": None,
        "supportiveness": None,
        "answer": answer,
    }


def run_confidence_baseline(
    query: Query,
    retriever,
    generator,
    initial_k: int = 3,
    expanded_k: int = 5,
    threshold: float = 0.88,
) -> dict[str, Any]:
    initial_docs = retriever.retrieve(query, top_k=initial_k)
    raw_scores = [doc.retrieval_score for doc in initial_docs]
    avg_score = sum(raw_scores) / max(len(raw_scores), 1)
    confidence_score = compute_retrieval_confidence(initial_docs)

    if confidence_score >= threshold:
        answer = generator.generate(query, initial_docs)
        return {
            "baseline": "confidence_adaptive_rag",
            "query": query.text,
            "decision": "answer_now",
            "reason": "high_confidence",
            "used_docs": [doc.doc_id for doc in initial_docs],
            "initial_doc_ids": [doc.doc_id for doc in initial_docs],
            "final_doc_ids": [doc.doc_id for doc in initial_docs],
            "expanded": False,
            "retrieval_calls": 1,
            "initial_doc_count": len(initial_docs),
            "doc_count": len(initial_docs),
            "final_k": len(initial_docs),
            "sufficiency_score": confidence_score,
            "relevance": avg_score,
            "redundancy": None,
            "coverage": None,
            "supportiveness": None,
            "answer": answer,
        }

    expanded_docs = retriever.retrieve(query, top_k=expanded_k)
    answer = generator.generate(query, expanded_docs)
    return {
        "baseline": "confidence_adaptive_rag",
        "query": query.text,
        "decision": "retrieve_more",
        "reason": "low_confidence",
        "used_docs": [doc.doc_id for doc in expanded_docs],
        "initial_doc_ids": [doc.doc_id for doc in initial_docs],
        "final_doc_ids": [doc.doc_id for doc in expanded_docs],
        "expanded": True,
        "retrieval_calls": 2,
        "initial_doc_count": len(initial_docs),
        "doc_count": len(expanded_docs),
        "final_k": len(expanded_docs),
        "sufficiency_score": confidence_score,
        "relevance": avg_score,
        "redundancy": None,
        "coverage": None,
        "supportiveness": None,
        "answer": answer,
    }


def run_structure_aware(
    query: Query,
    retriever,
    generator,
    estimator,
    initial_k: int = 3,
    expanded_k: int = 5,
    aspect_model: str = "BAAI/bge-small-en-v1.5",
    baseline_name: str = "structure_aware_adaptive_rag",
    sufficiency_scorer=None,
) -> dict[str, Any]:
    pipeline = StructureAwareAdaptiveRAG(
        retriever=retriever,
        generator=generator,
        estimator=estimator,
        initial_k=initial_k,
        expanded_k=expanded_k,
        aspect_model=aspect_model,
        sufficiency_scorer=sufficiency_scorer,
    )
    result = pipeline.answer(query)
    retrieval_calls = 1 if result["decision"] == "answer_now" else 2
    return {
        "baseline": baseline_name,
        "query": query.text,
        "decision": result["decision"],
        "reason": result["reason"],
        "used_docs": result["used_docs"],
        "initial_doc_ids": result["initial_doc_ids"],
        "final_doc_ids": result["final_doc_ids"],
        "expanded": result["expansion_triggered"],
        "retrieval_calls": retrieval_calls,
        "initial_doc_count": len(result["initial_doc_ids"]),
        "doc_count": len(result["used_docs"]),
        "final_k": result["final_doc_count"],
        "sufficiency_score": result["sufficiency_score"],
        "relevance": result["features"].relevance,
        "redundancy": result["features"].redundancy,
        "coverage": result["features"].coverage,
        "supportiveness": result["features"].supportiveness,
        "answer": result["answer"],
    }


def run_stability_aware_selection(
    query: Query,
    retriever,
    generator,
    estimator,
    initial_k: int = 3,
    expanded_k: int = 8,
    candidate_pool_k: int = 8,
    stability_threshold: float = 0.8,
    utility_rho: float = 0.1,
    tail_level: float = 1.0,
    sufficiency_tolerance: float = 0.0,
    enforce_sufficiency_filter: bool = True,
    aspect_model: str = "BAAI/bge-small-en-v1.5",
    baseline_name: str = "stability_aware_selection",
    selection_strategy: str = "utility",
    sufficiency_scorer=None,
) -> dict[str, Any]:
    pipeline = StabilityAwareEvidenceSelector(
        retriever=retriever,
        generator=generator,
        estimator=estimator,
        initial_k=initial_k,
        expanded_k=expanded_k,
        candidate_pool_k=candidate_pool_k,
        stability_threshold=stability_threshold,
        utility_rho=utility_rho,
        tail_level=tail_level,
        sufficiency_tolerance=sufficiency_tolerance,
        enforce_sufficiency_filter=enforce_sufficiency_filter,
        aspect_model=aspect_model,
        sufficiency_scorer=sufficiency_scorer,
    )
    result = pipeline.answer(query, selection_strategy=selection_strategy)
    retrieval_calls = 1 if result["decision"] in {"answer_now", "select_evidence"} else 2
    return {
        "baseline": baseline_name,
        "query": query.text,
        "decision": result["decision"],
        "reason": result["reason"],
        "used_docs": result["used_docs"],
        "initial_doc_ids": result["initial_doc_ids"],
        "final_doc_ids": result["final_doc_ids"],
        "expanded": result["expansion_triggered"],
        "retrieval_calls": retrieval_calls,
        "initial_doc_count": len(result["initial_doc_ids"]),
        "doc_count": len(result["used_docs"]),
        "final_k": result["final_doc_count"],
        "sufficiency_score": result["sufficiency_score"],
        "relevance": result["features"].relevance,
        "redundancy": result["features"].redundancy,
        "coverage": result["features"].coverage,
        "supportiveness": result["features"].supportiveness,
        "anchoring_consistency": result["anchoring_consistency"],
        "post_selection_consistency": result["post_selection_consistency"],
        "stability_gain": result["stability_gain"],
        "recovered": result["recovered"],
        "diagnostic_generations": result["diagnostic_generations"],
        "selected_doc_id": result["selected_doc_id"],
        "candidate_utility": result["candidate_utility"],
        "candidate_delta_sufficiency": result["candidate_delta_sufficiency"],
        "candidate_delta_consistency": result["candidate_delta_consistency"],
        "candidate_redundancy_penalty": result["candidate_redundancy_penalty"],
        "anchor_deficit_reduction": result["anchor_deficit_reduction"],
        "tail_level": result["tail_level"],
        "sufficiency_tolerance": result["sufficiency_tolerance"],
        "candidate_details": result["candidate_details"],
        "answer": result["answer"],
    }


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_rows = []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query_uid",
                "generator_type",
                "model_version",
                "baseline",
                "query",
                "decision",
                "reason",
                "expanded",
                "used_docs",
                "initial_doc_ids",
                "final_doc_ids",
                "retrieval_calls",
                "initial_doc_count",
                "doc_count",
                "final_k",
                "sufficiency_score",
                "relevance",
                "redundancy",
                "coverage",
                "supportiveness",
                "anchoring_consistency",
                "post_selection_consistency",
                "stability_gain",
                "recovered",
                "diagnostic_generations",
                "selected_doc_id",
                "candidate_utility",
                "candidate_delta_sufficiency",
                "candidate_delta_consistency",
                "candidate_redundancy_penalty",
                "anchor_deficit_reduction",
                "tail_level",
                "sufficiency_tolerance",
                "candidate_details",
                "oracle_initial_support",
                "oracle_expanded_support",
                "oracle_should_expand",
                "oracle_has_signal",
                "decision_correct",
                "decision_error_type",
                "label_strategy",
                "calibration_source",
                "gold_answer",
                "exact_match",
                "f1",
                "answer",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            row = row.copy()
            row["used_docs"] = "|".join(row["used_docs"])
            row["initial_doc_ids"] = "|".join(row["initial_doc_ids"])
            row["final_doc_ids"] = "|".join(row["final_doc_ids"])
            if isinstance(row.get("candidate_details"), list):
                row["candidate_details"] = json.dumps(
                    row["candidate_details"],
                    ensure_ascii=False,
                    default=_json_safe,
                )
            writer.writerow(row)
            serialized_rows.append(row)

    unstable_path = output_path.with_name(f"{output_path.stem}_unstable_only{output_path.suffix}")
    unstable_rows = [row for row in serialized_rows if row.get("reason") == "sufficient_but_unstable"]
    if serialized_rows:
        with unstable_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(serialized_rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(unstable_rows)

    candidate_path = output_path.with_name(f"{output_path.stem}_candidates{output_path.suffix}")
    candidate_rows: list[dict[str, Any]] = []
    candidate_fieldnames = [
        "query_id",
        "query_uid",
        "dataset_query",
        "baseline",
        "candidate_rank",
        "candidate_doc_id",
        "delta_f",
        "delta_c",
        "anchor_deficit_reduction",
        "base_anchor_deficit",
        "post_anchor_deficit",
        "redundancy",
        "utility",
        "post_consistency",
        "feasible",
        "selected",
        "action",
        "is_sbu",
    ]
    for row in serialized_rows:
        raw_details = row.get("candidate_details", "")
        if not raw_details:
            continue
        try:
            details = json.loads(raw_details)
        except json.JSONDecodeError:
            continue
        if not isinstance(details, list):
            continue
        for rank, detail in enumerate(details, start=1):
            candidate_rows.append(
                {
                    "query_id": row.get("query_id"),
                    "query_uid": row.get("query_uid"),
                    "dataset_query": row.get("query"),
                    "baseline": row.get("baseline"),
                    "candidate_rank": rank,
                    "candidate_doc_id": detail.get("candidate_doc_id"),
                    "delta_f": detail.get("delta_sufficiency"),
                    "delta_c": detail.get("delta_consistency"),
                    "anchor_deficit_reduction": detail.get("anchor_deficit_reduction"),
                    "base_anchor_deficit": detail.get("base_anchor_deficit"),
                    "post_anchor_deficit": detail.get("post_anchor_deficit"),
                    "redundancy": detail.get("redundancy_penalty"),
                    "utility": detail.get("utility"),
                    "post_consistency": detail.get("post_consistency"),
                    "feasible": detail.get("feasible"),
                    "selected": detail.get("candidate_doc_id") == row.get("selected_doc_id"),
                    "action": row.get("decision"),
                    "is_sbu": row.get("reason") == "sufficient_but_unstable",
                }
            )
    if serialized_rows:
        with candidate_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=candidate_fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(candidate_rows)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "hotpotqa", "musique", "nq", "triviaqa"], default="demo")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--allow-simple-generator", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--hf-model-id", default="")
    parser.add_argument("--hf-max-new-tokens", type=int, default=64)
    parser.add_argument("--openai-cache-path", default="results/openai_cache.jsonl")
    parser.add_argument("--retrieval-cache-dir", default="results/cache")
    parser.add_argument("--nq-max-tokens", type=int, default=220)
    parser.add_argument("--nq-stride", type=int, default=110)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-limit", type=int, default=20000)
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--query-split", default="validation")
    parser.add_argument("--initial-k", type=int, default=3)
    parser.add_argument("--expanded-k", type=int, default=5)
    parser.add_argument("--candidate-pool-k", type=int, default=8)
    parser.add_argument("--stability-threshold", type=float, default=0.8)
    parser.add_argument("--tail-level", type=float, default=1.0)
    parser.add_argument("--sufficiency-tolerance", type=float, default=0.0)
    parser.add_argument("--utility-rho", type=float, default=0.1)
    parser.add_argument("--weak-support-overlap-threshold", type=float, default=0.2)
    parser.add_argument("--confidence-threshold", type=float, default=0.88)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--confidence-calibration-file", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-file", default="")
    parser.add_argument("--stability-calibration-file", default="")
    parser.add_argument(
        "--baselines",
        default="vanilla_rag,fixed_large_k_rag,confidence_adaptive_rag,structure_aware_adaptive_rag",
    )
    parser.add_argument("--structure-aware-label", default="")
    parser.add_argument("--output", default="results/baseline_results.csv")
    args = parser.parse_args()
    args = resolve_manifest_overrides(args)
    args = apply_stability_calibration(args)
    if not args.use_openai and not args.hf_model_id and args.mode != "demo" and not args.allow_simple_generator:
        raise ValueError("For paper-grade EM/F1, run evaluate.py with --use-openai or --hf-model-id.")
    set_global_seed(args.seed)

    def _format_eta(seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        total = int(seconds)
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _maybe_log_progress(stage: str, completed: int, total: int, started_at: float, force: bool = False) -> None:
        if total <= 0:
            return
        interval = max(1, total // 20)  # ~5% step
        if not force and completed % interval != 0 and completed != total:
            return
        elapsed = max(time.time() - started_at, 1e-9)
        rate = completed / elapsed
        remaining = max(total - completed, 0)
        eta = remaining / rate if rate > 0 else 0.0
        progress = (completed / total) * 100
        print(
            f"[PROGRESS] {stage}: {completed}/{total} ({progress:.1f}%) "
            f"elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}"
        )

    _, queries, simple_retriever, _, generator = build_resources(args)
    generator_type = "openai" if args.use_openai else ("hf_local" if args.hf_model_id else "simple_placeholder")
    model_version = args.openai_model if args.use_openai else (args.hf_model_id if args.hf_model_id else "simple_placeholder")
    estimator, structure_aware_name, calibration_config = load_estimator(args)
    confidence_threshold = load_confidence_threshold(args)
    selected_baselines = set(parse_baselines(args.baselines))
    calibration_source = Path(args.calibration_file).name if args.calibration_file else "default"
    label_strategy = calibration_config.get("label_strategy", "default")
    feature_aspect_model = "" if args.mode == "demo" else args.embedding_model
    shared_sufficiency_scorer = LightweightSufficiencyScorer.from_estimator(
        estimator,
        aspect_model=feature_aspect_model,
    )

    rows: list[dict[str, Any]] = []
    total_queries = len(queries)
    stage_started = time.time()
    for query_id, query in enumerate(queries):
        oracle = get_oracle_support(
            query,
            simple_retriever,
            initial_k=args.initial_k,
            expanded_k=args.expanded_k,
            overlap_threshold=args.weak_support_overlap_threshold,
        )
        if "vanilla_rag" in selected_baselines:
            row = add_metrics(
                run_vanilla(query, simple_retriever, generator, top_k=args.initial_k),
                query,
                generator_type=generator_type,
            )
            row = add_oracle_metrics(row, oracle)
            row["query_id"] = query_id
            row["query_uid"] = query.query_id or f"{args.mode}::{args.query_split}::{args.query_start + query_id}"
            row["generator_type"] = generator_type
            row["model_version"] = model_version
            row["label_strategy"] = label_strategy
            row["calibration_source"] = calibration_source
            rows.append(row)
        if "fixed_large_k_rag" in selected_baselines:
            row = add_metrics(
                run_fixed_large_k(
                    query,
                    simple_retriever,
                    generator,
                    initial_k=args.initial_k,
                    expanded_k=args.expanded_k,
                ),
                query,
                generator_type=generator_type,
            )
            row = add_oracle_metrics(row, oracle)
            row["query_id"] = query_id
            row["query_uid"] = query.query_id or f"{args.mode}::{args.query_split}::{args.query_start + query_id}"
            row["generator_type"] = generator_type
            row["model_version"] = model_version
            row["label_strategy"] = label_strategy
            row["calibration_source"] = calibration_source
            rows.append(row)
        if "confidence_adaptive_rag" in selected_baselines:
            row = add_metrics(
                run_confidence_baseline(
                    query,
                    simple_retriever,
                    generator,
                    initial_k=args.initial_k,
                    expanded_k=args.expanded_k,
                    threshold=confidence_threshold,
                ),
                query,
                generator_type=generator_type,
            )
            row = add_oracle_metrics(row, oracle)
            row["query_id"] = query_id
            row["query_uid"] = query.query_id or f"{args.mode}::{args.query_split}::{args.query_start + query_id}"
            row["generator_type"] = generator_type
            row["model_version"] = model_version
            row["label_strategy"] = label_strategy
            row["calibration_source"] = (
                Path(args.confidence_calibration_file).name if args.confidence_calibration_file else "default"
            )
            rows.append(row)
        if "structure_aware_adaptive_rag" in selected_baselines:
            row = add_metrics(
                run_structure_aware(
                    query,
                    simple_retriever,
                    generator,
                    estimator,
                    initial_k=args.initial_k,
                    expanded_k=args.expanded_k,
                    aspect_model=feature_aspect_model,
                    baseline_name=structure_aware_name,
                    sufficiency_scorer=shared_sufficiency_scorer,
                ),
                query,
                generator_type=generator_type,
            )
            row = add_oracle_metrics(row, oracle)
            row["query_id"] = query_id
            row["query_uid"] = query.query_id or f"{args.mode}::{args.query_split}::{args.query_start + query_id}"
            row["generator_type"] = generator_type
            row["model_version"] = model_version
            row["label_strategy"] = label_strategy
            row["calibration_source"] = calibration_source
            rows.append(row)
        stability_baselines = {
            "diagnose_then_expand": ("diagnose_then_expand", args.utility_rho),
            "random_selection": ("random", args.utility_rho),
            "next_ranked_selection": ("next_ranked", args.utility_rho),
            "stability_aware_selection": ("utility", args.utility_rho),
            "oracle_best_candidate": ("oracle", args.utility_rho),
            "selection_no_redundancy": ("utility", 0.0),
            "selection_mean_consistency": ("utility", args.utility_rho),
            "selection_no_filter": ("utility", args.utility_rho),
        }
        for baseline_name, (strategy, utility_rho) in stability_baselines.items():
            if baseline_name not in selected_baselines:
                continue
            row = add_metrics(
                run_stability_aware_selection(
                    query,
                    simple_retriever,
                    generator,
                    estimator,
                    initial_k=args.initial_k,
                    expanded_k=args.expanded_k,
                    candidate_pool_k=args.candidate_pool_k,
                    stability_threshold=args.stability_threshold,
                    utility_rho=utility_rho,
                    tail_level=1.0 if baseline_name == "selection_mean_consistency" else args.tail_level,
                    sufficiency_tolerance=args.sufficiency_tolerance,
                    enforce_sufficiency_filter=baseline_name != "selection_no_filter",
                    aspect_model=feature_aspect_model,
                    baseline_name=baseline_name,
                    selection_strategy=strategy,
                    sufficiency_scorer=shared_sufficiency_scorer,
                ),
                query,
                generator_type=generator_type,
            )
            row = add_oracle_metrics(row, oracle)
            row["query_id"] = query_id
            row["query_uid"] = query.query_id or f"{args.mode}::{args.query_split}::{args.query_start + query_id}"
            row["generator_type"] = generator_type
            row["model_version"] = model_version
            row["label_strategy"] = label_strategy
            row["calibration_source"] = calibration_source
            rows.append(row)
        _maybe_log_progress("evaluate_queries", query_id + 1, total_queries, stage_started)

    output_path = Path(args.output)
    write_results(rows, output_path)
    write_run_config(
        output_path.with_suffix(".meta.json"),
        {
            "script": "evaluate.py",
            "args": vars(args),
            "num_queries": len(queries),
            "num_rows": len(rows),
            "generator_type": generator_type,
            "model_version": model_version,
            "openai_cache_stats": generator.get_cache_stats() if args.use_openai else None,
            "retrieval_cache_dir": args.retrieval_cache_dir,
            "manifest_path": args.manifest_path or None,
            "manifest_id": args.manifest_id,
            "prompt_template_version": GENERATOR_PROMPT_VERSION,
            "calibration_source": calibration_source,
            "confidence_calibration_source": (
                Path(args.confidence_calibration_file).name if args.confidence_calibration_file else "default"
            ),
            "output": str(output_path),
        },
    )

    print(f"Saved results to {output_path}")
    print(
        f"[DONE] evaluate_queries={total_queries} baselines={len(selected_baselines)} rows={len(rows)}"
    )
    for row in rows[:4]:
        print(row["baseline"], row["decision"], row["doc_count"], row["retrieval_calls"])


if __name__ == "__main__":
    main()
