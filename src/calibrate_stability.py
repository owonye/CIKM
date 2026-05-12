import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evaluate import build_resources, load_estimator, resolve_manifest_overrides
from experiment_utils import set_global_seed, write_run_config
from rag.pipeline import (
    GENERATOR_PROMPT_VERSION,
    StabilityAwareEvidenceSelector,
    build_diagnostic_perturbations,
    compute_anchoring_consistency,
)
from scoring.sufficiency import LightweightSufficiencyScorer


def parse_float_grid(raw: str) -> list[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Grid must contain at least one value.")
    return values


def format_eta(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def maybe_log_progress(stage: str, completed: int, total: int, started_at: float) -> None:
    if total <= 0:
        return
    interval = max(1, total // 20)
    if completed % interval != 0 and completed != total:
        return
    elapsed = max(time.time() - started_at, 1e-9)
    rate = completed / elapsed
    eta = (total - completed) / rate if rate > 0 else 0.0
    print(
        f"[PROGRESS] {stage}: {completed}/{total} ({completed / total * 100:.1f}%) "
        f"elapsed={format_eta(elapsed)} eta={format_eta(eta)}"
    )


def build_candidate_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, queries, retriever, _, generator = build_resources(args)
    estimator, _, calibration_config = load_estimator(args)
    feature_aspect_model = "" if args.mode == "demo" else args.embedding_model
    shared_sufficiency_scorer = LightweightSufficiencyScorer.from_estimator(
        estimator,
        aspect_model=feature_aspect_model,
    )

    records: list[dict[str, Any]] = []
    insufficient_count = 0
    no_candidate_count = 0
    stage_started = time.time()

    for idx, query in enumerate(queries, start=1):
        pool = retriever.retrieve(query, top_k=args.candidate_pool_k)
        initial_docs = pool[: args.initial_k]
        candidates = pool[args.initial_k : args.candidate_pool_k]
        features = shared_sufficiency_scorer.score_components(query, initial_docs).to_features()
        decision = estimator.predict(features)
        if not decision.sufficient:
            insufficient_count += 1
            maybe_log_progress("stability_candidate_records", idx, len(queries), stage_started)
            continue
        if not candidates:
            no_candidate_count += 1
            maybe_log_progress("stability_candidate_records", idx, len(queries), stage_started)
            continue

        perturbations = build_diagnostic_perturbations(initial_docs, candidates)
        consistency, diagnostic_generations, _ = compute_anchoring_consistency(
            query,
            initial_docs,
            generator,
            perturbations,
            tail_level=args.tail_level,
        )
        selector = StabilityAwareEvidenceSelector(
            retriever=retriever,
            generator=generator,
            estimator=estimator,
            initial_k=args.initial_k,
            expanded_k=args.expanded_k,
            candidate_pool_k=args.candidate_pool_k,
            stability_threshold=args.stability_threshold,
            tail_level=args.tail_level,
            sufficiency_tolerance=args.sufficiency_tolerance,
            enforce_sufficiency_filter=True,
            aspect_model=feature_aspect_model,
            sufficiency_scorer=shared_sufficiency_scorer,
        )
        candidate_rows = [
            selector._candidate_utility(
                query,
                initial_docs,
                candidate,
                decision.sufficiency_score,
                consistency,
            )
            for candidate in candidates
        ]
        records.append(
            {
                "query_id": query.query_id,
                "base_consistency": consistency,
                "base_sufficiency": decision.sufficiency_score,
                "threshold": estimator.threshold,
                "diagnostic_generations": diagnostic_generations,
                "candidates": candidate_rows,
            }
        )
        maybe_log_progress("stability_candidate_records", idx, len(queries), stage_started)

    metadata = {
        "num_queries": len(queries),
        "sufficient_with_candidates": len(records),
        "insufficient_count": insufficient_count,
        "no_candidate_count": no_candidate_count,
        "generator_type": "openai" if args.use_openai else ("hf_local" if args.hf_model_id else "simple_placeholder"),
        "model_version": args.openai_model if args.use_openai else (args.hf_model_id if args.hf_model_id else "simple_placeholder"),
        "openai_cache_stats": generator.get_cache_stats() if args.use_openai else None,
        "calibration_config": calibration_config,
    }
    return records, metadata


def evaluate_setting(
    records: list[dict[str, Any]],
    gamma: float,
    rho: float,
    sufficiency_tolerance: float,
) -> dict[str, Any]:
    stable_count = 0
    unstable_count = 0
    recovered_count = 0
    post_consistency_sum = 0.0
    stability_gain_sum = 0.0
    selected_utility_sum = 0.0
    selected_redundancy_sum = 0.0

    for record in records:
        base_consistency = float(record["base_consistency"])
        if base_consistency >= gamma:
            stable_count += 1
            post_consistency_sum += base_consistency
            continue

        unstable_count += 1
        candidates = record["candidates"]
        feasible_candidates = [
            item
            for item in candidates
            if record["base_sufficiency"] + item.delta_sufficiency >= record["threshold"] - sufficiency_tolerance
        ]
        if not feasible_candidates:
            post_consistency_sum += base_consistency
            continue
        base_deficit = max(gamma - base_consistency, 0.0)
        selected = max(
            feasible_candidates,
            key=lambda item: base_deficit - max(gamma - item.post_consistency, 0.0) - rho * item.redundancy_penalty,
        )
        post_deficit = max(gamma - selected.post_consistency, 0.0)
        utility = base_deficit - post_deficit - rho * selected.redundancy_penalty
        post_consistency_sum += selected.post_consistency
        stability_gain_sum += selected.delta_consistency
        selected_utility_sum += utility
        selected_redundancy_sum += selected.redundancy_penalty
        if selected.post_consistency >= gamma:
            recovered_count += 1

    count = max(len(records), 1)
    unstable_denominator = max(unstable_count, 1)
    recovery_rate = recovered_count / unstable_denominator
    avg_post_consistency = post_consistency_sum / count
    avg_stability_gain = stability_gain_sum / unstable_denominator
    sbu_rate = unstable_count / count
    objective = avg_post_consistency + 0.5 * recovery_rate - 0.05 * sbu_rate
    return {
        "stability_threshold": gamma,
        "utility_rho": rho,
        "sufficiency_tolerance": sufficiency_tolerance,
        "objective": objective,
        "sufficient_with_candidates": len(records),
        "stable_count": stable_count,
        "unstable_count": unstable_count,
        "sbu_rate_among_sufficient": sbu_rate,
        "recovered_count": recovered_count,
        "recovery_rate": recovery_rate,
        "avg_post_consistency": avg_post_consistency,
        "avg_stability_gain_unstable": avg_stability_gain,
        "avg_selected_utility_unstable": selected_utility_sum / unstable_denominator,
        "avg_selected_redundancy_unstable": selected_redundancy_sum / unstable_denominator,
    }


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
    parser.add_argument("--expanded-k", type=int, default=8)
    parser.add_argument("--candidate-pool-k", type=int, default=8)
    parser.add_argument("--stability-threshold", type=float, default=0.8)
    parser.add_argument("--tail-level", type=float, default=1.0)
    parser.add_argument("--tail-grid", default="")
    parser.add_argument("--sufficiency-tolerance", type=float, default=0.0)
    parser.add_argument("--weak-support-overlap-threshold", type=float, default=0.2)
    parser.add_argument("--confidence-threshold", type=float, default=0.88)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--confidence-calibration-file", default="")
    parser.add_argument("--calibration-file", default="")
    parser.add_argument("--structure-aware-label", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma-grid", default="0.6,0.7,0.8,0.9")
    parser.add_argument("--rho-grid", default="0.0,0.1,0.2")
    parser.add_argument("--epsilon-f-grid", default="0.0,0.02,0.05")
    parser.add_argument("--output", default="results/stability_calib.json")
    args = parser.parse_args()
    args = resolve_manifest_overrides(args)
    if not args.use_openai and not args.hf_model_id and args.mode != "demo" and not args.allow_simple_generator:
        raise ValueError("Non-demo stability calibration requires --use-openai, --hf-model-id, or --allow-simple-generator.")
    set_global_seed(args.seed)

    gamma_grid = parse_float_grid(args.gamma_grid)
    rho_grid = parse_float_grid(args.rho_grid)
    epsilon_f_grid = parse_float_grid(args.epsilon_f_grid)
    tail_grid = parse_float_grid(args.tail_grid) if args.tail_grid else [args.tail_level]

    best = None
    all_results = []
    metadata_by_tail: dict[str, Any] = {}
    for tail_level in tail_grid:
        args.tail_level = tail_level
        records, metadata = build_candidate_records(args)
        metadata_by_tail[str(tail_level)] = metadata
        for gamma, rho, epsilon_f in product(gamma_grid, rho_grid, epsilon_f_grid):
            result = evaluate_setting(records, gamma=gamma, rho=rho, sufficiency_tolerance=epsilon_f)
            result["tail_level"] = tail_level
            all_results.append(result)
            if best is None or result["objective"] > best["objective"]:
                best = result

    payload = {
        "best": best,
        "grid_results": all_results,
        "metadata": metadata_by_tail,
        "args": vars(args),
        "prompt_template_version": GENERATOR_PROMPT_VERSION,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_run_config(
        output_path.with_suffix(".meta.json"),
        {
            "script": "calibrate_stability.py",
            "args": vars(args),
            "metadata": metadata_by_tail,
            "best": best,
            "output": str(output_path),
        },
    )
    print(f"Saved stability calibration to {output_path}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
