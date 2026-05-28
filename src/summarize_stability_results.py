import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


STABILITY_BASELINES = {
    "diagnose_then_expand",
    "random_selection",
    "next_ranked_selection",
    "stability_aware_selection",
    "selection_max_sufficiency_gain",
    "selection_max_query_overlap",
    "oracle_best_candidate",
}


def safe_float(value: str | None) -> float:
    if value is None or value == "" or str(value).lower() in {"none", "null"}:
        return 0.0
    return float(value)


def safe_bool(value: str | None) -> bool:
    return str(value).lower() == "true"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = sum((x - x_mean) ** 2 for x in xs) ** 0.5
    y_den = sum((y - y_mean) ** 2 for y in ys) ** 0.5
    if x_den == 0.0 or y_den == 0.0:
        return None
    return numerator / (x_den * y_den)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys))


def candidate_spearman(row: dict[str, str]) -> float | None:
    raw = row.get("candidate_details", "")
    if not raw:
        return None
    try:
        details = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(details, list) or len(details) < 2:
        return None
    scored_details = [item for item in details if item.get("utility") is not None]
    if len(scored_details) < 2:
        return None
    utilities = [safe_float(str(item.get("utility", ""))) for item in scored_details]
    gains = [
        safe_float(str(item.get("anchor_deficit_reduction", item.get("delta_consistency", ""))))
        for item in scored_details
    ]
    return spearman(utilities, gains)


def summarize_sbu(rows: list[dict[str, str]], dataset: str) -> list[dict[str, object]]:
    output = []
    for baseline in ["stability_aware_selection"]:
        group = [row for row in rows if row.get("baseline") == baseline and row.get("anchoring_consistency", "") != ""]
        stop_cases = len(group)
        sbu_cases = sum(1 for row in group if row.get("reason") == "sufficient_but_unstable")
        total_queries = len({row.get("query_id", "") for row in rows if row.get("baseline") == baseline})
        output.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "total_queries": total_queries,
                "sufficient_stop_cases": stop_cases,
                "stop_cases_pct": stop_cases / max(total_queries, 1),
                "sbu_cases": sbu_cases,
                "sbu_rate_among_stop_cases": sbu_cases / max(stop_cases, 1),
                "sbu_rate_among_all_queries": sbu_cases / max(total_queries, 1),
            }
        )
    return output


def summarize_repair(rows: list[dict[str, str]], dataset: str) -> list[dict[str, object]]:
    output = []
    for baseline in [
        "diagnose_then_expand",
        "random_selection",
        "next_ranked_selection",
        "selection_max_sufficiency_gain",
        "selection_max_query_overlap",
        "stability_aware_selection",
        "selection_mean_consistency",
        "selection_no_filter",
        "selection_no_redundancy",
        "oracle_best_candidate",
    ]:
        group = [
            row
            for row in rows
            if row.get("baseline") == baseline and row.get("reason") == "sufficient_but_unstable"
        ]
        count = len(group)
        spearman_values = [value for row in group if (value := candidate_spearman(row)) is not None]
        output.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "unstable_cases": count,
                "anchor_deficit_reduction": sum(safe_float(row.get("anchor_deficit_reduction")) for row in group) / max(count, 1),
                "delta_consistency": sum(safe_float(row.get("stability_gain")) for row in group) / max(count, 1),
                "recovery_rate": sum(1 for row in group if safe_bool(row.get("recovered"))) / max(count, 1),
                "post_consistency": sum(safe_float(row.get("post_selection_consistency")) for row in group) / max(count, 1),
                "answer_variance_proxy": sum(1.0 - safe_float(row.get("post_selection_consistency")) for row in group) / max(count, 1),
                "diagnostic_generations": sum(safe_float(row.get("diagnostic_generations")) for row in group) / max(count, 1),
                "utility_repair_spearman": sum(spearman_values) / max(len(spearman_values), 1) if spearman_values else "",
            }
        )
    return output


def summarize_end_to_end(rows: list[dict[str, str]], dataset: str) -> list[dict[str, object]]:
    output = []
    by_baseline: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_baseline[row.get("baseline", "")].append(row)
    for baseline, group in sorted(by_baseline.items()):
        count = len(group)
        if count == 0:
            continue
        recovery_labeled = [row for row in group if row.get("recovered", "") != ""]
        consistency_labeled = [row for row in group if row.get("post_selection_consistency", "") != ""]
        consistency = ""
        if consistency_labeled:
            consistency = sum(safe_float(row.get("post_selection_consistency")) for row in consistency_labeled) / len(
                consistency_labeled
            )
        recovery_rate = ""
        if recovery_labeled:
            recovery_rate = sum(1 for row in recovery_labeled if safe_bool(row.get("recovered"))) / len(recovery_labeled)
        output.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "n": count,
                "em": sum(safe_float(row.get("exact_match")) for row in group) / count,
                "f1": sum(safe_float(row.get("f1")) for row in group) / count,
                "avg_docs": sum(safe_float(row.get("doc_count")) for row in group) / count,
                "expand_rate": sum(1 for row in group if safe_bool(row.get("expanded"))) / count,
                "consistency": consistency,
                "recovery_rate": recovery_rate,
                "diagnostic_generations": sum(safe_float(row.get("diagnostic_generations")) for row in group) / count,
            }
        )
    return output


def summarize_rank_corr(rows: list[dict[str, str]], dataset: str) -> list[dict[str, object]]:
    output = []
    for baseline in ["stability_aware_selection", "oracle_best_candidate"]:
        group = [row for row in rows if row.get("baseline") == baseline]
        values = [value for row in group if (value := candidate_spearman(row)) is not None]
        output.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "queries_with_candidate_rankings": len(values),
                "utility_repair_spearman": sum(values) / max(len(values), 1) if values else "",
            }
        )
    return output


def summarize_selection_agreement(rows: list[dict[str, str]], dataset: str) -> list[dict[str, object]]:
    by_query_baseline: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        query_key = row.get("query_uid") or row.get("query_id") or ""
        baseline = row.get("baseline") or ""
        if query_key and baseline:
            by_query_baseline[(query_key, baseline)] = row

    output = []
    proposed_baseline = "stability_aware_selection"
    comparison_baselines = [
        "oracle_best_candidate",
        "selection_no_filter",
        "selection_no_redundancy",
        "selection_mean_consistency",
        "selection_max_sufficiency_gain",
        "selection_max_query_overlap",
        "random_selection",
        "next_ranked_selection",
    ]
    proposed_rows = [
        row
        for (query_key, baseline), row in by_query_baseline.items()
        if baseline == proposed_baseline
        and row.get("reason") == "sufficient_but_unstable"
        and row.get("selected_doc_id")
    ]
    for comparison in comparison_baselines:
        comparable = []
        for proposed in proposed_rows:
            query_key = proposed.get("query_uid") or proposed.get("query_id") or ""
            other = by_query_baseline.get((query_key, comparison))
            if other is not None and other.get("selected_doc_id"):
                comparable.append((proposed, other))
        same = sum(1 for proposed, other in comparable if proposed.get("selected_doc_id") == other.get("selected_doc_id"))
        output.append(
            {
                "dataset": dataset,
                "proposed_baseline": proposed_baseline,
                "comparison_baseline": comparison,
                "comparable_cases": len(comparable),
                "same_selection": same,
                "agreement_rate": same / max(len(comparable), 1),
            }
        )
    return output


def row_by_baseline(rows: list[dict[str, object]], baseline: str) -> dict[str, object] | None:
    for row in rows:
        if row.get("baseline") == baseline:
            return row
    return None


def as_float(value: object) -> float:
    if value in {"", None}:
        return 0.0
    return float(value)


def validate_result_pattern(
    dataset: str,
    sbu_rows: list[dict[str, object]],
    repair_rows: list[dict[str, object]],
    end_to_end_rows: list[dict[str, object]],
    min_sbu_rate: float,
    target_sbu_low: float,
    target_sbu_high: float,
    min_recovery_gain: float,
    max_f1_drop: float,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    sbu = row_by_baseline(sbu_rows, "stability_aware_selection")
    proposed_repair = row_by_baseline(repair_rows, "stability_aware_selection")
    expand_repair = row_by_baseline(repair_rows, "diagnose_then_expand")
    next_repair = row_by_baseline(repair_rows, "next_ranked_selection")
    proposed_main = row_by_baseline(end_to_end_rows, "stability_aware_selection")
    sufficiency_main = row_by_baseline(end_to_end_rows, "structure_aware_adaptive_rag")
    mean_consistency_repair = row_by_baseline(repair_rows, "selection_mean_consistency")
    no_filter_repair = row_by_baseline(repair_rows, "selection_no_filter")
    no_redundancy_repair = row_by_baseline(repair_rows, "selection_no_redundancy")

    if sbu is not None:
        sbu_rate = as_float(sbu.get("sbu_rate_among_stop_cases"))
        checks.append(
            {
                "dataset": dataset,
                "check": "sbu_non_trivial",
                "status": "pass" if sbu_rate >= min_sbu_rate else "weak",
                "value": sbu_rate,
                "threshold": min_sbu_rate,
                "message": "SBU is frequent enough to support the core claim."
                if sbu_rate >= min_sbu_rate
                else "SBU is rare; the core claim is weak.",
            }
        )
        checks.append(
            {
                "dataset": dataset,
                "check": "sbu_target_range",
                "status": "pass" if target_sbu_low <= sbu_rate <= target_sbu_high else "watch",
                "value": sbu_rate,
                "threshold": f"{target_sbu_low}-{target_sbu_high}",
                "message": "SBU rate is in the preferred dataset-specific range."
                if target_sbu_low <= sbu_rate <= target_sbu_high
                else "SBU rate is outside the preferred range; inspect gamma/tau and perturbations.",
            }
        )

    if proposed_repair is not None and expand_repair is not None:
        proposed_recovery = as_float(proposed_repair.get("recovery_rate"))
        expand_recovery = as_float(expand_repair.get("recovery_rate"))
        recovery_gain = proposed_recovery - expand_recovery
        proposed_delta_c = as_float(proposed_repair.get("anchor_deficit_reduction"))
        expand_delta_c = as_float(expand_repair.get("anchor_deficit_reduction"))
        checks.append(
            {
                "dataset": dataset,
                "check": "beats_diagnose_expand_recovery",
                "status": "pass" if recovery_gain >= min_recovery_gain else "weak",
                "value": recovery_gain,
                "threshold": min_recovery_gain,
                "message": "Proposed selection repairs instability better than naive expansion."
                if recovery_gain >= min_recovery_gain
                else "Diagnose->expand is too close to proposed; selection novelty is weak.",
            }
        )
        checks.append(
            {
                "dataset": dataset,
                "check": "beats_diagnose_expand_deficit_reduction",
                "status": "pass" if proposed_delta_c > expand_delta_c else "weak",
                "value": proposed_delta_c - expand_delta_c,
                "threshold": ">0",
                "message": "Proposed selection reduces anchor deficit more than naive expansion."
                if proposed_delta_c > expand_delta_c
                else "Proposed selection does not improve anchor-deficit reduction over naive expansion.",
            }
        )

    if proposed_repair is not None and next_repair is not None:
        proposed_recovery = as_float(proposed_repair.get("recovery_rate"))
        next_recovery = as_float(next_repair.get("recovery_rate"))
        proposed_delta_c = as_float(proposed_repair.get("anchor_deficit_reduction"))
        next_delta_c = as_float(next_repair.get("anchor_deficit_reduction"))
        proposed_var = as_float(proposed_repair.get("answer_variance_proxy"))
        next_var = as_float(next_repair.get("answer_variance_proxy"))
        checks.append(
            {
                "dataset": dataset,
                "check": "beats_next_ranked",
                "status": "pass" if proposed_recovery > next_recovery and proposed_delta_c > next_delta_c and proposed_var < next_var else "weak",
                "value": {
                    "recovery_gain": proposed_recovery - next_recovery,
                    "deficit_reduction_gain": proposed_delta_c - next_delta_c,
                    "variance_reduction": next_var - proposed_var,
                },
                "threshold": "all improvements > 0",
                "message": "Utility-based selection beats next-ranked selection."
                if proposed_recovery > next_recovery and proposed_delta_c > next_delta_c and proposed_var < next_var
                else "Next-ranked is too close to proposed; utility module is not clearly justified.",
            }
        )

    if proposed_main is not None and sufficiency_main is not None:
        proposed_f1 = as_float(proposed_main.get("f1"))
        sufficiency_f1 = as_float(sufficiency_main.get("f1"))
        f1_delta = proposed_f1 - sufficiency_f1
        proposed_consistency = as_float(proposed_main.get("consistency"))
        sufficiency_consistency = as_float(sufficiency_main.get("consistency"))
        checks.append(
            {
                "dataset": dataset,
                "check": "quality_preserved",
                "status": "pass" if f1_delta >= -max_f1_drop else "weak",
                "value": f1_delta,
                "threshold": f">= {-max_f1_drop}",
                "message": "F1 is roughly preserved."
                if f1_delta >= -max_f1_drop
                else "F1 drops too much; stability gains may not be acceptable.",
            }
        )
        checks.append(
            {
                "dataset": dataset,
                "check": "end_to_end_consistency_gain",
                "status": "pass" if proposed_consistency > sufficiency_consistency else "weak",
                "value": proposed_consistency - sufficiency_consistency,
                "threshold": ">0",
                "message": "End-to-end consistency improves over sufficiency-only stopping."
                if proposed_consistency > sufficiency_consistency
                else "End-to-end consistency does not improve over sufficiency-only stopping.",
            }
        )

    ablation_rows = [row for row in [mean_consistency_repair, no_filter_repair, no_redundancy_repair] if row is not None]
    if proposed_repair is not None and ablation_rows:
        proposed_recovery = as_float(proposed_repair.get("recovery_rate"))
        proposed_delta_c = as_float(proposed_repair.get("anchor_deficit_reduction"))
        best_ablation_recovery = max(as_float(row.get("recovery_rate")) for row in ablation_rows)
        best_ablation_delta_c = max(as_float(row.get("anchor_deficit_reduction")) for row in ablation_rows)
        checks.append(
            {
                "dataset": dataset,
                "check": "beats_utility_ablations",
                "status": "pass" if proposed_recovery >= best_ablation_recovery and proposed_delta_c >= best_ablation_delta_c else "weak",
                "value": {
                    "recovery_gain_vs_best_ablation": proposed_recovery - best_ablation_recovery,
                    "deficit_reduction_gain_vs_best_ablation": proposed_delta_c - best_ablation_delta_c,
                },
                "threshold": ">=0 against best method ablation",
                "message": "Full robust marginal value is justified over method ablations."
                if proposed_recovery >= best_ablation_recovery and proposed_delta_c >= best_ablation_delta_c
                else "A method ablation matches or beats the full utility; robust marginal value is weakly justified.",
            }
        )

    weak_count = sum(1 for check in checks if check["status"] == "weak")
    checks.append(
        {
            "dataset": dataset,
            "check": "overall",
            "status": "pass" if weak_count == 0 else "weak",
            "value": weak_count,
            "threshold": 0,
            "message": "The run matches the intended paper result pattern."
            if weak_count == 0
            else "The run is weak for the paper claim; inspect failed checks before using these results.",
        }
    )
    return checks


def write_examples(rows: list[dict[str, str]], output_path: Path, limit: int = 50) -> None:
    examples = [
        row
        for row in rows
        if row.get("baseline") == "stability_aware_selection" and row.get("reason") == "sufficient_but_unstable"
    ][:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--min-sbu-rate", type=float, default=0.15)
    parser.add_argument("--min-recovery-gain", type=float, default=0.10)
    parser.add_argument("--max-f1-drop", type=float, default=0.02)
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = load_rows(input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    stem = input_path.stem

    sbu_rows = summarize_sbu(rows, args.dataset)
    repair_rows = summarize_repair(rows, args.dataset)
    end_to_end_rows = summarize_end_to_end(rows, args.dataset)
    rank_corr_rows = summarize_rank_corr(rows, args.dataset)
    selection_agreement_rows = summarize_selection_agreement(rows, args.dataset)
    if args.dataset == "hotpotqa":
        target_sbu_low, target_sbu_high = 0.20, 0.40
    elif args.dataset == "nq":
        target_sbu_low, target_sbu_high = 0.10, 0.25
    else:
        target_sbu_low, target_sbu_high = args.min_sbu_rate, 1.0
    pattern_checks = validate_result_pattern(
        args.dataset,
        sbu_rows,
        repair_rows,
        end_to_end_rows,
        min_sbu_rate=args.min_sbu_rate,
        target_sbu_low=target_sbu_low,
        target_sbu_high=target_sbu_high,
        min_recovery_gain=args.min_recovery_gain,
        max_f1_drop=args.max_f1_drop,
    )

    write_csv(
        output_dir / f"{stem}_sbu_summary.csv",
        sbu_rows,
        [
            "dataset",
            "baseline",
            "total_queries",
            "sufficient_stop_cases",
            "stop_cases_pct",
            "sbu_cases",
            "sbu_rate_among_stop_cases",
            "sbu_rate_among_all_queries",
        ],
    )
    write_csv(
        output_dir / "table_sbu.csv",
        sbu_rows,
        [
            "dataset",
            "baseline",
            "total_queries",
            "sufficient_stop_cases",
            "stop_cases_pct",
            "sbu_cases",
            "sbu_rate_among_stop_cases",
            "sbu_rate_among_all_queries",
        ],
    )
    write_csv(
        output_dir / f"{stem}_repair_summary.csv",
        repair_rows,
        [
            "dataset",
            "baseline",
            "unstable_cases",
            "anchor_deficit_reduction",
            "delta_consistency",
            "recovery_rate",
            "post_consistency",
            "answer_variance_proxy",
            "diagnostic_generations",
            "utility_repair_spearman",
        ],
    )
    write_csv(
        output_dir / "table_repair.csv",
        repair_rows,
        [
            "dataset",
            "baseline",
            "unstable_cases",
            "anchor_deficit_reduction",
            "delta_consistency",
            "recovery_rate",
            "post_consistency",
            "answer_variance_proxy",
            "diagnostic_generations",
            "utility_repair_spearman",
        ],
    )
    write_csv(
        output_dir / f"{stem}_end_to_end_summary.csv",
        end_to_end_rows,
        [
            "dataset",
            "baseline",
            "n",
            "em",
            "f1",
            "avg_docs",
            "expand_rate",
            "consistency",
            "recovery_rate",
            "diagnostic_generations",
        ],
    )
    write_csv(
        output_dir / "table_main.csv",
        end_to_end_rows,
        [
            "dataset",
            "baseline",
            "n",
            "em",
            "f1",
            "avg_docs",
            "expand_rate",
            "consistency",
            "recovery_rate",
            "diagnostic_generations",
        ],
    )
    write_csv(
        output_dir / "candidate_rank_corr.csv",
        rank_corr_rows,
        ["dataset", "baseline", "queries_with_candidate_rankings", "utility_repair_spearman"],
    )
    write_csv(
        output_dir / "selection_agreement.csv",
        selection_agreement_rows,
        [
            "dataset",
            "proposed_baseline",
            "comparison_baseline",
            "comparable_cases",
            "same_selection",
            "agreement_rate",
        ],
    )
    write_csv(
        output_dir / "result_pattern_check.csv",
        pattern_checks,
        ["dataset", "check", "status", "value", "threshold", "message"],
    )
    (output_dir / "result_pattern_check.json").write_text(
        json.dumps(pattern_checks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_examples(rows, output_dir / "examples_sbu.jsonl")


if __name__ == "__main__":
    main()
