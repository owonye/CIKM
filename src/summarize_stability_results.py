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
    "oracle_best_candidate",
}


def safe_float(value: str | None) -> float:
    if value is None or value == "":
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
    utilities = [safe_float(str(item.get("utility", ""))) for item in details]
    gains = [safe_float(str(item.get("delta_consistency", ""))) for item in details]
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
        "stability_aware_selection",
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
                "delta_consistency": sum(safe_float(row.get("stability_gain")) for row in group) / max(count, 1),
                "recovery_rate": sum(1 for row in group if safe_bool(row.get("recovered"))) / max(count, 1),
                "post_consistency": sum(safe_float(row.get("post_selection_consistency")) for row in group) / max(count, 1),
                "answer_variance_proxy": sum(1.0 - safe_float(row.get("post_selection_consistency")) for row in group) / max(count, 1),
                "diagnostic_generations": sum(safe_float(row.get("diagnostic_generations")) for row in group) / max(count, 1),
                "utility_gain_spearman": sum(spearman_values) / max(len(spearman_values), 1) if spearman_values else "",
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
        output.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "n": count,
                "em": sum(safe_float(row.get("exact_match")) for row in group) / count,
                "f1": sum(safe_float(row.get("f1")) for row in group) / count,
                "avg_docs": sum(safe_float(row.get("doc_count")) for row in group) / count,
                "expand_rate": sum(1 for row in group if safe_bool(row.get("expanded"))) / count,
                "consistency": (
                    sum(safe_float(row.get("post_selection_consistency")) for row in consistency_labeled)
                    / max(len(consistency_labeled), 1)
                ),
                "recovery_rate": (
                    sum(1 for row in recovery_labeled if safe_bool(row.get("recovered")))
                    / max(len(recovery_labeled), 1)
                ),
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
                "utility_gain_spearman": sum(values) / max(len(values), 1) if values else "",
            }
        )
    return output


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
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = load_rows(input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    stem = input_path.stem

    sbu_rows = summarize_sbu(rows, args.dataset)
    repair_rows = summarize_repair(rows, args.dataset)
    end_to_end_rows = summarize_end_to_end(rows, args.dataset)
    rank_corr_rows = summarize_rank_corr(rows, args.dataset)

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
            "delta_consistency",
            "recovery_rate",
            "post_consistency",
            "answer_variance_proxy",
            "diagnostic_generations",
            "utility_gain_spearman",
        ],
    )
    write_csv(
        output_dir / "table_repair.csv",
        repair_rows,
        [
            "dataset",
            "baseline",
            "unstable_cases",
            "delta_consistency",
            "recovery_rate",
            "post_consistency",
            "answer_variance_proxy",
            "diagnostic_generations",
            "utility_gain_spearman",
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
        ["dataset", "baseline", "queries_with_candidate_rankings", "utility_gain_spearman"],
    )
    write_examples(rows, output_dir / "examples_sbu.jsonl")


if __name__ == "__main__":
    main()
