import argparse
import csv
import json
from pathlib import Path


DEFAULT_DATASETS = ["hotpotqa", "musique", "nq", "triviaqa"]
DEFAULT_BASELINES = [
    "diagnose_then_expand",
    "selection_max_query_overlap",
    "stability_aware_selection",
]


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_eval_csv(results_root: Path, dataset: str, size: int) -> Path:
    pattern = f"**/eval_{dataset}_{size}.csv"
    matches = sorted(
        [path for path in results_root.glob(pattern) if "semantic_agreement" not in path.parts],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No eval CSV found for {dataset}: {results_root}/{pattern}")
    return matches[0]


def find_calibration_threshold(results_root: Path, input_csv: Path, rows: list[dict[str, str]]) -> tuple[float, str]:
    calibration_source = next(
        (
            row.get("calibration_source", "").strip()
            for row in rows
            if row.get("calibration_source", "").strip()
            and row.get("calibration_source", "").strip() != "default"
        ),
        "",
    )
    if not calibration_source:
        return 1.0, "default_or_missing_calibration_source"

    candidates = [input_csv.parent / calibration_source]
    candidates.extend(sorted(results_root.glob(f"**/{calibration_source}")))
    for path in candidates:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            config = json.load(f)
        threshold = config.get("threshold")
        if threshold is None:
            continue
        return float(threshold), str(path)

    return 1.0, f"missing_calibration_file:{calibration_source}"


def post_sufficiency_maintained(row: dict[str, str], sufficiency_threshold: float) -> bool | None:
    delta = row.get("candidate_delta_sufficiency")
    if delta in (None, ""):
        return None
    score = safe_float(row.get("sufficiency_score"))
    tolerance = safe_float(row.get("sufficiency_tolerance"))
    return score + safe_float(delta) >= sufficiency_threshold - tolerance


def group_ratio(rows: list[dict[str, str]], predicate) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)


def mean(rows: list[dict[str, str]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(safe_float(row.get(field)) for row in rows) / len(rows)


def summarize_recovery(
    rows: list[dict[str, str]],
    dataset: str,
    baselines: list[str],
    sufficiency_threshold: float,
    calibration_source: str,
) -> list[dict[str, object]]:
    output = []
    for baseline in baselines:
        group = [
            row
            for row in rows
            if row.get("baseline") == baseline and row.get("reason") == "sufficient_but_unstable"
        ]
        maintained = [
            value
            for row in group
            if (value := post_sufficiency_maintained(row, sufficiency_threshold)) is not None
        ]
        output.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "sbu_cases": len(group),
                "recovered_rate": group_ratio(group, lambda row: safe_bool(row.get("recovered"))),
                "post_consistency_mean": mean(group, "post_selection_consistency"),
                "sufficiency_threshold": sufficiency_threshold,
                "sufficiency_calibration_source": calibration_source,
                "sufficiency_maintained_rate": (
                    sum(1 for value in maintained if value) / len(maintained)
                    if maintained
                    else ""
                ),
                "sufficiency_maintained_n": len(maintained),
                "gold_em_mean": mean(group, "exact_match"),
                "gold_f1_mean": mean(group, "f1"),
                "gold_f1_ge_0_8_rate": group_ratio(group, lambda row: safe_float(row.get("f1")) >= 0.8),
                "avg_docs": mean(group, "doc_count"),
            }
        )
    return output


def summarize_support(rows: list[dict[str, str]], dataset: str) -> list[dict[str, object]]:
    output = []
    proposed = [row for row in rows if row.get("baseline") == "stability_aware_selection"]
    buckets = {
        "sufficient_but_unstable": [
            row
            for row in proposed
            if row.get("reason") == "sufficient_but_unstable"
        ],
        "stable_sufficient": [
            row
            for row in proposed
            if row.get("decision") == "answer_now" and row.get("reason") != "sufficient_but_unstable"
        ],
        "insufficient_or_expanded": [
            row
            for row in proposed
            if row.get("decision") != "answer_now"
        ],
    }
    for bucket, group in buckets.items():
        output.append(
            {
                "dataset": dataset,
                "bucket": bucket,
                "n": len(group),
                "oracle_initial_support_rate": group_ratio(group, lambda row: safe_bool(row.get("oracle_initial_support"))),
                "oracle_expanded_support_rate": group_ratio(group, lambda row: safe_bool(row.get("oracle_expanded_support"))),
                "oracle_has_signal_rate": group_ratio(group, lambda row: safe_bool(row.get("oracle_has_signal"))),
                "gold_em_mean": mean(group, "exact_match"),
                "gold_f1_mean": mean(group, "f1"),
                "sufficiency_score_mean": mean(group, "sufficiency_score"),
                "supportiveness_mean": mean(group, "supportiveness"),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSV-only defense checks for recovery and SBU sufficiency-gate false positives."
    )
    parser.add_argument("--results-root", default="results/gpt_new_baselines")
    parser.add_argument("--output-dir", default="results/gpt_new_baselines/defense_checks")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--size", type=int, default=1000)
    args = parser.parse_args()

    root = Path.cwd()
    results_root = (root / args.results_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    datasets = parse_list(args.datasets)
    baselines = parse_list(args.baselines)

    recovery_rows = []
    support_rows = []
    for dataset in datasets:
        input_csv = find_eval_csv(results_root, dataset, args.size)
        rows = load_rows(input_csv)
        sufficiency_threshold, calibration_source = find_calibration_threshold(results_root, input_csv, rows)
        print(f"[RUN] {dataset}: {input_csv}")
        print(f"[CALIB] {dataset}: threshold={sufficiency_threshold} source={calibration_source}")
        recovery_rows.extend(
            summarize_recovery(rows, dataset, baselines, sufficiency_threshold, calibration_source)
        )
        support_rows.extend(summarize_support(rows, dataset))

    recovery_fields = [
        "dataset",
        "baseline",
        "sbu_cases",
        "recovered_rate",
        "post_consistency_mean",
        "sufficiency_threshold",
        "sufficiency_calibration_source",
        "sufficiency_maintained_rate",
        "sufficiency_maintained_n",
        "gold_em_mean",
        "gold_f1_mean",
        "gold_f1_ge_0_8_rate",
        "avg_docs",
    ]
    support_fields = [
        "dataset",
        "bucket",
        "n",
        "oracle_initial_support_rate",
        "oracle_expanded_support_rate",
        "oracle_has_signal_rate",
        "gold_em_mean",
        "gold_f1_mean",
        "sufficiency_score_mean",
        "supportiveness_mean",
    ]

    recovery_path = output_dir / "recovery_definition_check.csv"
    support_path = output_dir / "sufficiency_gate_support_check.csv"
    write_csv(recovery_path, recovery_rows, recovery_fields)
    write_csv(support_path, support_rows, support_fields)
    print(f"[DONE] saved {recovery_path}")
    print(f"[DONE] saved {support_path}")


if __name__ == "__main__":
    main()
