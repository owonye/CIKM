import argparse
import csv
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ["hotpotqa", "musique", "nq", "triviaqa"]
DEFAULT_BASELINES = [
    "diagnose_then_expand",
    "selection_max_query_overlap",
    "stability_aware_selection",
]


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_eval_csv(results_root: Path, dataset: str, size: int) -> Path:
    pattern = f"*/eval_{dataset}_{size}.csv"
    matches = sorted(
        results_root.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No eval CSV found for {dataset}: {results_root}/{pattern}")
    return matches[0]


def read_summary(path: Path, dataset: str, input_csv: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    output = []
    for row in rows:
        row = dict(row)
        row["dataset"] = dataset
        row["input_csv"] = str(input_csv)
        output.append(row)
    return output


def write_combined_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "baseline",
        "phase",
        "pairs",
        "semantic_agreement",
        "token_f1",
        "input_csv",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_post_minus_pre(rows: list[dict[str, str]]) -> None:
    print("\n[SUMMARY] semantic post_minus_pre")
    for row in rows:
        if row.get("phase") != "post_minus_pre":
            continue
        print(
            f"{row.get('dataset')}\t{row.get('baseline')}\t"
            f"semantic={row.get('semantic_agreement')}\ttoken_f1={row.get('token_f1')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run sampled semantic agreement checks over GPT stability eval outputs. "
            "This reuses existing generation cache and only calls the judge for sampled answer pairs."
        )
    )
    parser.add_argument("--results-root", default="results/gpt_new_baselines")
    parser.add_argument("--generation-cache", default="results/gpt_new_baselines/openai_cache_shared.jsonl")
    parser.add_argument("--output-dir", default="results/gpt_new_baselines/semantic_agreement")
    parser.add_argument("--judge-cache", default="results/gpt_new_baselines/semantic_judge_cache.jsonl")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--prompt-version", default="short_answer_v3")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    results_root = (root / args.results_root).resolve()
    generation_cache = (root / args.generation_cache).resolve()
    output_dir = (root / args.output_dir).resolve()
    judge_cache = (root / args.judge_cache).resolve()
    checker = root / "src" / "semantic_agreement_check.py"

    if not checker.exists():
        raise FileNotFoundError(f"Missing checker script: {checker}")
    if not generation_cache.exists():
        raise FileNotFoundError(f"Missing generation cache: {generation_cache}")

    datasets = parse_list(args.datasets)
    baselines = ",".join(parse_list(args.baselines))
    combined_rows: list[dict[str, str]] = []

    for dataset in datasets:
        input_csv = find_eval_csv(results_root, dataset, args.size)
        pair_output = output_dir / f"semantic_{dataset}_pairs_s{args.sample_size}.csv"
        summary_output = output_dir / f"semantic_{dataset}_summary_s{args.sample_size}.csv"
        command = [
            args.python,
            str(checker),
            "--input",
            str(input_csv),
            "--generation-cache",
            str(generation_cache),
            "--judge-cache",
            str(judge_cache),
            "--output",
            str(pair_output),
            "--summary-output",
            str(summary_output),
            "--sample-size",
            str(args.sample_size),
            "--seed",
            str(args.seed),
            "--baselines",
            baselines,
            "--judge-model",
            args.judge_model,
            "--prompt-version",
            args.prompt_version,
        ]
        print(f"[RUN] {dataset}: {input_csv}")
        print(" ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, check=True, cwd=root)
        combined_rows.extend(read_summary(summary_output, dataset, input_csv))

    if args.dry_run:
        return

    combined_output = output_dir / f"semantic_agreement_summary_all_s{args.sample_size}.csv"
    write_combined_summary(combined_output, combined_rows)
    print(f"\n[DONE] saved combined summary: {combined_output}")
    print_post_minus_pre(combined_rows)


if __name__ == "__main__":
    main()
