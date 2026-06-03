import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return [token for token in text.split() if token not in {"a", "an", "the"}]


def answer_f1(left: str, right: str) -> float:
    left_tokens = normalize(left)
    right_tokens = normalize(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    overlap = sum(min(left_counts[token], right_counts[token]) for token in left_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_cache_key(key: str) -> tuple[str, tuple[str, ...]] | None:
    # Supports both old and new cache keys:
    #   model::question::doc|doc
    #   prompt_version::model::question::doc|doc
    match = re.search(r"::(?:hotpotqa|musique|nq|triviaqa)::", key)
    if match is None:
        return None
    prefix = key[: match.start()]
    docs_part = key[match.start() + 2 :]
    parts = prefix.split("::")
    if len(parts) < 2:
        return None
    question = parts[-1].strip()
    docs = tuple(doc for doc in docs_part.split("|") if doc)
    return question, docs


def one_replacement(original: tuple[str, ...], variant: tuple[str, ...]) -> bool:
    if len(original) != len(variant):
        return False
    return sum(left != right for left, right in zip(original, variant)) == 1


def short(text: str, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/stability_runs")
    parser.add_argument("--cache", default="results/stability_runs/openai_cache_shared.jsonl")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--require-original-correct", action="store_true")
    parser.add_argument("--gold-f1-threshold", type=float, default=0.8)
    parser.add_argument("--max-consistency", type=float, default=0.8)
    parser.add_argument("--max-variant-f1", type=float, default=1.0)
    parser.add_argument("--require-two-variants", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cache_path = Path(args.cache)

    answers: dict[tuple[str, tuple[str, ...]], str] = {}
    grouped: dict[str, list[tuple[tuple[str, ...], str]]] = defaultdict(list)
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            parsed = parse_cache_key(str(item.get("key", "")))
            value = item.get("value")
            if parsed is None or not isinstance(value, str):
                continue
            question, docs = parsed
            answers[(question, docs)] = value
            grouped[question].append((docs, value))

    rows = []
    for csv_path in results_dir.rglob("eval_*_1000.csv"):
        if args.dataset and args.dataset not in csv_path.name:
            continue
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("baseline") != "stability_aware_selection":
                    continue
                if row.get("reason") != "sufficient_but_unstable":
                    continue
                rows.append((csv_path, row))

    found = 0
    for csv_path, row in rows:
        question = row.get("query", "").strip()
        initial_docs = tuple(doc for doc in row.get("initial_doc_ids", "").split("|") if doc)
        if not question or not initial_docs:
            continue
        original = answers.get((question, initial_docs))
        if original is None:
            continue
        gold_answer = row.get("gold_answer", "")
        original_gold_f1 = answer_f1(original, gold_answer)
        if args.require_original_correct and original_gold_f1 < args.gold_f1_threshold:
            continue

        reversed_docs = tuple(reversed(initial_docs))
        reversed_answer = answers.get((question, reversed_docs))

        replacement = None
        for docs, answer in grouped.get(question, []):
            if docs == initial_docs or docs == reversed_docs:
                continue
            if len(docs) == len(initial_docs) and one_replacement(initial_docs, docs):
                replacement = (docs, answer)
                break

        variants = []
        if reversed_answer is not None:
            variants.append(("reordered", reversed_docs, reversed_answer, answer_f1(original, reversed_answer)))
        if replacement is not None:
            docs, answer = replacement
            variants.append(("single replacement", docs, answer, answer_f1(original, answer)))
        if not variants:
            continue
        if args.require_two_variants and len(variants) < 2:
            continue

        consistency = sum(item[3] for item in variants) / len(variants)
        if consistency > args.max_consistency:
            continue
        if args.max_variant_f1 < 1.0 and not any(item[3] <= args.max_variant_f1 for item in variants):
            continue

        found += 1
        print("=" * 88)
        print(f"source: {csv_path}")
        print(f"dataset row query_id: {row.get('query_id')}  consistency={consistency:.3f}")
        print(f"sufficiency_score={row.get('sufficiency_score')}  threshold source={row.get('calibration_source')}")
        print(f"gold: {gold_answer}")
        print(f"original-gold F1: {original_gold_f1:.3f}")
        print(f"question: {question}")
        print(f"initial docs: {' | '.join(initial_docs)}")
        print(f"original answer: {short(original, 260)}")
        for label, docs, answer, score in variants:
            print(f"{label} docs: {' | '.join(docs)}")
            print(f"{label} answer [F1={score:.3f}]: {short(answer, 260)}")
        print()
        if found >= args.limit:
            break

    if found == 0:
        print("No qualitative SBU example found in cache. Check cache path and eval CSV availability.")


if __name__ == "__main__":
    main()
