import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional convenience only
    load_dotenv = None


DEFAULT_BASELINES = [
    "diagnose_then_expand",
    "selection_max_query_overlap",
    "stability_aware_selection",
]


def split_ids(raw: str) -> list[str]:
    return [part for part in str(raw or "").split("|") if part]


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def token_f1(answer_a: str, answer_b: str) -> float:
    a_tokens = normalize_answer(answer_a).split()
    b_tokens = normalize_answer(answer_b).split()
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    common = {}
    for token in a_tokens:
        common[token] = min(a_tokens.count(token), b_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_generation_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = entry.get("key")
            value = entry.get("value")
            if isinstance(key, str) and isinstance(value, str):
                cache[key] = value
    return cache


def load_judge_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = entry.get("key")
            if isinstance(key, str):
                cache[key] = entry
    return cache


def append_judge_cache(path: Path, entry: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def answer_cache_keys(prompt_version: str, model: str, query: str, doc_ids: list[str]) -> list[str]:
    joined_ids = "|".join(doc_ids)
    return [
        f"{prompt_version}::{model}::{query}::{joined_ids}",
        f"{model}::{query}::{joined_ids}",
    ]


def cached_answer(
    generation_cache: dict[str, str],
    prompt_version: str,
    model: str,
    query: str,
    doc_ids: list[str],
) -> str | None:
    for key in answer_cache_keys(prompt_version, model, query, doc_ids):
        if key in generation_cache:
            return generation_cache[key]
    joined_ids = "|".join(doc_ids)
    suffix = f"::{query}::{joined_ids}"
    for key, value in generation_cache.items():
        if key.endswith(suffix):
            return value
    return None


def parse_candidate_details(row: dict[str, str]) -> list[dict[str, object]]:
    raw = row.get("candidate_details", "")
    if not raw:
        return []
    try:
        details = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return details if isinstance(details, list) else []


def candidate_ids(row: dict[str, str], proposed_row: dict[str, str] | None = None) -> list[str]:
    details = parse_candidate_details(row)
    if not details and proposed_row is not None:
        details = parse_candidate_details(proposed_row)
    ids = []
    for item in details:
        doc_id = item.get("candidate_doc_id") if isinstance(item, dict) else None
        if isinstance(doc_id, str) and doc_id:
            ids.append(doc_id)
    return list(dict.fromkeys(ids))


def feasible_candidate_ids(row: dict[str, str], proposed_row: dict[str, str] | None = None) -> list[str]:
    details = parse_candidate_details(row)
    if not details and proposed_row is not None:
        details = parse_candidate_details(proposed_row)
    ids = []
    for item in details:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("candidate_doc_id")
        feasible = item.get("feasible", True)
        if isinstance(doc_id, str) and doc_id and feasible is not False:
            ids.append(doc_id)
    return list(dict.fromkeys(ids))


def pre_pairs(row: dict[str, str], proposed_row: dict[str, str] | None) -> list[tuple[str, list[str], list[str]]]:
    initial = split_ids(row.get("initial_doc_ids", ""))
    candidates = candidate_ids(row, proposed_row)
    pairs = []
    if len(initial) > 1:
        pairs.append(("pre_reordered", initial, list(reversed(initial))))
    if initial and candidates:
        pairs.append(("pre_replacement", initial, initial[:-1] + [candidates[0]]))
    return pairs


def post_pairs(row: dict[str, str], proposed_row: dict[str, str] | None) -> list[tuple[str, list[str], list[str]]]:
    final_ids = split_ids(row.get("final_doc_ids", ""))
    baseline = row.get("baseline", "")
    if len(final_ids) <= 1:
        return []
    if baseline == "diagnose_then_expand":
        return [("post_reordered", final_ids, list(reversed(final_ids)))]

    fixed_candidate = final_ids[-1]
    base_docs = final_ids[:-1]
    pairs = [("post_reordered", final_ids, list(reversed(base_docs)) + [fixed_candidate])]
    used = set(final_ids)
    replacement = next((doc_id for doc_id in feasible_candidate_ids(row, proposed_row) if doc_id not in used), None)
    if replacement is not None and base_docs:
        pairs.append(("post_replacement", final_ids, base_docs[:-1] + [replacement, fixed_candidate]))
    elif len(base_docs) > 1:
        pairs.append(("post_rotated", final_ids, base_docs[1:] + base_docs[:1] + [fixed_candidate]))
    return pairs


def judge_key(query: str, answer_a: str, answer_b: str, model: str) -> str:
    first, second = sorted([normalize_answer(answer_a), normalize_answer(answer_b)])
    raw = json.dumps([model, query, first, second], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def semantic_agreement(
    query: str,
    answer_a: str,
    answer_b: str,
    model: str,
    judge_cache_path: Path,
    judge_cache: dict[str, dict[str, object]],
) -> tuple[int, str]:
    key = judge_key(query, answer_a, answer_b, model)
    cached = judge_cache.get(key)
    if cached is not None:
        return int(cached["agreement"]), str(cached.get("raw", ""))

    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").strip() or None)
    prompt = (
        "Decide whether two short answers are semantically equivalent for the question.\n"
        "Return only 1 if they mean the same answer, otherwise return only 0.\n\n"
        f"Question: {query}\n"
        f"Answer A: {answer_a}\n"
        f"Answer B: {answer_b}\n"
    )
    if hasattr(client, "responses"):
        response = client.responses.create(model=model, input=prompt)
        raw = response.output_text.strip()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
    agreement = 1 if raw.startswith("1") else 0
    entry = {"key": key, "agreement": agreement, "raw": raw}
    judge_cache[key] = entry
    append_judge_cache(judge_cache_path, entry)
    return agreement, raw


def make_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["baseline"]), str(row["phase"]))].append(row)
    output = []
    by_baseline: dict[str, dict[str, float]] = defaultdict(dict)
    for (baseline, phase), group in sorted(groups.items()):
        semantic = mean(float(row["semantic_agreement"]) for row in group)
        lexical = mean(float(row["token_f1"]) for row in group)
        by_baseline[baseline][phase] = semantic
        output.append(
            {
                "baseline": baseline,
                "phase": phase,
                "pairs": len(group),
                "semantic_agreement": semantic,
                "token_f1": lexical,
            }
        )
    for baseline, values in sorted(by_baseline.items()):
        if "pre" in values and "post" in values:
            output.append(
                {
                    "baseline": baseline,
                    "phase": "post_minus_pre",
                    "pairs": "",
                    "semantic_agreement": values["post"] - values["pre"],
                    "token_f1": "",
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Stability eval CSV.")
    parser.add_argument("--generation-cache", required=True, help="OpenAI generation cache JSONL from the eval run.")
    parser.add_argument("--judge-cache", default="results/semantic_judge_cache.jsonl")
    parser.add_argument("--output", default="results/semantic_agreement_pairs.csv")
    parser.add_argument("--summary-output", default="results/semantic_agreement_summary.csv")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--prompt-version", default="short_answer_v3")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    generation_cache = load_generation_cache(Path(args.generation_cache))
    judge_cache_path = Path(args.judge_cache)
    judge_cache = load_judge_cache(judge_cache_path)
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]

    by_query_baseline: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        query_key = row.get("query_uid") or row.get("query_id") or ""
        baseline = row.get("baseline") or ""
        if query_key and baseline:
            by_query_baseline[(query_key, baseline)] = row

    proposed_cases = [
        row
        for (query_key, baseline), row in by_query_baseline.items()
        if baseline == "stability_aware_selection" and row.get("reason") == "sufficient_but_unstable"
    ]
    random.Random(args.seed).shuffle(proposed_cases)
    proposed_cases = proposed_cases[: args.sample_size]

    detail_rows: list[dict[str, object]] = []
    missing_pairs = 0
    for proposed in proposed_cases:
        query_key = proposed.get("query_uid") or proposed.get("query_id") or ""
        for baseline in baselines:
            row = by_query_baseline.get((query_key, baseline))
            if row is None or row.get("reason") != "sufficient_but_unstable":
                continue
            query = row.get("query", "")
            model = row.get("model_version") or args.judge_model
            pairs = [("pre", *item) for item in pre_pairs(row, proposed)]
            pairs.extend(("post", *item) for item in post_pairs(row, proposed))
            for phase, pair_type, ids_a, ids_b in pairs:
                answer_a = cached_answer(generation_cache, args.prompt_version, model, query, ids_a)
                answer_b = cached_answer(generation_cache, args.prompt_version, model, query, ids_b)
                if answer_a is None or answer_b is None:
                    missing_pairs += 1
                    continue
                semantic, raw = semantic_agreement(
                    query,
                    answer_a,
                    answer_b,
                    args.judge_model,
                    judge_cache_path,
                    judge_cache,
                )
                detail_rows.append(
                    {
                        "query_key": query_key,
                        "baseline": baseline,
                        "phase": phase,
                        "pair_type": pair_type,
                        "ids_a": "|".join(ids_a),
                        "ids_b": "|".join(ids_b),
                        "answer_a": answer_a,
                        "answer_b": answer_b,
                        "semantic_agreement": semantic,
                        "token_f1": token_f1(answer_a, answer_b),
                        "judge_raw": raw,
                    }
                )

    write_csv(
        Path(args.output),
        detail_rows,
        [
            "query_key",
            "baseline",
            "phase",
            "pair_type",
            "ids_a",
            "ids_b",
            "answer_a",
            "answer_b",
            "semantic_agreement",
            "token_f1",
            "judge_raw",
        ],
    )
    write_csv(
        Path(args.summary_output),
        make_summary(detail_rows),
        ["baseline", "phase", "pairs", "semantic_agreement", "token_f1"],
    )
    print(f"sampled_sbu_cases={len(proposed_cases)}")
    print(f"judged_pairs={len(detail_rows)}")
    print(f"missing_cached_pairs={missing_pairs}")
    print(f"saved_pairs={args.output}")
    print(f"saved_summary={args.summary_output}")


if __name__ == "__main__":
    main()
