import csv
import json
from pathlib import Path


def export_sbu_examples(input_csv: str | Path, output_jsonl: str | Path, limit: int = 50) -> None:
    with Path(input_csv).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    examples = [
        row
        for row in rows
        if row.get("reason") == "sufficient_but_unstable" and row.get("baseline") == "stability_aware_selection"
    ][:limit]
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
