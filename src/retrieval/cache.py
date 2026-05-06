import json
from pathlib import Path


class RetrievalCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: dict[str, list[str]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                self.records[str(entry["query_id"])] = list(entry["doc_ids"])

    def get(self, query_id: str) -> list[str] | None:
        return self.records.get(query_id)

    def put(self, query_id: str, doc_ids: list[str]) -> None:
        self.records[query_id] = doc_ids

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for query_id, doc_ids in sorted(self.records.items()):
                f.write(json.dumps({"query_id": query_id, "doc_ids": doc_ids}, ensure_ascii=False) + "\n")
