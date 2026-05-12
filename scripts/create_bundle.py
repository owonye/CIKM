from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_EXCLUDES = [
    ".git/",
    ".venv/",
    ".env",
    ".env.",
    "results/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.zip",
    "*.tar",
    "*.tar.gz",
]


def should_exclude(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    for pattern in patterns:
        if pattern.endswith("/"):
            prefix = pattern.rstrip("/")
            if rel == prefix or rel.startswith(prefix + "/"):
                return True
        elif pattern.endswith("."):
            if rel.startswith(pattern):
                return True
        elif "*" in pattern:
            if path.match(pattern):
                return True
        else:
            if rel == pattern:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deployment bundle excluding sensitive/local files.")
    parser.add_argument("--output", default="cikm_run.zip", help="Output zip path")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional exclude pattern (can repeat)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    patterns = DEFAULT_EXCLUDES + args.exclude

    if output.exists():
        output.unlink()

    written = 0
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if path.resolve() == output:
                continue
            if should_exclude(path, root, patterns):
                continue
            arcname = path.relative_to(root).as_posix()
            zf.write(path, arcname)
            written += 1

    print(f"Created {output} with {written} files")


if __name__ == "__main__":
    main()
