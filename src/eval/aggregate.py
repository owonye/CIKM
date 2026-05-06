from pathlib import Path

from summarize_stability_results import main as summarize_stability_main


def make_tables_from_records(input_path: str | Path, dataset: str, output_dir: str | Path | None = None) -> None:
    import sys

    argv = [
        "summarize_stability_results.py",
        "--input",
        str(input_path),
        "--dataset",
        dataset,
    ]
    if output_dir is not None:
        argv.extend(["--output-dir", str(output_dir)])
    old_argv = sys.argv
    try:
        sys.argv = argv
        summarize_stability_main()
    finally:
        sys.argv = old_argv
