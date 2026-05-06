import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    command = [
        sys.executable,
        "src/summarize_stability_results.py",
        "--input",
        args.input,
        "--dataset",
        args.dataset,
    ]
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
