import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hotpotqa", "nq"], required=True)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--doc-limit", type=int, default=20000)
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--run-name", default="stability-selection")
    args = parser.parse_args()

    command = [
        sys.executable,
        "src/run_experiments.py",
        "--mode",
        args.dataset,
        "--sizes",
        str(args.size),
        "--doc-limit",
        str(args.doc_limit),
        "--corpus-split",
        "validation",
        "--query-split",
        "validation",
        "--initial-k",
        "3",
        "--expanded-k",
        "8" if args.dataset == "hotpotqa" else "5",
        "--candidate-pool-k",
        "8",
        "--label-strategy",
        "evidence",
        "--run-stability-selection",
        "--output-dir",
        "outputs/runs",
        "--use-run-subdir",
        "--run-name",
        args.run_name,
    ]
    if args.use_openai:
        command.append("--use-openai")
    else:
        command.append("--allow-simple-generator")
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
