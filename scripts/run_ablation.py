import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hotpotqa", "nq"], required=True)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--use-openai", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable,
        "src/run_experiments.py",
        "--mode",
        args.dataset,
        "--sizes",
        str(args.size),
        "--corpus-split",
        "validation",
        "--query-split",
        "validation",
        "--run-ablation",
        "--run-stability-selection",
        "--output-dir",
        "outputs/runs",
        "--use-run-subdir",
        "--run-name",
        "stability-ablation",
    ]
    command.append("--use-openai" if args.use_openai else "--allow-simple-generator")
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
