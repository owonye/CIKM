#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-cikm_run}"
DATASETS="${DATASETS:-hotpotqa,musique,nq,triviaqa}"
SIZE="${SIZE:-100}"
MODEL="${MODEL:-gpt-4.1-mini-2025-04-14}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gpt_smoke}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
RETRIEVAL_CACHE_DIR="${RETRIEVAL_CACHE_DIR:-$OUTPUT_DIR/cache_shared}"
CLEAR_RETRIEVAL_CACHE="${CLEAR_RETRIEVAL_CACHE:-0}"

run_smoke() {
  cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

  if [ ! -d .venv ]; then
    python3 -m venv .venv
  fi
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

  if [ ! -f .env ]; then
    echo "Missing .env with OPENAI_API_KEY" >&2
    exit 1
  fi
  set -a
  source .env
  set +a
  export OPENAI_API_KEY="${OPENAI_API_KEY//$'\r'/}"
  export OPENAI_API_KEY="${OPENAI_API_KEY#"${OPENAI_API_KEY%%[![:space:]]*}"}"
  export OPENAI_API_KEY="${OPENAI_API_KEY%"${OPENAI_API_KEY##*[![:space:]]}"}"

  mkdir -p "$OUTPUT_DIR"
  if [ "$CLEAR_RETRIEVAL_CACHE" = "1" ]; then
    rm -rf "$RETRIEVAL_CACHE_DIR"
  fi
  mkdir -p "$RETRIEVAL_CACHE_DIR"

  IFS="," read -ra dataset_list <<< "$DATASETS"
  for dataset in "${dataset_list[@]}"; do
    expanded_k=5
    if [ "$dataset" = "hotpotqa" ] || [ "$dataset" = "musique" ]; then
      expanded_k=8
    fi

    suffix_part=""
    if [ -n "$RUN_SUFFIX" ]; then
      suffix_part="-$RUN_SUFFIX"
    fi
    run_name="gpt-smoke-$dataset-s$SIZE$suffix_part"
    log_path="$OUTPUT_DIR/${dataset}_s${SIZE}${suffix_part}.log"

    echo "[RUN] dataset=$dataset size=$SIZE model=$MODEL expanded_k=$expanded_k run=$run_name"
    python src/run_experiments.py \
      --mode "$dataset" \
      --sizes "$SIZE" \
      --doc-limit 20000 \
      --corpus-split validation \
      --query-split validation \
      --initial-k 3 \
      --expanded-k "$expanded_k" \
      --candidate-pool-k 8 \
      --tail-level 0.5 \
      --sufficiency-tolerance 0.02 \
      --label-strategy evidence \
      --use-openai \
      --openai-model "$MODEL" \
      --run-stability-selection \
      --use-run-subdir \
      --run-name "$run_name" \
      --output-dir "$OUTPUT_DIR" \
      --retrieval-cache-dir "$RETRIEVAL_CACHE_DIR" \
      --openai-cache-path "$OUTPUT_DIR/openai_cache_shared.jsonl" \
      2>&1 | tee "$log_path"
  done

  echo "[DONE] GPT smoke runs complete."
}

if [ -n "${TMUX:-}" ]; then
  run_smoke
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed on this machine." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "cd \"$(pwd)\" && DATASETS=\"$DATASETS\" SIZE=\"$SIZE\" MODEL=\"$MODEL\" OUTPUT_DIR=\"$OUTPUT_DIR\" RUN_SUFFIX=\"$RUN_SUFFIX\" RETRIEVAL_CACHE_DIR=\"$RETRIEVAL_CACHE_DIR\" CLEAR_RETRIEVAL_CACHE=\"$CLEAR_RETRIEVAL_CACHE\" bash scripts/gpt_smoke_tmux.sh; exec bash"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
