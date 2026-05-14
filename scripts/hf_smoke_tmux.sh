#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-cikm_hf_smoke}"
DATASETS="${DATASETS:-hotpotqa}"
SIZE="${SIZE:-20}"
HF_MODEL_ID="${HF_MODEL_ID:?Set HF_MODEL_ID, e.g. Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hf_smoke}"
RUN_SUFFIX="${RUN_SUFFIX:-hf-s${SIZE}}"
RETRIEVAL_CACHE_DIR="${RETRIEVAL_CACHE_DIR:-results/gpt_smoke/cache_shared}"
CLEAR_RETRIEVAL_CACHE="${CLEAR_RETRIEVAL_CACHE:-0}"
HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-64}"

run_smoke() {
  cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

  if [ ! -d .venv ]; then
    python3 -m venv .venv
  fi
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

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
    safe_model_name="$(echo "$HF_MODEL_ID" | tr '/:' '__')"
    run_name="hf-smoke-$safe_model_name-$dataset-s$SIZE$suffix_part"
    log_path="$OUTPUT_DIR/${safe_model_name}_${dataset}_s${SIZE}${suffix_part}.log"

    echo "[RUN] dataset=$dataset size=$SIZE hf_model=$HF_MODEL_ID expanded_k=$expanded_k run=$run_name"
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
      --hf-model-id "$HF_MODEL_ID" \
      --hf-max-new-tokens "$HF_MAX_NEW_TOKENS" \
      --run-stability-selection \
      --use-run-subdir \
      --run-name "$run_name" \
      --output-dir "$OUTPUT_DIR" \
      --retrieval-cache-dir "$RETRIEVAL_CACHE_DIR" \
      2>&1 | tee "$log_path"
  done

  echo "[DONE] HF smoke runs complete."
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

tmux new-session -d -s "$SESSION" "cd \"$(pwd)\" && DATASETS=\"$DATASETS\" SIZE=\"$SIZE\" HF_MODEL_ID=\"$HF_MODEL_ID\" OUTPUT_DIR=\"$OUTPUT_DIR\" RUN_SUFFIX=\"$RUN_SUFFIX\" RETRIEVAL_CACHE_DIR=\"$RETRIEVAL_CACHE_DIR\" CLEAR_RETRIEVAL_CACHE=\"$CLEAR_RETRIEVAL_CACHE\" HF_MAX_NEW_TOKENS=\"$HF_MAX_NEW_TOKENS\" bash scripts/hf_smoke_tmux.sh; exec bash"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
