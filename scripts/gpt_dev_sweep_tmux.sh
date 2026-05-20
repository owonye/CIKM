#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-cikm_gpt_dev_sweep}"
DATASETS="${DATASETS:-hotpotqa,musique,nq,triviaqa}"
SIZE="${SIZE:-300}"
MODEL="${MODEL:-gpt-4.1-mini-2025-04-14}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gpt_sweep}"
RUN_SUFFIX="${RUN_SUFFIX:-dev-sweep}"
RETRIEVAL_CACHE_DIR="${RETRIEVAL_CACHE_DIR:-results/gpt_smoke/cache_shared}"
CANDIDATE_POOL_KS="${CANDIDATE_POOL_KS:-8,12}"
UTILITY_RHOS="${UTILITY_RHOS:-0.0,0.1,0.2,0.5}"
TAIL_LEVELS="${TAIL_LEVELS:-0.25,0.5}"

run_sweep() {
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

  mkdir -p "$OUTPUT_DIR" "$RETRIEVAL_CACHE_DIR"

  IFS="," read -ra dataset_list <<< "$DATASETS"
  IFS="," read -ra pool_list <<< "$CANDIDATE_POOL_KS"
  IFS="," read -ra rho_list <<< "$UTILITY_RHOS"
  IFS="," read -ra tail_list <<< "$TAIL_LEVELS"

  for pool_k in "${pool_list[@]}"; do
    for rho in "${rho_list[@]}"; do
      for tail in "${tail_list[@]}"; do
        tag="pool${pool_k}-rho${rho//./p}-tail${tail//./p}-$RUN_SUFFIX"
        for dataset in "${dataset_list[@]}"; do
          expanded_k=5
          if [ "$dataset" = "hotpotqa" ] || [ "$dataset" = "musique" ]; then
            expanded_k=8
          fi

          run_name="gpt-sweep-$dataset-s$SIZE-$tag"
          log_path="$OUTPUT_DIR/${dataset}_s${SIZE}_${tag}.log"

          echo "[RUN] dataset=$dataset size=$SIZE pool=$pool_k rho=$rho tail=$tail run=$run_name"
          python src/run_experiments.py \
            --mode "$dataset" \
            --sizes "$SIZE" \
            --doc-limit 20000 \
            --corpus-split validation \
            --query-split validation \
            --initial-k 3 \
            --expanded-k "$expanded_k" \
            --candidate-pool-k "$pool_k" \
            --tail-level "$tail" \
            --sufficiency-tolerance 0.02 \
            --utility-rho "$rho" \
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
      done
    done
  done

  echo "[DONE] GPT dev sweep complete."
}

if [ -n "${TMUX:-}" ]; then
  run_sweep
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

tmux new-session -d -s "$SESSION" "cd \"$(pwd)\" && DATASETS=\"$DATASETS\" SIZE=\"$SIZE\" MODEL=\"$MODEL\" OUTPUT_DIR=\"$OUTPUT_DIR\" RUN_SUFFIX=\"$RUN_SUFFIX\" RETRIEVAL_CACHE_DIR=\"$RETRIEVAL_CACHE_DIR\" CANDIDATE_POOL_KS=\"$CANDIDATE_POOL_KS\" UTILITY_RHOS=\"$UTILITY_RHOS\" TAIL_LEVELS=\"$TAIL_LEVELS\" bash scripts/gpt_dev_sweep_tmux.sh; exec bash"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
