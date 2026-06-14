#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-semantic_agreement}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gpt_new_baselines}"
GENERATION_CACHE="${GENERATION_CACHE:-results/gpt_new_baselines/openai_cache_shared.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gpt_new_baselines/semantic_agreement}"
JUDGE_CACHE="${JUDGE_CACHE:-results/gpt_new_baselines/semantic_judge_cache.jsonl}"
DATASETS="${DATASETS:-hotpotqa,musique,nq,triviaqa}"
BASELINES="${BASELINES:-diagnose_then_expand,selection_max_query_overlap,stability_aware_selection}"
SIZE="${SIZE:-1000}"
SAMPLE_SIZE="${SAMPLE_SIZE:-25}"
SEED="${SEED:-42}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"
PROMPT_VERSION="${PROMPT_VERSION:-short_answer_v3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

run_check() {
  cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

  if [ ! -d "$VENV_DIR" ]; then
    echo "[ERROR] virtualenv not found: $VENV_DIR" >&2
    exit 1
  fi
  source "$VENV_DIR/bin/activate"

  if [ "$INSTALL_REQUIREMENTS" = "1" ]; then
    python -m pip install -r requirements.txt
  fi

  mkdir -p "$OUTPUT_DIR"
  log_path="$OUTPUT_DIR/semantic_agreement_s${SAMPLE_SIZE}.log"

  cmd=(
    python src/run_semantic_agreement_batch.py
    --results-root "$RESULTS_ROOT"
    --generation-cache "$GENERATION_CACHE"
    --output-dir "$OUTPUT_DIR"
    --judge-cache "$JUDGE_CACHE"
    --datasets "$DATASETS"
    --baselines "$BASELINES"
    --size "$SIZE"
    --sample-size "$SAMPLE_SIZE"
    --seed "$SEED"
    --judge-model "$JUDGE_MODEL"
    --prompt-version "$PROMPT_VERSION"
  )

  if [ "$DRY_RUN" = "1" ]; then
    cmd+=(--dry-run)
  fi

  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$log_path"
}

if [ -n "${TMUX:-}" ]; then
  run_check
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

tmux new-session -d -s "$SESSION" \
  "cd \"$(pwd)\" && SESSION=\"$SESSION\" RESULTS_ROOT=\"$RESULTS_ROOT\" GENERATION_CACHE=\"$GENERATION_CACHE\" OUTPUT_DIR=\"$OUTPUT_DIR\" JUDGE_CACHE=\"$JUDGE_CACHE\" DATASETS=\"$DATASETS\" BASELINES=\"$BASELINES\" SIZE=\"$SIZE\" SAMPLE_SIZE=\"$SAMPLE_SIZE\" SEED=\"$SEED\" JUDGE_MODEL=\"$JUDGE_MODEL\" PROMPT_VERSION=\"$PROMPT_VERSION\" VENV_DIR=\"$VENV_DIR\" INSTALL_REQUIREMENTS=\"$INSTALL_REQUIREMENTS\" DRY_RUN=\"$DRY_RUN\" bash repro/semantic_agreement_tmux.sh; exec bash"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Log path: $OUTPUT_DIR/semantic_agreement_s${SAMPLE_SIZE}.log"
