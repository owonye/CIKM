#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-cikm_hf_model_sweep}"
DATASETS="${DATASETS:-hotpotqa,musique,nq,triviaqa}"
SIZE="${SIZE:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hf_smoke}"
RETRIEVAL_CACHE_DIR="${RETRIEVAL_CACHE_DIR:-results/gpt_smoke/cache_shared}"
HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-64}"
HF_CACHE_PATH="${HF_CACHE_PATH:-$OUTPUT_DIR/hf_cache_shared.jsonl}"
HF_INPUT_MAX_TOKENS="${HF_INPUT_MAX_TOKENS:-8192}"
HF_DOC_MAX_CHARS="${HF_DOC_MAX_CHARS:-1800}"
TAIL_LEVEL="${TAIL_LEVEL:-0.5}"
UTILITY_RHO="${UTILITY_RHO:-0.1}"
UTILITY_ALPHA="${UTILITY_ALPHA:-0.0}"
UTILITY_BETA="${UTILITY_BETA:-0.0}"
STABILITY_RHO_GRID="${STABILITY_RHO_GRID:-$UTILITY_RHO}"
STABILITY_ALPHA_GRID="${STABILITY_ALPHA_GRID:-$UTILITY_ALPHA}"
STABILITY_BETA_GRID="${STABILITY_BETA_GRID:-$UTILITY_BETA}"
STABILITY_TAIL_GRID="${STABILITY_TAIL_GRID:-$TAIL_LEVEL}"
MODEL_SPECS="${MODEL_SPECS:-qwen7=Qwen/Qwen2.5-7B-Instruct,qwen14=Qwen/Qwen2.5-14B-Instruct,gemma4=google/gemma-4-E4B-it,phi4=microsoft/phi-4,mistral=mistralai/Mistral-7B-Instruct-v0.3}"
HF_FIX_TORCH_CUDA="${HF_FIX_TORCH_CUDA:-1}"
HF_TORCH_CUDA_INDEX="${HF_TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
HF_TORCH_CUDA_FALLBACK_INDEX="${HF_TORCH_CUDA_FALLBACK_INDEX:-https://download.pytorch.org/whl/cu126}"
VENV_DIR="${VENV_DIR:-.venv}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"

run_sweep() {
  cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

  IFS="," read -ra specs <<< "$MODEL_SPECS"
  for spec in "${specs[@]}"; do
    name="${spec%%=*}"
    model_id="${spec#*=}"
    child_session="${SESSION}_${name}_s${SIZE}"

    echo "[RUN] model=$name id=$model_id datasets=$DATASETS size=$SIZE"
    SESSION="$child_session" \
      DATASETS="$DATASETS" \
      SIZE="$SIZE" \
      OUTPUT_DIR="$OUTPUT_DIR" \
      RUN_SUFFIX="${name}-s${SIZE}" \
      RETRIEVAL_CACHE_DIR="$RETRIEVAL_CACHE_DIR" \
      HF_MODEL_ID="$model_id" \
      HF_MAX_NEW_TOKENS="$HF_MAX_NEW_TOKENS" \
      HF_CACHE_PATH="$HF_CACHE_PATH" \
      HF_INPUT_MAX_TOKENS="$HF_INPUT_MAX_TOKENS" \
      HF_DOC_MAX_CHARS="$HF_DOC_MAX_CHARS" \
      TAIL_LEVEL="$TAIL_LEVEL" \
      UTILITY_RHO="$UTILITY_RHO" \
      UTILITY_ALPHA="$UTILITY_ALPHA" \
      UTILITY_BETA="$UTILITY_BETA" \
      STABILITY_RHO_GRID="$STABILITY_RHO_GRID" \
      STABILITY_ALPHA_GRID="$STABILITY_ALPHA_GRID" \
      STABILITY_BETA_GRID="$STABILITY_BETA_GRID" \
      STABILITY_TAIL_GRID="$STABILITY_TAIL_GRID" \
      HF_FIX_TORCH_CUDA="$HF_FIX_TORCH_CUDA" \
      HF_TORCH_CUDA_INDEX="$HF_TORCH_CUDA_INDEX" \
      HF_TORCH_CUDA_FALLBACK_INDEX="$HF_TORCH_CUDA_FALLBACK_INDEX" \
      VENV_DIR="$VENV_DIR" \
      EXTRA_PIP_PACKAGES="$EXTRA_PIP_PACKAGES" \
      bash scripts/hf_smoke_tmux.sh

    while tmux has-session -t "$child_session" 2>/dev/null; do
      sleep 60
    done
    echo "[DONE] model=$name"
  done

  echo "[DONE] HF model sweep complete."
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

tmux new-session -d -s "$SESSION" "cd \"$(pwd)\" && DATASETS=\"$DATASETS\" SIZE=\"$SIZE\" OUTPUT_DIR=\"$OUTPUT_DIR\" RETRIEVAL_CACHE_DIR=\"$RETRIEVAL_CACHE_DIR\" HF_MAX_NEW_TOKENS=\"$HF_MAX_NEW_TOKENS\" HF_CACHE_PATH=\"$HF_CACHE_PATH\" HF_INPUT_MAX_TOKENS=\"$HF_INPUT_MAX_TOKENS\" HF_DOC_MAX_CHARS=\"$HF_DOC_MAX_CHARS\" TAIL_LEVEL=\"$TAIL_LEVEL\" UTILITY_RHO=\"$UTILITY_RHO\" UTILITY_ALPHA=\"$UTILITY_ALPHA\" UTILITY_BETA=\"$UTILITY_BETA\" STABILITY_RHO_GRID=\"$STABILITY_RHO_GRID\" STABILITY_ALPHA_GRID=\"$STABILITY_ALPHA_GRID\" STABILITY_BETA_GRID=\"$STABILITY_BETA_GRID\" STABILITY_TAIL_GRID=\"$STABILITY_TAIL_GRID\" MODEL_SPECS=\"$MODEL_SPECS\" HF_FIX_TORCH_CUDA=\"$HF_FIX_TORCH_CUDA\" HF_TORCH_CUDA_INDEX=\"$HF_TORCH_CUDA_INDEX\" HF_TORCH_CUDA_FALLBACK_INDEX=\"$HF_TORCH_CUDA_FALLBACK_INDEX\" VENV_DIR=\"$VENV_DIR\" EXTRA_PIP_PACKAGES=\"$EXTRA_PIP_PACKAGES\" bash scripts/hf_model_sweep_tmux.sh; exec bash"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
