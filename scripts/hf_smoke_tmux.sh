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
HF_FIX_TORCH_CUDA="${HF_FIX_TORCH_CUDA:-1}"
HF_TORCH_CUDA_INDEX="${HF_TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
HF_TORCH_CUDA_FALLBACK_INDEX="${HF_TORCH_CUDA_FALLBACK_INDEX:-https://download.pytorch.org/whl/cu126}"
VENV_DIR="${VENV_DIR:-.venv}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"

ensure_torch_cuda() {
  if [ "$HF_FIX_TORCH_CUDA" != "1" ]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi

  if python - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    return
  fi

  echo "[WARN] torch CUDA is unavailable; reinstalling torch from $HF_TORCH_CUDA_INDEX"
  python -m pip uninstall -y torch torchvision torchaudio
  if ! python -m pip install --index-url "$HF_TORCH_CUDA_INDEX" torch torchvision torchaudio; then
    echo "[WARN] failed to install torch from $HF_TORCH_CUDA_INDEX; trying $HF_TORCH_CUDA_FALLBACK_INDEX"
    python -m pip install --index-url "$HF_TORCH_CUDA_FALLBACK_INDEX" torch torchvision torchaudio
  fi

  python - <<'PY'
import torch
print("[INFO] torch:", torch.__version__)
print("[INFO] cuda available:", torch.cuda.is_available())
print("[INFO] torch cuda:", torch.version.cuda)
print("[INFO] device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

run_smoke() {
  cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
  fi
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  if [ -n "$EXTRA_PIP_PACKAGES" ]; then
    python -m pip install -U $EXTRA_PIP_PACKAGES
  fi
  ensure_torch_cuda

  mkdir -p "$OUTPUT_DIR"
  export HF_CACHE_PATH
  export HF_INPUT_MAX_TOKENS
  export HF_DOC_MAX_CHARS
  if [ "$CLEAR_RETRIEVAL_CACHE" = "1" ]; then
    rm -rf "$RETRIEVAL_CACHE_DIR"
  fi
  mkdir -p "$RETRIEVAL_CACHE_DIR"

  IFS="," read -ra dataset_list <<< "$DATASETS"
  extra_utility_args=()
  extra_stability_args=()
  if python src/run_experiments.py --help 2>&1 | grep -q -- "--utility-alpha"; then
    extra_utility_args+=(--utility-alpha "$UTILITY_ALPHA" --utility-beta "$UTILITY_BETA")
    extra_stability_args+=(--stability-alpha-grid "$STABILITY_ALPHA_GRID" --stability-beta-grid "$STABILITY_BETA_GRID")
  fi
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
      --tail-level "$TAIL_LEVEL" \
      --sufficiency-tolerance 0.02 \
      --utility-rho "$UTILITY_RHO" \
      "${extra_utility_args[@]}" \
      --stability-rho-grid "$STABILITY_RHO_GRID" \
      "${extra_stability_args[@]}" \
      --stability-tail-grid "$STABILITY_TAIL_GRID" \
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

tmux new-session -d -s "$SESSION" "cd \"$(pwd)\" && DATASETS=\"$DATASETS\" SIZE=\"$SIZE\" HF_MODEL_ID=\"$HF_MODEL_ID\" OUTPUT_DIR=\"$OUTPUT_DIR\" RUN_SUFFIX=\"$RUN_SUFFIX\" RETRIEVAL_CACHE_DIR=\"$RETRIEVAL_CACHE_DIR\" CLEAR_RETRIEVAL_CACHE=\"$CLEAR_RETRIEVAL_CACHE\" HF_MAX_NEW_TOKENS=\"$HF_MAX_NEW_TOKENS\" HF_CACHE_PATH=\"$HF_CACHE_PATH\" HF_INPUT_MAX_TOKENS=\"$HF_INPUT_MAX_TOKENS\" HF_DOC_MAX_CHARS=\"$HF_DOC_MAX_CHARS\" TAIL_LEVEL=\"$TAIL_LEVEL\" UTILITY_RHO=\"$UTILITY_RHO\" UTILITY_ALPHA=\"$UTILITY_ALPHA\" UTILITY_BETA=\"$UTILITY_BETA\" STABILITY_RHO_GRID=\"$STABILITY_RHO_GRID\" STABILITY_ALPHA_GRID=\"$STABILITY_ALPHA_GRID\" STABILITY_BETA_GRID=\"$STABILITY_BETA_GRID\" STABILITY_TAIL_GRID=\"$STABILITY_TAIL_GRID\" HF_FIX_TORCH_CUDA=\"$HF_FIX_TORCH_CUDA\" HF_TORCH_CUDA_INDEX=\"$HF_TORCH_CUDA_INDEX\" HF_TORCH_CUDA_FALLBACK_INDEX=\"$HF_TORCH_CUDA_FALLBACK_INDEX\" VENV_DIR=\"$VENV_DIR\" EXTRA_PIP_PACKAGES=\"$EXTRA_PIP_PACKAGES\" bash scripts/hf_smoke_tmux.sh; exec bash"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
