#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-cikm_hf_phi4mini_s100}"
DATASETS="${DATASETS:-hotpotqa,musique,nq,triviaqa}"
SIZE="${SIZE:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hf_phi4mini_s100}"
RETRIEVAL_CACHE_DIR="${RETRIEVAL_CACHE_DIR:-results/gpt_smoke/cache_shared}"
HF_CACHE_PATH="${HF_CACHE_PATH:-$OUTPUT_DIR/hf_cache_shared.jsonl}"

VENV_DIR="${VENV_DIR:-.venv_phi}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-transformers>=4.51,<6}"
MODEL_SPECS="${MODEL_SPECS:-phi4mini=microsoft/Phi-4-mini-instruct}"

HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-64}"
HF_INPUT_MAX_TOKENS="${HF_INPUT_MAX_TOKENS:-8192}"
HF_DOC_MAX_CHARS="${HF_DOC_MAX_CHARS:-1800}"
TAIL_LEVEL="${TAIL_LEVEL:-0.5}"
UTILITY_RHO="${UTILITY_RHO:-0.1}"
STABILITY_RHO_GRID="${STABILITY_RHO_GRID:-0.1}"
STABILITY_TAIL_GRID="${STABILITY_TAIL_GRID:-0.5}"

export SESSION
export DATASETS
export SIZE
export OUTPUT_DIR
export RETRIEVAL_CACHE_DIR
export HF_CACHE_PATH
export VENV_DIR
export EXTRA_PIP_PACKAGES
export MODEL_SPECS
export HF_MAX_NEW_TOKENS
export HF_INPUT_MAX_TOKENS
export HF_DOC_MAX_CHARS
export TAIL_LEVEL
export UTILITY_RHO
export STABILITY_RHO_GRID
export STABILITY_TAIL_GRID

bash scripts/hf_model_sweep_tmux.sh
