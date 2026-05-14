#!/usr/bin/env bash
set -euo pipefail

# GPT 20-example smoke test wrapper. Runs through scripts/gpt_smoke_tmux.sh
# so server/tmux behavior stays identical to the main GPT smoke workflow.
export SESSION="${SESSION:-cikm_gpt_s20}"
export DATASETS="${DATASETS:-hotpotqa,musique,nq,triviaqa}"
export SIZE="${SIZE:-20}"
export MODEL="${MODEL:-gpt-4.1-mini-2025-04-14}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt_smoke}"
export RUN_SUFFIX="${RUN_SUFFIX:-s20}"
export RETRIEVAL_CACHE_DIR="${RETRIEVAL_CACHE_DIR:-$OUTPUT_DIR/cache_gpt_s20}"
export CLEAR_RETRIEVAL_CACHE="${CLEAR_RETRIEVAL_CACHE:-0}"

bash scripts/gpt_smoke_tmux.sh
