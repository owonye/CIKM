#!/usr/bin/env bash
set -euo pipefail

export SESSION="${SESSION:-gpt_new_baselines}"
export DATASETS="${DATASETS:-hotpotqa,musique,nq,triviaqa}"
export SIZE="${SIZE:-1000}"
export MODEL="${MODEL:-gpt-4.1-mini-2025-04-14}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt_new_baselines}"
export RUN_SUFFIX="${RUN_SUFFIX:-newbaselines}"
export CANDIDATE_POOL_K="${CANDIDATE_POOL_K:-8}"
export TAIL_LEVEL="${TAIL_LEVEL:-0.5}"
export UTILITY_RHO="${UTILITY_RHO:-0.1}"
export UTILITY_ALPHA="${UTILITY_ALPHA:-0.0}"
export UTILITY_BETA="${UTILITY_BETA:-0.0}"
export STABILITY_RHO_GRID="${STABILITY_RHO_GRID:-$UTILITY_RHO}"
export STABILITY_ALPHA_GRID="${STABILITY_ALPHA_GRID:-$UTILITY_ALPHA}"
export STABILITY_BETA_GRID="${STABILITY_BETA_GRID:-$UTILITY_BETA}"
export STABILITY_TAIL_GRID="${STABILITY_TAIL_GRID:-$TAIL_LEVEL}"
export INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"

bash scripts/gpt_smoke_tmux.sh
