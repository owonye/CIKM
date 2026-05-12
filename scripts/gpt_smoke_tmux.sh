#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-cikm_gpt_smoke}"
DATASETS="${DATASETS:-hotpotqa,musique,nq,triviaqa}"
SIZE="${SIZE:-50}"
MODEL="${MODEL:-gpt-4.1-mini-2025-04-14}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gpt_smoke}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed on this machine." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash -lc '
set -euo pipefail
cd \"$(pwd)\"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  echo \"Missing .env with OPENAI_API_KEY\" >&2
  exit 1
fi
set -a
source .env
set +a

mkdir -p \"$OUTPUT_DIR\"
IFS=\",\" read -ra dataset_list <<< \"$DATASETS\"
for dataset in \"\${dataset_list[@]}\"; do
  expanded_k=5
  if [ \"\$dataset\" = \"hotpotqa\" ] || [ \"\$dataset\" = \"musique\" ]; then
    expanded_k=8
  fi

  echo \"[RUN] dataset=\$dataset size=$SIZE model=$MODEL expanded_k=\$expanded_k\"
  python src/run_experiments.py \
    --mode \"\$dataset\" \
    --sizes \"$SIZE\" \
    --doc-limit 20000 \
    --corpus-split validation \
    --query-split validation \
    --initial-k 3 \
    --expanded-k \"\$expanded_k\" \
    --candidate-pool-k 8 \
    --tail-level 0.5 \
    --sufficiency-tolerance 0.02 \
    --label-strategy evidence \
    --use-openai \
    --openai-model \"$MODEL\" \
    --run-stability-selection \
    --use-run-subdir \
    --run-name \"gpt-smoke-\$dataset-s$SIZE\" \
    --output-dir \"$OUTPUT_DIR\" \
    --retrieval-cache-dir \"$OUTPUT_DIR/cache_shared\" \
    --openai-cache-path \"$OUTPUT_DIR/openai_cache_shared.jsonl\"
done

echo \"[DONE] GPT smoke runs complete.\"
exec bash
'"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
