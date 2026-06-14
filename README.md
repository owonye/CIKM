# Sufficient but Unstable: Stability-Aware Evidence Selection for Anchored Generation in RAG

This repository contains code and scripts for reproducing the experiments in the CIKM 2026 short paper submission. The project implements a stability-aware evidence selection controller for retrieval-augmented generation (RAG): it uses a deterministic sufficiency scorer, then applies a stability gate and targeted repair policy for sufficient-but-unstable evidence.

The main paper path evaluates whether:

- sufficient-but-unstable evidence occurs at non-trivial frequency
- diagnose-then-expand is not enough to repair instability
- targeted evidence selection beats random and next-ranked candidate selection
- gains come from lower-tail anchoring consistency, sufficiency-preserving candidate filtering, and anchor-deficit reduction

The benchmark suite is HotpotQA, MuSiQue, Natural Questions, and TriviaQA.

## Reproduction Scope

The reproducible experiment path uses tracked files under `src/`, `configs/`,
and `repro/`. Paper sanity-check helpers are kept in `repro/`, while generated
outputs are written under `results/`.

## Repository Layout

```text
src/
  data/                        # Dataset helpers and shared dataclass schemas
  retrieval/                   # Retrieval interface and snapshot cache helpers
  generation/                  # Answer normalization and generation-facing helpers
  scoring/                     # Sufficiency, stability, and candidate utility scores
  policies/                    # STOP / EXPAND / SELECT policy wrappers
  eval/                        # Metrics, aggregation, and qualitative export helpers
  calibrate.py                 # Calibrates sufficiency and confidence thresholds
  calibrate_stability.py       # Calibrates stability threshold and utility weights
  evaluate.py                  # Runs RAG baselines and computes EM/F1
  extract_case_analysis.py     # Extracts representative disagreement cases
  summarize_results.py         # Prints baseline summaries from evaluation CSVs
  summarize_failure_modes.py   # Aggregates sufficiency-vs-confidence failure modes
  summarize_stability_results.py # Exports CIKM stability tables and examples
  run_experiments.py           # End-to-end experiment runner
  run_semantic_agreement_batch.py # Runs semantic-agreement checks across datasets
  semantic_agreement_check.py  # LLM-judged semantic agreement sanity check
  rag/                         # Retrieval, signal, and generation utilities

repro/
  semantic_agreement_tmux.sh   # Reproduces the semantic-agreement sanity check
  defense_sanity_checks.py     # CSV-only recovery and support-signal checks

configs/
  base.yaml
  hotpot.yaml
  musique.yaml
  nq.yaml
  triviaqa.yaml
  ablation.yaml

```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For paper-grade generation, set an OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

You can also place the key in a local `.env` file. `.env` is ignored by git.

## Main Experimental Protocol

The paper uses:

- generator: `gpt-4.1-mini`
- prompt version: `short_answer_v2`
- dense encoder: `BAAI/bge-small-en-v1.5`
- corpus split: `validation`
- query split: `validation`
- document slice: `doc_start=0`, `doc_limit=20000`
- initial retrieval budget: `k_init=3`
- candidate pool: `k_pool=8`
- stability score: lower-tail anchoring consistency `C_eta`
- candidate utility: anchor-deficit reduction minus redundancy
- label strategy: `evidence`
- seed: `42`

For a size of 1,000, `run_experiments.py` uses disjoint query slices by default:

- calibration: query indices `0--999`
- evaluation: query indices `1000--1999`

Overlapping calibration/evaluation slices are rejected unless `--allow-overlap-splits` is explicitly set.

## Reproducing the Main Runs

The commands below reproduce the GPT-4.1-mini paper runs. To run the same
pipeline with local Hugging Face generators, replace `--use-openai` with
`--hf-model-id <model-id>`. The paper robustness checks use the same command
shape for Gemma and Qwen, with the model id recorded in each output metadata
file.

### HotpotQA 1000

```bash
python src/run_experiments.py \
  --mode hotpotqa \
  --sizes 1000 \
  --doc-limit 20000 \
  --corpus-split validation \
  --query-split validation \
  --initial-k 3 \
  --expanded-k 8 \
  --label-strategy evidence \
  --use-openai \
  --run-ablation \
  --use-run-subdir \
  --run-name hotpot-v2-conf-target-final-ablation-1000-clean \
  --retrieval-cache-dir results/cache_shared \
  --openai-cache-path results/openai_cache_shared.jsonl
```

### MuSiQue 1000

```bash
python src/run_experiments.py \
  --mode musique \
  --sizes 1000 \
  --doc-limit 20000 \
  --corpus-split validation \
  --query-split validation \
  --initial-k 3 \
  --expanded-k 8 \
  --label-strategy evidence \
  --use-openai \
  --run-ablation \
  --use-run-subdir \
  --run-name musique-v2-conf-target-final-ablation-1000-clean \
  --retrieval-cache-dir results/cache_shared \
  --openai-cache-path results/openai_cache_shared.jsonl
```

### Natural Questions 1000

```bash
python src/run_experiments.py \
  --mode nq \
  --sizes 1000 \
  --doc-limit 20000 \
  --corpus-split validation \
  --query-split validation \
  --initial-k 3 \
  --expanded-k 5 \
  --nq-max-tokens 180 \
  --nq-stride 90 \
  --label-strategy evidence \
  --use-openai \
  --run-ablation \
  --use-run-subdir \
  --run-name nq-v2-k5-chunk180-conf-target-final-ablation-1000-clean \
  --retrieval-cache-dir results/cache_shared \
  --openai-cache-path results/openai_cache_shared.jsonl
```

### TriviaQA 1000

```bash
python src/run_experiments.py \
  --mode triviaqa \
  --sizes 1000 \
  --doc-limit 20000 \
  --corpus-split validation \
  --query-split validation \
  --initial-k 3 \
  --expanded-k 5 \
  --label-strategy evidence \
  --use-openai \
  --run-ablation \
  --use-run-subdir \
  --run-name triviaqa-v2-k5-conf-target-final-ablation-1000-clean \
  --retrieval-cache-dir results/cache_shared \
  --openai-cache-path results/openai_cache_shared.jsonl
```

The shared retrieval and OpenAI caches are optional but useful for repeated
runs. Output metadata records the exact model version, prompt version,
retrieval cache path, OpenAI cache path, calibration source, and run settings.

## Summarizing Results

After a run finishes, summarize the latest matching output directory. The
stability summarizer exports numeric tables used for the main end-to-end
comparison, sufficient-but-unstable rates, and repair-only analysis.

### HotpotQA

```bash
HOTPOT_DIR=$(ls -td results/*hotpot-v2-conf-target-final-ablation-1000-clean* | head -1)

python src/summarize_results.py \
  --input "$HOTPOT_DIR"/eval_hotpotqa_1000.csv

python src/summarize_stability_results.py \
  --input "$HOTPOT_DIR"/eval_hotpotqa_1000.csv \
  --dataset hotpotqa

cat "$HOTPOT_DIR"/ablation_summary_hotpotqa_1000.csv
head -5 "$HOTPOT_DIR"/case_analysis_hotpotqa_1000.csv
```

### Natural Questions

```bash
NQ_DIR=$(ls -td results/*nq-v2-k5-chunk180-conf-target-final-ablation-1000-clean* | head -1)

python src/summarize_results.py \
  --input "$NQ_DIR"/eval_nq_1000.csv

python src/summarize_stability_results.py \
  --input "$NQ_DIR"/eval_nq_1000.csv \
  --dataset nq

cat "$NQ_DIR"/ablation_summary_nq_1000.csv
head -5 "$NQ_DIR"/case_analysis_nq_1000.csv
```

For MuSiQue and TriviaQA, use the same summary commands with
`--dataset musique` or `--dataset triviaqa` and the matching evaluation CSV.
The stability summary writes:

```text
table_main.csv
table_sbu.csv
table_repair.csv
candidate_rank_corr.csv
selection_agreement.csv
result_pattern_check.csv
examples_sbu.jsonl
```

## Failure-Mode Analysis

The paper includes an aggregate analysis of cases where the confidence baseline stops but the sufficiency baseline expands.

```bash
python src/summarize_failure_modes.py \
  --input "$HOTPOT_DIR"/eval_hotpotqa_1000.csv \
  --output "$HOTPOT_DIR"/failure_modes_hotpotqa_1000.csv

python src/summarize_failure_modes.py \
  --input "$NQ_DIR"/eval_nq_1000.csv \
  --output "$NQ_DIR"/failure_modes_nq_1000.csv
```

The script reports:

- confidence premature stops
- sufficiency-baseline premature stops
- premature stops corrected by the sufficiency baseline
- confidence STOP / sufficiency-baseline EXPAND cases
- breakdowns by sufficiency-baseline reason, such as `high_redundancy` and `mixed_insufficiency`

## Stability-Aware Evidence Selection

For the CIKM follow-up setting, `evaluate.py` can additionally run stability-aware baselines that diagnose sufficient-but-unstable evidence and select a targeted repair passage:

Sufficiency scoring is implemented as a deterministic lightweight pipeline in `src/scoring/sufficiency.py`. `LightweightSufficiencyScorer` returns `relevance`, `coverage`, `supportiveness`, and `redundancy` components in `[0,1]`, applies calibrated sufficiency weights, and is shared across the stability-aware baselines during evaluation.

Stability-aware selection uses lower-tail anchoring consistency, filters candidates by `F(D + c, q) >= tau - epsilon_F`, and ranks feasible candidates by anchor-deficit reduction:

```text
U(c | D, q) = ([gamma - C_eta(D, q)]_+ - [gamma - C_eta^+(D, c, q)]_+) - rho R(c, D)
```

```bash
python src/calibrate_stability.py \
  --mode hotpotqa \
  --manifest-path "$HOTPOT_DIR"/manifest_hotpotqa_1000.json \
  --calibration-file "$HOTPOT_DIR"/calib_hotpotqa_1000.json \
  --use-openai \
  --candidate-pool-k 8 \
  --output "$HOTPOT_DIR"/stability_calib_hotpotqa_1000.json

python src/evaluate.py \
  --mode hotpotqa \
  --manifest-path "$HOTPOT_DIR"/manifest_hotpotqa_1000.json \
  --calibration-file "$HOTPOT_DIR"/calib_hotpotqa_1000.json \
  --confidence-calibration-file "$HOTPOT_DIR"/confidence_calib_hotpotqa_1000.json \
  --stability-calibration-file "$HOTPOT_DIR"/stability_calib_hotpotqa_1000.json \
  --use-openai \
  --baselines vanilla_rag,fixed_large_k_rag,structure_aware_adaptive_rag,diagnose_then_expand,random_selection,next_ranked_selection,stability_aware_selection,selection_mean_consistency,selection_no_filter,selection_no_redundancy,oracle_best_candidate \
  --candidate-pool-k 8 \
  --output "$HOTPOT_DIR"/eval_hotpotqa_1000_stability.csv

python src/summarize_stability_results.py \
  --input "$HOTPOT_DIR"/eval_hotpotqa_1000_stability.csv \
  --dataset hotpotqa
```

The resulting CSV includes `anchoring_consistency`, `post_selection_consistency`, `anchor_deficit_reduction`, `recovered`, `diagnostic_generations`, `selected_doc_id`, feasibility flags, and per-candidate details for ranking-quality analysis.

## Defense Sanity Checks

The paper also reports reviewer-facing sanity checks for the stability repair
claim. These scripts assume the evaluation CSVs already exist under
`results/gpt_new_baselines/` or another directory passed through
`--results-root`.

### Semantic Agreement Check

This check samples sufficient-but-unstable cases and compares token-F1 answer
agreement against LLM-judged semantic answer agreement.

```bash
SAMPLE_SIZE=50 SESSION=semantic_agreement_s200 bash repro/semantic_agreement_tmux.sh
```

With the default dataset list, `SAMPLE_SIZE=50` evaluates 200 sampled SBU cases:
50 each from HotpotQA, MuSiQue, Natural Questions, and TriviaQA. The default
baselines are:

- `diagnose_then_expand`
- `selection_max_query_overlap`
- `stability_aware_selection`

Outputs are written to:

```text
results/gpt_new_baselines/semantic_agreement/
```

The combined summary is:

```text
results/gpt_new_baselines/semantic_agreement/semantic_agreement_summary_all_s50.csv
```

### Recovery and Support-Signal Checks

This CSV-only check summarizes recovery, calibrated sufficiency preservation,
and oracle-support signals for initially sufficient-but-unstable cases.

```bash
python repro/defense_sanity_checks.py \
  --results-root results/gpt_new_baselines \
  --output-dir results/gpt_new_baselines/defense_checks \
  --size 1000
```

It writes:

```text
results/gpt_new_baselines/defense_checks/recovery_definition_check.csv
results/gpt_new_baselines/defense_checks/sufficiency_gate_support_check.csv
```

## Quality-Cost Figure

Local plotting helpers and generated figure files used during development are
not part of the public reproduction path. The numeric CSV summaries above are
the reproducible source for checking the reported quality-cost and repair
trade-offs. Reviewers can verify Figure 2 from `table_sbu.csv` and Figure 3
from `table_repair.csv`; the paper figure images themselves are generated
artifacts rather than required executable inputs.

## Output Files

Each run writes artifacts under `results/`, typically including:

- `manifest_<dataset>_<size>.json`
- `calib_<dataset>_<size>.json`
- `confidence_calib_<dataset>_<size>.json`
- `eval_<dataset>_<size>.csv`
- `ablation_summary_<dataset>_<size>.csv`
- `case_analysis_<dataset>_<size>.csv`
- `failure_modes_<dataset>_<size>.csv` when generated
- sidecar `*.meta.json` files

The metadata files record the generator, prompt version, embedding model, calibration source, retrieval cache path, OpenAI cache path, and related run configuration.

## Notes

- Paper-grade runs require `--use-openai`.
- `--allow-simple-generator` is available only for lightweight debugging.
- `label_strategy=hybrid_generation` is supported as an optional sensitivity mode, but the default paper setting is `label_strategy=evidence`.
- The experiments are intentionally controlled: retrieval uses slice-based benchmark corpora and a one-step STOP/EXPAND policy.
