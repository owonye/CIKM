# Stability-Aware Evidence Selection for RAG

This repository contains the experimental code for the CIKM follow-up to STAR: a stability-aware evidence selection controller for retrieval-augmented generation (RAG). The code keeps the STAR-style deterministic sufficiency scorer, then adds a stability gate and targeted repair policy for sufficient-but-unstable evidence.

The main paper path evaluates whether:

- sufficient-but-unstable evidence occurs at non-trivial frequency
- diagnose-then-expand is not enough to repair instability
- targeted evidence selection beats random and next-ranked candidate selection
- gains come from lower-tail anchoring consistency, sufficiency-preserving candidate filtering, and anchor-deficit reduction

The benchmark suite is HotpotQA, MuSiQue, Natural Questions, and TriviaQA.
## Repository Layout

```text
src/
  data/                        # Dataset helpers and shared dataclass schemas
  retrieval/                   # Retrieval interface and snapshot cache helpers
  generation/                  # Answer normalization and generation-facing helpers
  scoring/                     # Sufficiency, stability, and candidate utility scores
  policies/                    # STOP / EXPAND / SELECT policy wrappers
  eval/                        # Metrics, aggregation, and qualitative export helpers
  calibrate.py                 # Calibrates STAR and confidence thresholds
  calibrate_stability.py       # Calibrates stability threshold and utility weights
  evaluate.py                  # Runs RAG baselines and computes EM/F1
  extract_case_analysis.py     # Extracts representative disagreement cases
  summarize_results.py         # Prints baseline summaries from evaluation CSVs
  summarize_failure_modes.py   # Aggregates STAR-vs-confidence failure modes
  summarize_stability_results.py # Exports CIKM stability tables and examples
  run_experiments.py           # End-to-end experiment runner
  rag/                         # Retrieval, signal, and generation utilities

scripts/
  run_eval.py                  # End-to-end stability evaluation wrapper
  run_ablation.py              # Ablation wrapper
  make_tables.py               # Paper table export wrapper
  plot_tradeoff.py             # Regenerates the quality-cost figure
  run_hotpotqa_smoke.sh        # Small smoke-test runner

configs/
  base.yaml
  hotpot.yaml
  musique.yaml
  nq.yaml
  triviaqa.yaml
  ablation.yaml

assets/
  rag_tradeoff.png             # Quality-cost figure used in the paper
  rag_tradeoff.pdf             # Vector version of the same figure
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

### HotpotQA 1000 with Full Ablation

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

### Natural Questions 1000 with Full Ablation

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

The shared retrieval and OpenAI caches are optional but useful for repeated runs.

## Summarizing Results

After a run finishes, summarize the latest matching output directory.

### HotpotQA

```bash
HOTPOT_DIR=$(ls -td results/*hotpot-v2-conf-target-final-ablation-1000-clean* | head -1)

python src/summarize_results.py \
  --input "$HOTPOT_DIR"/eval_hotpotqa_1000.csv

cat "$HOTPOT_DIR"/ablation_summary_hotpotqa_1000.csv
head -5 "$HOTPOT_DIR"/case_analysis_hotpotqa_1000.csv
```

### Natural Questions

```bash
NQ_DIR=$(ls -td results/*nq-v2-k5-chunk180-conf-target-final-ablation-1000-clean* | head -1)

python src/summarize_results.py \
  --input "$NQ_DIR"/eval_nq_1000.csv

cat "$NQ_DIR"/ablation_summary_nq_1000.csv
head -5 "$NQ_DIR"/case_analysis_nq_1000.csv
```

## Failure-Mode Analysis

The paper includes an aggregate analysis of cases where the confidence baseline stops but STAR expands.

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
- STAR premature stops
- premature stops corrected by STAR
- confidence STOP / STAR EXPAND cases
- breakdowns by STAR reason, such as `high_redundancy` and `mixed_insufficiency`

## Stability-Aware Evidence Selection

For the CIKM follow-up setting, `evaluate.py` can additionally run stability-aware baselines that diagnose sufficient-but-unstable evidence and select a targeted repair passage:

Sufficiency scoring is implemented as a deterministic lightweight STAR-style pipeline in `src/scoring/sufficiency.py`. `LightweightSufficiencyScorer` returns `relevance`, `coverage`, `supportiveness`, and `redundancy` components in `[0,1]`, applies the calibrated STAR weights, and is shared across the stability-aware baselines during evaluation.

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

## Regenerating the Quality-Cost Figure

The figure in `assets/rag_tradeoff.png` is generated from the paper table values:

```bash
python scripts/plot_tradeoff.py
```

This creates:

- `assets/rag_tradeoff.png`
- `assets/rag_tradeoff.pdf`

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
