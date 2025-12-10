# Vidore Benchmark v3 Evaluation Guide

This repository does not include the Vidore benchmark data or model weights. The following steps show how to evaluate `vidore/colqwen2.5-v0.2` on the Vidore Benchmark v3 collection once you are in an environment with internet access and sufficient GPU resources.

## Prerequisites
- Python 3.10+
- A GPU-capable environment (the model is multimodal and large).
- Access to the Hugging Face Hub (set `HUGGINGFACE_TOKEN` if required).

Install the suggested dependencies declared in `pyproject.toml` using [uv](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Download the benchmark
Vidore Benchmark v3 is hosted on the Hugging Face Hub. Use `datasets` to pull the tasks locally (replace `ocr` with other configs as needed):

```bash
python - <<'PY'
from datasets import load_dataset

# Example: OCR subset; list available configs with `load_dataset(..., trust_remote_code=True).builder_configs`
data = load_dataset("vidore/vidore-benchmark-v3", name="ocr", split="validation")
print(data)
print(data[0])
PY
```

> **Note:** The container used to prepare this repository has outbound network restrictions, so the dataset cannot be downloaded here. Run the commands in an environment with internet access.

## Run a small evaluation sample
Use the helper script added in `scripts/vidore_evaluation.py` to score a few samples and collect lightweight metrics. This example caps the run to five items to validate that the pipeline works before attempting a full sweep.

```bash
python scripts/vidore_evaluation.py \
  --model vidore/colqwen2.5-v0.2 \
  --dataset vidore/vidore-benchmark-v3 \
  --config ocr \
  --split validation \
  --max-samples 5 \
  --output results/colqwen2.5_sample.jsonl
```

The script builds simple prompts from the `question`/`answer` style fields common across the benchmark tasks and reports exact-match accuracy. For multimodal samples containing images, provide `--image-column` so that the script can attach images to the prompt via the Transformers `AutoProcessor` interface.

## Evaluate retrieval runs with NDCG@10
Retrieval-style Vidore configs (for example, `vidore/tabfquad_test`) expose three subsets: `queries`, `corpus`, and `qrels`. Once you have a run file that lists ranked document IDs per query, compute NDCG@10 with:

```bash
python scripts/vidore_ndcg.py \
  --config vidore/tabfquad_test \
  --run results/tabfquad_run.jsonl \
  --k 10 \
  --output results/tabfquad_ndcg.jsonl
```

The run file should be JSONL with one entry per query:

```json
{"query_id": "q_1", "doc_ids": ["doc_45", "doc_72", "doc_8"], "scores": [0.9, 0.4, 0.2]}
```

The `scores` array is optional; ordering of `doc_ids` determines the rank used for NDCG. The script prints the macro-averaged NDCG@K and, if `--output` is provided, writes per-query NDCG scores to a JSONL file for downstream analysis.

## Visualize results
Once you have a JSONL results file (either from the sample run or your own evaluation loop), create a quick chart:

```bash
python scripts/visualize_vidore_results.py \
  --results results/colqwen2.5_sample.jsonl \
  --metric accuracy \
  --output results/colqwen2.5_accuracy.png
```

The visualizer aggregates per-task scores and produces a bar chart.

## Next steps
- Increase `--max-samples` or drop the flag to evaluate the full split.
- Run multiple configs and combine their JSONL files with the visualizer to compare performance across tasks.
- If you already have official Vidore evaluation outputs, point the visualizer at those files to generate plots without rerunning inference.
