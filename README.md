# Copali Experiment

This repository contains helper scripts to evaluate multimodal models on the Vidore Benchmark v3 collection and visualize results.

- See `evaluation/README.md` for detailed steps to run `vidore/colqwen2.5-v0.2` evaluations in an environment with internet and GPU access.
- `scripts/vidore_evaluation.py` downloads benchmark splits, runs generation, and logs predictions.
- `scripts/visualize_vidore_results.py` aggregates JSONL logs and produces quick accuracy charts.

## Environment setup with uv

Use [uv](https://docs.astral.sh/uv/) to create a virtual environment and install the project's Python dependencies:

```bash
# Create and activate a local virtual environment
uv venv
source .venv/bin/activate

# Install dependencies from `pyproject.toml` and generate `uv.lock`
uv sync

# Run helper scripts without manually activating the environment
uv run python scripts/visualize_vidore_results.py --help
```

> **Note:** The container used to prepare this repository cannot reach PyPI, so `uv sync` and locking must be run in an environment with internet access.
