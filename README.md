# Copali Experiment

This repository contains helper scripts to evaluate multimodal models on the Vidore Benchmark v3 collection and visualize results.

- See `evaluation/README.md` for detailed steps to run `vidore/colqwen2.5-v0.2` evaluations in an environment with internet and GPU access.
- `scripts/vidore_evaluation.py` downloads benchmark splits, runs generation, and logs predictions.
- `scripts/visualize_vidore_results.py` aggregates JSONL logs and produces quick accuracy charts.
