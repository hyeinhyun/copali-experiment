"""Aggregate Vidore evaluation logs and plot simple charts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Vidore evaluation JSONL logs")
    parser.add_argument("--results", nargs="+", type=Path, required=True, help="One or more JSONL result files")
    parser.add_argument("--metric", default="accuracy", choices=["accuracy"], help="Metric to plot")
    parser.add_argument("--output", type=Path, default=Path("vidore_results.png"), help="Output image path")
    return parser.parse_args()


def load_rows(paths: Iterable[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def compute_accuracy(rows: List[Dict]) -> Dict[str, float]:
    grouped = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        task = row.get("task", "overall")
        grouped[task]["total"] += 1
        grouped[task]["correct"] += int(row.get("correct", False))

    return {task: counts["correct"] / max(counts["total"], 1) for task, counts in grouped.items()}


def plot_bar(values: Dict[str, float], output: Path, metric: str) -> None:
    labels = list(values.keys())
    scores = [values[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4 + 0.3 * len(labels)))
    ax.barh(labels, scores, color="#2b8cbe")
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1)
    ax.set_title(f"Vidore {metric}")
    for idx, score in enumerate(scores):
        ax.text(score + 0.01, idx, f"{score:.2%}", va="center")
    fig.tight_layout()
    fig.savefig(output)
    print(f"Saved chart to {output}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results)
    if args.metric == "accuracy":
        values = compute_accuracy(rows)
    else:
        raise ValueError(f"Unsupported metric: {args.metric}")

    plot_bar(values, args.output, args.metric)


if __name__ == "__main__":
    main()
