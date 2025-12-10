"""Evaluate Vidore-style retrieval runs with NDCG@K.

The Vidore Benchmark v3 retrieval configs expose three subsets via
`datasets.load_dataset`:

- Queries (e.g., `{ "query_id": "q_1", "query": "..." }`)
- Corpus (e.g., `{ "doc_id": "doc_45", "image": <PIL.Image>, "text": "..." }`)
- Qrels (e.g., `{ "query_id": "q_1", "doc_id": "doc_45", "score": 1 }`)

This script consumes a JSONL run file containing the ranked document IDs for
each query and reports macro-averaged NDCG@K.

Example run file line:
    {"query_id": "q_1", "doc_ids": ["doc_45", "doc_72", "doc_8"], "scores": [0.9, 0.4, 0.2]}

The `scores` field is optional; ordering of `doc_ids` determines rank.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute NDCG@K for Vidore Benchmark v3 runs")
    parser.add_argument("--dataset", default="vidore/vidore-benchmark-v3", help="Dataset repo id")
    parser.add_argument("--config", required=True, help="Dataset config name, e.g., vidore/tabfquad_test")
    parser.add_argument("--queries-split", default="queries", help="Split containing queries")
    parser.add_argument("--corpus-split", default="corpus", help="Split containing corpus documents")
    parser.add_argument("--qrels-split", default="qrels", help="Split containing relevance judgments")
    parser.add_argument("--run", type=Path, required=True, help="JSONL file with retrieval results")
    parser.add_argument("--k", type=int, default=10, help="Rank cutoff for NDCG")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL file for per-query scores")
    return parser.parse_args()


def load_vidore_qrels(dataset_id: str, config: str, qrels_split: str) -> Dict[str, Dict[str, float]]:
    data = load_dataset(dataset_id, name=config, split=qrels_split, trust_remote_code=True)
    qrels: Dict[str, Dict[str, float]] = {}
    for row in data:
        query_id = row["query_id"]
        doc_id = row["doc_id"]
        score = float(row.get("score", 0.0))
        qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def load_run(path: Path) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = record["query_id"]
            doc_ids = record.get("doc_ids") or record.get("ranked_doc_ids")
            if doc_ids is None:
                raise KeyError("Run file entries must include `doc_ids` or `ranked_doc_ids`.")
            run[query_id] = list(doc_ids)
    return run


def dcg(relevances: Sequence[float]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def compute_ndcg(qrels: Dict[str, Dict[str, float]], run: Dict[str, List[str]], k: int) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for query_id, doc_rels in qrels.items():
        ranked_docs = run.get(query_id, [])
        observed = [float(doc_rels.get(doc_id, 0.0)) for doc_id in ranked_docs[:k]]
        ideal = sorted(doc_rels.values(), reverse=True)[:k]

        dcg_val = dcg(observed)
        idcg_val = dcg(ideal)
        ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0
        scores[query_id] = ndcg
    return scores


def write_query_scores(path: Path, ndcg_by_query: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for query_id, ndcg in ndcg_by_query.items():
            f.write(json.dumps({"query_id": query_id, "ndcg": ndcg}) + "\n")


def main() -> None:
    args = parse_args()

    qrels = load_vidore_qrels(args.dataset, args.config, args.qrels_split)
    run = load_run(args.run)

    ndcg_by_query = compute_ndcg(qrels, run, args.k)
    mean_ndcg = sum(ndcg_by_query.values()) / max(len(ndcg_by_query), 1)

    if args.output:
        write_query_scores(args.output, ndcg_by_query)

    print(f"Evaluated {len(ndcg_by_query)} queries")
    print(f"Mean NDCG@{args.k}: {mean_ndcg:.4f}")
    missing = len([qid for qid in qrels if qid not in run])
    if missing:
        print(f"Warning: {missing} queries had no predictions in the run file")


if __name__ == "__main__":
    main()
