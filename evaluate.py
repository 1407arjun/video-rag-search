#!/usr/bin/env python3
"""
Evaluation script for video RAG search.

Measures Precision@k and Recall@k across three strategies:
  1. Caption only        – BM25 over title + description
  2. Caption + transcript – BM25 over title + description + transcript
  3. Current system  – semantic top-N re-ranked by BM25

Usage:
    python evaluate.py
    python evaluate.py --output results.json
"""

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

SECRETS_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"
COLLECTION_NAME = "video_contents"


def load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        sys.exit(f"secrets.toml not found at {SECRETS_PATH}")
    with open(SECRETS_PATH, "rb") as f:
        return tomllib.load(f)


def build_clients(secrets: dict):
    embedding_model = AzureOpenAIEmbeddings(
        api_version="2025-03-01-preview",
        model="text-embedding-3-small",
        azure_endpoint=secrets["llm"]["endpoint"],
        api_key=secrets["llm"]["api_key"],
    )
    q_client = QdrantClient(
        url=secrets["qdrant"]["url"],
        api_key=secrets["qdrant"]["api_key"],
    )
    store = QdrantVectorStore(
        client=q_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )
    return store, q_client


def fetch_all_metadata(q_client: QdrantClient) -> list[dict]:
    records, _ = q_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    return [r.payload.get("metadata", r.payload) for r in records]


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", (text or "").lower()).split()


def _text_title_caption(m: dict) -> str:
    return " ".join(filter(None, [m.get("title", ""), m.get("description", "")]))


def _text_caption_transcript(m: dict) -> str:
    return " ".join(filter(None, [m.get("title", ""), m.get("description", ""), m.get("transcript", "")]))


def _text_full(m: dict) -> str:
    return " ".join(filter(None, [
        m.get("title", ""),
        m.get("description", ""),
        m.get("transcript", ""),
        m.get("visual_description", ""),
    ]))


def retrieve_by_text(all_metadata: list[dict], query: str, text_fn, k: int) -> list[str]:
    corpus = [_tokenize(text_fn(m)) for m in all_metadata]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(all_metadata, scores), key=lambda x: x[1], reverse=True)
    return [m.get("url") for m, _ in ranked[:k]]


def retrieve_hybrid(store: QdrantVectorStore, query: str, retrieval_count: int, k: int) -> list[str]:
    docs = store.similarity_search(query, k=retrieval_count)
    corpus = [_tokenize(_text_full(doc.metadata)) for doc in docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc.metadata.get("url") for doc, _ in ranked[:k]]


def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    effective_k = min(k, len(relevant))
    if effective_k == 0:
        return 0.0
    return sum(1 for url in retrieved[:k] if url in relevant) / effective_k


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for url in retrieved[:k] if url in relevant) / len(relevant)


def f1_at_k(retrieved: list, relevant: set, k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_metrics(results: list[tuple[list, set]], ks: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"P@{k}"] = float(np.mean([precision_at_k(r, rel, k) for r, rel in results]))
        metrics[f"R@{k}"] = float(np.mean([recall_at_k(r, rel, k) for r, rel in results]))
        metrics[f"F1@{k}"] = float(np.mean([f1_at_k(r, rel, k) for r, rel in results]))
    return metrics


def print_table(all_metrics: dict[str, dict[str, float]]) -> None:
    strategy_names = list(all_metrics.keys())
    metric_names = list(next(iter(all_metrics.values())).keys())
    name_w = max(len(n) for n in strategy_names) + 2
    col_w = 10
    header = f"{'Strategy':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in metric_names)
    print("\n" + header)
    print("─" * len(header))
    for name, metrics in all_metrics.items():
        row = f"{name:<{name_w}}" + "".join(f"{metrics[m]:>{col_w}.4f}" for m in metric_names)
        print(row)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate video RAG retrieval quality")
    parser.add_argument("--benchmark", default="eval_benchmark.json")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--retrieval-count", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        sys.exit(f"Benchmark file not found: {benchmark_path}")

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    queries = benchmark.get("queries", [])
    if not queries:
        sys.exit("Benchmark has no queries.")

    secrets = load_secrets()

    print("Connecting to Qdrant and loading embeddings...")
    store, q_client = build_clients(secrets)
    all_metadata = fetch_all_metadata(q_client)
    print(f"  {len(all_metadata)} documents in collection.\n")

    strategies = {
        "Caption only":         lambda q: retrieve_by_text(all_metadata, q, _text_title_caption, 5),
        "Caption + transcript": lambda q: retrieve_by_text(all_metadata, q, _text_caption_transcript, 5),
        "Current system":   lambda q: retrieve_hybrid(store, q, args.retrieval_count, 5),
    }

    all_metrics: dict[str, dict] = {}
    for name, fn in strategies.items():
        print(f"  Running: {name} ...")
        results = [(fn(item["query"]), set(item["relevant_urls"])) for item in queries]
        all_metrics[name] = compute_metrics(results, args.k)

    print_table(all_metrics)

    if args.output:
        out = {
            "strategies": all_metrics,
            "config": {"k": args.k, "retrieval_count": args.retrieval_count},
            "num_queries": len(queries),
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
