"""Hybrid search with RRF: when keyword search still wins.

Article: https://chekh.dev/writing/hybrid-search-with-rrf-when-keyword-search-still-wins/
Run:     uv run python hybrid_search.py                (reuses the cached embeddings;
                                                        otherwise encodes both sets first)
         uv run python hybrid_search.py --charts-only  (redraw from results/)

BM25, one dense model, and their reciprocal rank fusion on SciFact and NFCorpus: recall
at 1 to 100, nDCG@10, a sweep of RRF's one parameter, and a per-query comparison of
where the keywords win and where the vectors do.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import ACCENT, INK, INK_2, MUTED, arrow, box, machine, save
from retrieval import BM25, encode, load_benchmark, ndcg, rank, recall, rrf

SLUG = "hybrid-search-with-rrf-when-keyword-search-still-wins"
DENSE = "BAAI/bge-small-en-v1.5"
BENCHMARKS = ["scifact", "nfcorpus"]
KS = [1, 3, 5, 10, 20, 50, 100]
RRF_K = [1, 10, 60, 200, 1000]
NOT_FOUND = 200  # the rank recorded when no relevant document is in the top 100


def first_relevant(ranking: list[str], relevant: dict[str, int]) -> int:
    return next((i for i, d in enumerate(ranking, start=1) if d in relevant), NOT_FOUND)


def metrics(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict:
    out = {"ndcg@10": float(np.mean([ndcg(r, qrels[q]) for q, r in rankings.items()]))}
    for k in KS:
        out[f"recall@{k}"] = float(np.mean([recall(r, qrels[q], k) for q, r in rankings.items()]))
    return out


def compute() -> dict:
    results: dict = {"machine": machine(), "dense": DENSE, "benchmarks": {}}
    for name in BENCHMARKS:
        benchmark = load_benchmark(name)
        bm25 = BM25(benchmark.docs)
        docs, _ = encode(DENSE, benchmark.docs, "docs", name)
        queries, _ = encode(DENSE, benchmark.queries, "queries", name)
        dense_scores = queries @ docs.T
        lists = {"BM25": {}, "dense": {}, "RRF": {}}
        sweep = {k: {} for k in RRF_K}
        per_query = []
        for i, (qid, text) in enumerate(zip(benchmark.query_ids, benchmark.queries)):
            keyword = rank(bm25.scores(text), benchmark.doc_ids, 100)
            vector = rank(dense_scores[i], benchmark.doc_ids, 100)
            lists["BM25"][qid], lists["dense"][qid] = keyword, vector
            lists["RRF"][qid] = rrf([keyword, vector])[:100]
            for k in RRF_K:
                sweep[k][qid] = rrf([keyword, vector], k=k)[:100]
            relevant = benchmark.qrels[qid]
            per_query.append({"query": text, "bm25": first_relevant(keyword, relevant), "dense": first_relevant(vector, relevant), "rrf": first_relevant(lists["RRF"][qid], relevant)})
        entry = {
            "docs": len(benchmark.docs), "queries": len(benchmark.queries),
            "systems": {system: metrics(r, benchmark.qrels) for system, r in lists.items()},
            "rrf_k": {k: metrics(r, benchmark.qrels)["ndcg@10"] for k, r in sweep.items()},
            "per_query": per_query,
            "wins": {
                "BM25": sum(1 for p in per_query if p["bm25"] < p["dense"]),
                "dense": sum(1 for p in per_query if p["dense"] < p["bm25"]),
                "tie": sum(1 for p in per_query if p["bm25"] == p["dense"]),
                "RRF best or equal": sum(1 for p in per_query if p["rrf"] <= min(p["bm25"], p["dense"])),
            },
        }
        results["benchmarks"][name] = entry
        print(f"{name}: {entry['docs']:,} docs, {entry['queries']} queries")
        for system, m in entry["systems"].items():
            print(f"  {system:6s} nDCG@10 {m['ndcg@10']:.3f}   recall@10 {m['recall@10']:.3f}   recall@100 {m['recall@100']:.3f}")
        print("  first relevant document ranked higher by: " + ", ".join(f"{k} {v}" for k, v in entry["wins"].items()))
        print("  RRF k sweep, nDCG@10: " + ", ".join(f"k={k} {v:.3f}" for k, v in entry["rrf_k"].items()))
        for label, key, other in [("keywords win", "bm25", "dense"), ("vectors win", "dense", "bm25")]:
            best = sorted(per_query, key=lambda p: p[other] - p[key], reverse=True)[:3]
            print(f"  {label}:")
            for p in best:
                print(f"     BM25 rank {p['bm25']:>3}, dense rank {p['dense']:>3}: {p['query'][:90]}")
    return results


def draw_fusion() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(-0.3, 3.4)
    ax.axis("off")
    box(ax, 0.2, 1.2, 1.5, 0.9, "query")
    box(ax, 2.6, 2.2, 2.4, 0.9, "BM25\nkeyword index", fontsize=8)
    box(ax, 2.6, 0.2, 2.4, 0.9, "embedding model\n+ vector index", fontsize=8)
    box(ax, 5.9, 2.2, 2.0, 0.9, "ranked list A\ntop 100 by score", fontsize=8)
    box(ax, 5.9, 0.2, 2.0, 0.9, "ranked list B\ntop 100 by cosine", fontsize=8)
    box(ax, 8.9, 1.2, 2.5, 0.9, "reciprocal rank fusion\none list, ranks only", accent=True, fontsize=8)
    arrow(ax, (1.7, 1.75), (2.6, 2.55))
    arrow(ax, (1.7, 1.55), (2.6, 0.75))
    arrow(ax, (5.0, 2.65), (5.9, 2.65))
    arrow(ax, (5.0, 0.65), (5.9, 0.65))
    arrow(ax, (7.9, 2.55), (8.9, 1.85))
    arrow(ax, (7.9, 0.75), (8.9, 1.45))
    ax.text(10.15, 1.0, "score(d) = 1/(60 + rank in A)" + chr(10) + "         + 1/(60 + rank in B)", ha="center", va="top", fontsize=7.5, color=INK, family="monospace", linespacing=1.4)
    ax.text(0.2, 3.3, "two searches that disagree, combined without comparing their scores", fontsize=8.5, color=INK, va="top")
    save(fig, SLUG, "fusion")


def draw_recall(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for ax, name in zip(axes, BENCHMARKS):
        systems = results["benchmarks"][name]["systems"]
        for system, color, style in [("BM25", INK_2, "-"), ("dense", INK, "-"), ("RRF", ACCENT, "-")]:
            values = [systems[system][f"recall@{k}"] for k in KS]
            label = {"BM25": "BM25", "dense": "dense (bge-small)", "RRF": "RRF of both"}[system]
            ax.plot(KS, values, color=color, linestyle=style, linewidth=2 if system == "RRF" else 1.6, marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.6, label=label)
        ax.set_xscale("log")
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.set_xlabel("k")
        ax.set_title(f"{name}: {results['benchmarks'][name]['docs']:,} documents, {results['benchmarks'][name]['queries']} queries", fontsize=9)
    axes[0].set_ylabel("recall@k")
    axes[0].set_ylim(0, 1)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Share of the relevant documents found in the top k", fontsize=10.5)
    fig.tight_layout()
    save(fig, SLUG, "recall")


def draw_queries(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7))
    for ax, name in zip(axes, BENCHMARKS):
        entry = results["benchmarks"][name]
        rng = np.random.default_rng(0)
        x = np.array([p["bm25"] for p in entry["per_query"]], dtype=float)
        y = np.array([p["dense"] for p in entry["per_query"]], dtype=float)
        jitter = rng.uniform(0.85, 1.15, size=(2, len(x)))
        colors = [ACCENT if a < b else (INK_2 if b < a else MUTED) for a, b in zip(x, y)]
        ax.scatter(x * jitter[0], y * jitter[1], s=12, c=colors, alpha=0.75, linewidths=0)
        ax.plot([1, NOT_FOUND], [1, NOT_FOUND], color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 10, 100, NOT_FOUND], ["1", "10", "100", "none"])
        ax.set_yticks([1, 10, 100, NOT_FOUND], ["1", "10", "100", "none"])
        ax.set_xlabel("rank of the first relevant document, BM25")
        wins = entry["wins"]
        ax.set_title(f"{name}: keywords win {wins['BM25']}, vectors win {wins['dense']}, tie {wins['tie']}", fontsize=8.5)
    axes[0].set_ylabel("rank of the first relevant document, dense")
    fig.suptitle("Every query, both ways: below the line the keywords did better", fontsize=10.5)
    fig.tight_layout()
    save(fig, SLUG, "queries")


if __name__ == "__main__":
    path = Path(__file__).parent / "results" / f"{SLUG}.json"
    if "--charts-only" in sys.argv:
        results = json.loads(path.read_text(encoding="utf-8"))
    else:
        results = compute()
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"  wrote {path.name}")
    draw_fusion()
    draw_recall(results)
    draw_queries(results)
