"""Benchmark embedding models on your data, not the leaderboard.

Article: https://chekh.dev/writing/benchmark-embedding-models-on-your-data-not-the-leaderboard/
Run:     uv run python benchmark_models.py                (first run: ~800 MB of downloads,
                                                            then about 40 minutes on a CPU)
         uv run python benchmark_models.py --charts-only  (redraw from results/)

Four open embedding models and BM25 on two public benchmarks, SciFact and NFCorpus:
quality (nDCG@10, recall@10, recall@100), encoding speed on a CPU, and size. The
embeddings are cached under results/embeddings/, so the other experiments reuse them.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import ACCENT, INK, INK_2, MUTED, machine, save
from retrieval import BM25, MODELS, encode, evaluate, load_benchmark, parameters, rank

SLUG = "benchmark-embedding-models-on-your-data-not-the-leaderboard"
BENCHMARKS = ["scifact", "nfcorpus"]
SHORT = {name: name.split("/")[-1] for name in MODELS}


def reported_scores() -> dict[str, dict[str, float]]:
    """nDCG@10 on the same benchmarks as published in each model's card, when listed."""
    from huggingface_hub import ModelCard

    reported: dict[str, dict[str, float]] = {}
    for name in MODELS:
        reported[name] = {}
        try:
            card = ModelCard.load(name).data.to_dict()
        except Exception:  # noqa: BLE001 - a missing card is not an error here
            continue
        for entry in card.get("model-index", []):
            for result in entry.get("results", []):
                dataset = result.get("dataset", {}).get("name", "")
                for metric in result.get("metrics", []):
                    if metric.get("type") == "ndcg_at_10" and dataset in ("MTEB SciFact", "MTEB NFCorpus"):
                        reported[name][dataset.split()[-1].lower()] = float(metric["value"])
    return reported


def compute() -> dict:
    results: dict = {"machine": machine(), "benchmarks": {}, "models": {}, "reported": reported_scores()}
    for name in MODELS:
        results["models"][name] = {"parameters": parameters(name)}
    for benchmark_name in BENCHMARKS:
        benchmark = load_benchmark(benchmark_name)
        relevant = sum(len(v) for v in benchmark.qrels.values())
        entry: dict = {"docs": len(benchmark.docs), "queries": len(benchmark.queries), "relevant_pairs": relevant, "systems": {}}
        print(f"{benchmark_name}: {len(benchmark.docs):,} documents, {len(benchmark.queries)} test queries, {relevant:,} relevant pairs")

        start = time.perf_counter()
        bm25 = BM25(benchmark.docs)
        index_seconds = time.perf_counter() - start
        start = time.perf_counter()
        rankings = {qid: rank(bm25.scores(q), benchmark.doc_ids, 100) for qid, q in zip(benchmark.query_ids, benchmark.queries)}
        query_ms = (time.perf_counter() - start) / len(benchmark.queries) * 1000
        entry["systems"]["BM25"] = {**evaluate(rankings, benchmark.qrels), "index_seconds": index_seconds, "query_ms": query_ms, "docs_per_second": len(benchmark.docs) / index_seconds}
        print(f"  {'BM25':32s} " + row(entry["systems"]["BM25"]))

        for name in MODELS:
            doc_vectors, doc_seconds = encode(name, benchmark.docs, "docs", benchmark_name)
            query_vectors, query_seconds = encode(name, benchmark.queries, "queries", benchmark_name)
            start = time.perf_counter()
            scores = query_vectors @ doc_vectors.T
            rankings = {qid: rank(scores[i], benchmark.doc_ids, 100) for i, qid in enumerate(benchmark.query_ids)}
            search_ms = (time.perf_counter() - start) / len(benchmark.queries) * 1000
            entry["systems"][name] = {
                **evaluate(rankings, benchmark.qrels),
                "dims": int(doc_vectors.shape[1]),
                "docs_per_second": len(benchmark.docs) / doc_seconds,
                "query_ms": query_seconds / len(benchmark.queries) * 1000 + search_ms,
                "storage_mb": doc_vectors.nbytes / 1e6,
            }
            print(f"  {SHORT[name]:32s} " + row(entry["systems"][name]))
        results["benchmarks"][benchmark_name] = entry
    return results


def row(system: dict) -> str:
    return (f"nDCG@10 {system['ndcg@10']:.3f}   recall@10 {system['recall@10']:.3f}   recall@100 {system['recall@100']:.3f}"
            f"   {system['docs_per_second']:6.0f} docs/s   {system['query_ms']:5.1f} ms/query")


def draw_quality(results: dict) -> None:
    systems = ["BM25"] + list(MODELS)
    labels = ["BM25 (keywords)"] + [SHORT[n] for n in MODELS]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    y = list(range(len(systems)))[::-1]
    for benchmark_name, color, marker in [("scifact", ACCENT, "o"), ("nfcorpus", INK_2, "s")]:
        values = [results["benchmarks"][benchmark_name]["systems"][s]["ndcg@10"] for s in systems]
        ax.plot(values, y, marker=marker, color=color, linestyle="none", markersize=7, markeredgecolor="white", markeredgewidth=0.8, label=benchmark_name)
        for v, yi in zip(values, y):
            ax.annotate(f"{v:.2f}", (v, yi), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, color=color)
    ax.set_yticks(y, labels)
    ax.set_xlabel("nDCG@10 (higher is better)")
    ax.set_xlim(0.2, 0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="center")
    ax.set_title("Five systems, two benchmarks: the order is not the same")
    fig.subplots_adjust(left=0.3)
    save(fig, SLUG, "quality")


def draw_speed(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for name in MODELS:
        per = [results["benchmarks"][b]["systems"][name] for b in BENCHMARKS]
        speed = np.mean([p["docs_per_second"] for p in per])
        quality = np.mean([p["ndcg@10"] for p in per])
        params = results["models"][name]["parameters"] / 1e6
        ax.plot(speed, quality, marker="o", color=ACCENT, markersize=6 + params / 12, markeredgecolor="white", linestyle="none")
        right = speed > 1000
        ax.annotate(f"{SHORT[name]}\n{params:.0f}M parameters, {per[0]['dims']} dims", (speed, quality), xytext=(-8 if right else 8, -4), textcoords="offset points", fontsize=7.5, color=INK, va="top", ha="right" if right else "left")
    bm25 = [results["benchmarks"][b]["systems"]["BM25"] for b in BENCHMARKS]
    baseline = np.mean([p["ndcg@10"] for p in bm25])
    ax.axhline(baseline, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("BM25, no model at all", (9000, baseline), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlim(10, 10_000)
    ax.set_ylim(0.43, 0.54)
    ax.set_xlabel("documents encoded per second on a 6-core CPU (log scale)")
    ax.set_ylabel("nDCG@10, mean of both benchmarks")
    ax.set_title("Quality against speed")
    save(fig, SLUG, "speed")


if __name__ == "__main__":
    path = Path(__file__).parent / "results" / f"{SLUG}.json"
    if "--charts-only" in sys.argv:
        results = json.loads(path.read_text(encoding="utf-8"))
    else:
        results = compute()
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"  wrote {path.name}")
    draw_quality(results)
    draw_speed(results)
