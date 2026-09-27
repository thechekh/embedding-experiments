"""How many embedding dimensions do you actually need?

Article: https://chekh.dev/writing/how-many-embedding-dimensions-do-you-actually-need/
Run:     uv run python dimensions.py                (reuses the embeddings benchmark_models.py
                                                     cached; otherwise encodes SciFact first)
         uv run python dimensions.py --charts-only  (redraw from results/)

Three models' SciFact embeddings cut down to 8...768 dimensions three ways: keeping the
first k coordinates (what Matryoshka-trained models are built for), PCA (which works for
any model), and a random k-dimensional subspace (the baseline both should beat). Quality
is nDCG@10 on SciFact's 300 test queries. Then the same vectors at full size in int8.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from _common import ACCENT, INK, INK_2, MUTED, machine, plain_numbers, save
from retrieval import encode, evaluate, load_benchmark, rank

SLUG = "how-many-embedding-dimensions-do-you-actually-need"
MODELS_HERE = ["sentence-transformers/static-retrieval-mrl-en-v1", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]
DIMS = [8, 16, 32, 64, 128, 256, 384, 512, 1024]
MATRYOSHKA = {"sentence-transformers/static-retrieval-mrl-en-v1"}  # trained to be truncated
SHORT = {name: name.split("/")[-1] for name in MODELS_HERE}


def normalise(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def quantise(vectors: np.ndarray) -> tuple[np.ndarray, float]:
    """int8 quantisation: one scale for the whole matrix, 127 steps each side."""
    scale = 127 / np.abs(vectors).max()
    return np.round(vectors * scale).astype(np.int8), scale


def score(benchmark, docs: np.ndarray, queries: np.ndarray) -> dict[str, float]:
    scores = normalise(queries) @ normalise(docs).T
    rankings = {qid: rank(scores[i], benchmark.doc_ids, 100)
                for i, qid in enumerate(benchmark.query_ids)}
    return evaluate(rankings, benchmark.qrels)


def compute() -> dict:
    benchmark = load_benchmark("scifact")
    results: dict = {"machine": machine(), "docs": len(benchmark.docs), "queries": len(benchmark.queries), "models": {}}
    rng = np.random.default_rng(0)
    for name in MODELS_HERE:
        docs, _ = encode(name, benchmark.docs, "docs", "scifact")
        queries, _ = encode(name, benchmark.queries, "queries", "scifact")
        full = docs.shape[1]
        pca = PCA(n_components=full, random_state=0).fit(docs)
        docs_pca, queries_pca = pca.transform(docs), pca.transform(queries)
        basis = np.linalg.qr(rng.standard_normal((full, full)))[0]  # random rotation
        docs_random, queries_random = docs @ basis, queries @ basis
        entry: dict = {
            "dims": full,
            "explained_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "full": score(benchmark, docs, queries),
            "truncate": {}, "pca": {}, "random": {},
        }
        for k in [d for d in DIMS if d <= full]:
            entry["truncate"][k] = score(benchmark, docs[:, :k], queries[:, :k])
            entry["pca"][k] = score(benchmark, docs_pca[:, :k], queries_pca[:, :k])
            entry["random"][k] = score(benchmark, docs_random[:, :k],
                                       queries_random[:, :k])
        quantised, scale = quantise(docs)
        entry["int8"] = score(benchmark, quantised.astype(np.float32) / scale, queries)
        # The recommended 128-dimension cut, in float32 and in int8
        cut = "truncate" if name in MATRYOSHKA else "pca"
        cut_docs, cut_queries = (docs, queries) if name in MATRYOSHKA else (docs_pca, queries_pca)
        small, small_scale = quantise(cut_docs[:, :128])
        entry["float_128"] = entry[cut][128]
        entry["int8_128"] = score(benchmark, small.astype(np.float32) / small_scale, cut_queries[:, :128])
        results["models"][name] = entry
        print(f"{SHORT[name]} ({full} dims): full nDCG@10 {entry['full']['ndcg@10']:.3f}, int8 {entry['int8']['ndcg@10']:.3f}; "
              f"128 dims by {cut}: float32 {entry['float_128']['ndcg@10']:.3f}, int8 {entry['int8_128']['ndcg@10']:.3f}")
        for k in entry["truncate"]:
            print(f"   {k:4d} dims   first-k {entry['truncate'][k]['ndcg@10']:.3f}   PCA {entry['pca'][k]['ndcg@10']:.3f}   random {entry['random'][k]['ndcg@10']:.3f}"
                  f"   variance kept by PCA {entry['explained_variance'][k - 1]:.0%}")
    return results


def draw_quality(results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.3), sharey=True)
    for ax, name in zip(axes, MODELS_HERE):
        entry = results["models"][name]
        for method, color, style, label in [("truncate", ACCENT, "-", "first k coordinates"), ("pca", INK, "-", "PCA"), ("random", MUTED, "--", "random subspace")]:
            ks = sorted(int(k) for k in entry[method])
            ax.plot(ks, [entry[method][str(k)]["ndcg@10"] for k in ks], color=color, linestyle=style, linewidth=1.8, marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.6, label=label)
        ax.axhline(entry["full"]["ndcg@10"], color=MUTED, linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 32, 128, 512], ["8", "32", "128", "512"])
        ax.set_title(f"{SHORT[name]}\n{entry['dims']} dims", fontsize=9)
        ax.set_xlabel("dimensions kept")
    axes[0].set_ylabel("nDCG@10 on SciFact")
    axes[0].set_ylim(0, 0.8)
    axes[2].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Fewer dimensions, and how you pick them", fontsize=10.5)
    fig.tight_layout()
    save(fig, SLUG, "quality")


def draw_variance(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    offsets = [(0, 8, "center"), (8, -12, "left"), (-8, -12, "right")]
    for name, color, (dx, dy, ha) in zip(MODELS_HERE, [ACCENT, INK, INK_2], offsets):
        entry = results["models"][name]
        variance = entry["explained_variance"]
        ax.plot(range(1, len(variance) + 1), variance, color=color, linewidth=1.8, label=SHORT[name])
        k = int(np.searchsorted(variance, 0.9)) + 1
        ax.plot([k], [0.9], marker="o", color=color, markersize=4, markeredgecolor="white")
        ax.annotate(f"{k}", (k, 0.9), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=7.5, color=color)
    ax.axhline(0.9, color=MUTED, linewidth=0.8, linestyle="--")
    ax.annotate("90% of the variance", (1.2, 0.9), xytext=(0, 4), textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xscale("log", base=2)
    plain_numbers(ax.xaxis)
    ax.set_xlim(1, 1500)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("principal components kept (log scale)")
    ax.set_ylabel("share of the variance kept")
    ax.set_title("How many directions the embeddings actually use")
    save(fig, SLUG, "variance")


if __name__ == "__main__":
    path = Path(__file__).parent / "results" / f"{SLUG}.json"
    if "--charts-only" in sys.argv:
        results = json.loads(path.read_text(encoding="utf-8"))
    else:
        results = compute()
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"  wrote {path.name}")
        results = json.loads(json.dumps(results))  # string keys, as when reloaded
    draw_quality(results)
    draw_variance(results)
