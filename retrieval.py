"""Shared pieces for the three experiments: two public retrieval benchmarks, a BM25
implementation, dense encoding with a cache, reciprocal rank fusion, and the metrics.

Everything downloads from the Hugging Face Hub without an account: the BEIR copies of
SciFact and NFCorpus, and the open embedding models.
"""

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "results" / "embeddings"

# name -> (query prefix, passage prefix): what each model card says it expects
MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
    "BAAI/bge-small-en-v1.5": (
        "Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "sentence-transformers/static-retrieval-mrl-en-v1": ("", ""),  # no attention at all
}


@dataclass
class Benchmark:
    name: str
    doc_ids: list[str]
    docs: list[str]  # title and text joined
    query_ids: list[str]
    queries: list[str]
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)  # query id -> {doc id: grade}


def load_benchmark(name: str) -> Benchmark:
    """A BEIR benchmark: corpus, test queries, and which documents answer which query."""
    corpus = pq.read_table(hf_hub_download(f"BeIR/{name}", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    queries = pq.read_table(hf_hub_download(f"BeIR/{name}", "queries/queries-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    qrels: dict[str, dict[str, int]] = {}
    with open(hf_hub_download(f"BeIR/{name}-qrels", "test.tsv", repo_type="dataset"), encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            query_id, doc_id, grade = line.rstrip("\n").split("\t")
            if int(grade) > 0:
                qrels.setdefault(query_id, {})[doc_id] = int(grade)
    test_queries = [q for q in queries if q["_id"] in qrels]
    return Benchmark(
        name,
        [d["_id"] for d in corpus],
        [f"{d['title']} {d['text']}".strip() for d in corpus],
        [q["_id"] for q in test_queries],
        [q["text"] for q in test_queries],
        qrels,
    )


# --- Keyword search ---------------------------------------------------------------------

STOPWORDS = set("a an and are as at be by for from has have in is it its of on or that the to was were will with".split())
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """The classic keyword scorer: term frequency saturated by k1, normalised by
    document length with b, weighted by how rare each term is in the collection."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.lengths = np.array([len(t) for t in self.tokens])
        self.average_length = self.lengths.mean()
        self.counts = [Counter(t) for t in self.tokens]
        frequency = Counter(term for count in self.counts for term in count)
        n = len(docs)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5))
                    for term, df in frequency.items()}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, count in enumerate(self.counts):
            for term, tf in count.items():
                self.postings.setdefault(term, []).append((i, tf))

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.tokens))
        for term in tokenize(query):
            for i, tf in self.postings.get(term, ()):
                norm = 1 - self.b + self.b * self.lengths[i] / self.average_length
                saturated = tf * (self.k1 + 1) / (tf + self.k1 * norm)
                scores[i] += self.idf[term] * saturated
        return scores


# --- Dense search -----------------------------------------------------------------------


def encode(model_name: str, texts: list[str], kind: str, benchmark: str, max_seq_length: int = 256, chunk: int = 256,
           revision: str | None = None) -> tuple[np.ndarray, float]:
    """Unit-length embeddings for `texts`, cached under results/embeddings/. Returns the
    vectors and the seconds the encoding took (0 when served from the cache)."""
    from sentence_transformers import SentenceTransformer

    slug = model_name.split("/")[-1]
    path = CACHE / f"{benchmark}-{kind}-{slug}.npy"
    meta = path.with_suffix(".json")
    if path.exists():
        return np.load(path), json.loads(meta.read_text())["seconds"]
    query_prefix, passage_prefix = MODELS[model_name]
    prefix = query_prefix if kind == "queries" else passage_prefix
    model = SentenceTransformer(model_name, device="cpu", revision=revision)
    try:
        model.max_seq_length = max_seq_length
    except AttributeError:  # a static model has no sequence limit to set
        pass
    CACHE.mkdir(parents=True, exist_ok=True)
    # Encoded in chunks with a checkpoint after each, so an interrupted run resumes.
    partial, partial_meta = path.with_suffix(".partial.npy"), path.with_suffix(".partial.json")
    done, seconds = [], 0.0
    if partial.exists():
        done, seconds = [np.load(partial)], json.loads(partial_meta.read_text())["seconds"]
    start = sum(len(d) for d in done)
    for i in range(start, len(texts), chunk):
        tick = time.perf_counter()
        done.append(model.encode([prefix + t for t in texts[i:i + chunk]], batch_size=32,
                                 normalize_embeddings=True, show_progress_bar=False))
        seconds += time.perf_counter() - tick
        np.save(partial, np.vstack(done).astype(np.float32))
        partial_meta.write_text(json.dumps({"seconds": seconds}))
    vectors = np.vstack(done).astype(np.float32)
    np.save(path, vectors)
    meta.write_text(json.dumps({"seconds": seconds, "count": len(texts), "dims": int(vectors.shape[1])}))
    partial.unlink(missing_ok=True)
    partial_meta.unlink(missing_ok=True)
    return vectors, seconds


def parameters(model_name: str) -> int:
    from sentence_transformers import SentenceTransformer

    return sum(p.numel() for p in SentenceTransformer(model_name, device="cpu").parameters())


# --- Ranking, fusion, metrics ------------------------------------------------------------


def rank(scores: np.ndarray, doc_ids: list[str], k: int) -> list[str]:
    top = np.argpartition(-scores, k)[:k]
    return [doc_ids[i] for i in top[np.argsort(-scores[top])]]


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Each list votes 1 / (k + rank) for every document it holds."""
    votes: dict[str, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            votes[doc_id] = votes.get(doc_id, 0.0) + 1 / (k + position)
    return sorted(votes, key=votes.get, reverse=True)


def ndcg(ranking: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """Graded relevance, discounted by position, divided by the ideal order."""
    gains = [relevant.get(d, 0) for d in ranking[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))
    best = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / best if best else 0.0


def recall(ranking: list[str], relevant: dict[str, int], k: int) -> float:
    return sum(1 for d in ranking[:k] if d in relevant) / len(relevant)


def evaluate(rankings: dict[str, list[str]],
             qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """Averages over the queries: nDCG@10 and recall at 10 and 100."""
    scores = {"ndcg@10": [], "recall@10": [], "recall@100": []}
    for query_id, ranking in rankings.items():
        relevant = qrels[query_id]
        scores["ndcg@10"].append(ndcg(ranking, relevant, 10))
        scores["recall@10"].append(recall(ranking, relevant, 10))
        scores["recall@100"].append(recall(ranking, relevant, 100))
    return {name: float(np.mean(values)) for name, values in scores.items()}
