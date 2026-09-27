"""BM25 matches the textbook formula, the metrics match hand-computed values, and RRF
fuses the way the paper says."""

import math

import numpy as np
import pytest

from retrieval import BM25, evaluate, ndcg, rank, recall, rrf, tokenize

DOCS = ["the cat sat on the mat", "a dog sat on a log", "cats and dogs", "the mat"]


def test_tokenize_drops_stopwords_and_case():
    assert tokenize("The Cat sat, on THE mat!") == ["cat", "sat", "mat"]


def test_bm25_matches_the_formula_by_hand():
    bm25 = BM25(DOCS, k1=1.5, b=0.75)
    scores = bm25.scores("cat mat")
    # document 0 holds both terms once; compute its score by hand
    n, lengths = 4, [len(tokenize(d)) for d in DOCS]
    avg = sum(lengths) / n
    expected = 0.0
    for term, df in (("cat", 1), ("mat", 2)):
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        tf = 1
        expected += idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * lengths[0] / avg))
    assert scores[0] == pytest.approx(expected)
    assert scores[1] == 0  # neither term
    assert scores[3] > 0 and scores[0] > scores[3]  # both terms beat one term


def test_bm25_prefers_rare_terms_and_short_documents():
    bm25 = BM25(DOCS)
    assert bm25.scores("cat")[0] > bm25.scores("mat")[0]  # "cat" is in 1 doc, "mat" in 2
    assert bm25.scores("mat")[3] > bm25.scores("mat")[0]  # the shorter document wins


def test_rank_returns_the_top_k_in_order():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert rank(scores, ["a", "b", "c", "d"], 3) == ["b", "d", "c"]


def test_rrf_worked_example():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]], k=60)
    votes = {"a": 1 / 61 + 1 / 62, "b": 1 / 62, "c": 1 / 63 + 1 / 61, "d": 1 / 63}
    assert fused == sorted(votes, key=votes.get, reverse=True)
    assert fused[0] == "a"  # first and second beats third and first


def test_metrics_by_hand():
    relevant = {"a": 1, "b": 1}
    assert recall(["a", "x", "y"], relevant, 3) == 0.5
    assert ndcg(["a", "b"], relevant) == pytest.approx(1.0)
    assert ndcg(["x", "a"], relevant) == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    graded = {"a": 2, "b": 1}
    assert ndcg(["b", "a"], graded) == pytest.approx((1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))


def test_evaluate_averages_over_queries():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1, "c": 1}}
    rankings = {"q1": ["a"] + [f"x{i}" for i in range(99)], "q2": ["b"] + [f"y{i}" for i in range(99)]}
    scores = evaluate(rankings, qrels)
    assert scores["recall@10"] == pytest.approx(0.75) and scores["recall@100"] == pytest.approx(0.75)
    assert scores["ndcg@10"] == pytest.approx((1 + (1 / (1 + 1 / math.log2(3)))) / 2)
