"""The helpers the experiments build on: normalising, int8 quantisation, first-relevant rank."""

import numpy as np

from dimensions import normalise, quantise
from hybrid_search import NOT_FOUND, first_relevant


def test_normalise_gives_unit_vectors():
    v = normalise(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(v, axis=1), 1)


def test_int8_round_trip_is_close():
    rng = np.random.default_rng(0)
    vectors = normalise(rng.standard_normal((50, 64)).astype(np.float32))
    q, scale = quantise(vectors)
    assert q.dtype == np.int8 and np.abs(q).max() == 127
    restored = q.astype(np.float32) / scale
    assert np.abs(restored - vectors).max() < 0.5 / scale + 1e-6
    scores_before = vectors @ vectors.T
    scores_after = normalise(restored) @ normalise(restored).T
    assert np.abs(scores_before - scores_after).max() < 0.02


def test_first_relevant_rank():
    assert first_relevant(["x", "a", "b"], {"a": 1}) == 2
    assert first_relevant(["x", "y"], {"a": 1}) == NOT_FOUND
