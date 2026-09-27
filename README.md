# embedding-experiments

The runnable experiments behind the embedding and retrieval articles on
[chekh.dev](https://chekh.dev): open embedding models and a from-scratch BM25 on two
public benchmarks, measured on a CPU. Every number and chart the articles quote comes
from a script here.

| Script | Article | Runtime |
|---|---|---|
| `benchmark_models.py` | [Benchmark embedding models on your data, not the leaderboard](https://chekh.dev/writing/benchmark-embedding-models-on-your-data-not-the-leaderboard/) | first run about 15 min on a 6-core CPU (encodes two corpora with four models and caches the vectors); seconds afterwards |
| `dimensions.py` | [How many embedding dimensions do you actually need?](https://chekh.dev/writing/how-many-embedding-dimensions-do-you-actually-need/) | about 1 min from the cached vectors |
| `hybrid_search.py` | [Hybrid search with RRF: when keyword search still wins](https://chekh.dev/writing/hybrid-search-with-rrf-when-keyword-search-still-wins/) | about 1 min from the cached vectors |

## Run it

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). No API keys, no accounts: the
benchmarks and the models download from the Hugging Face Hub anonymously, about 500 MB
of models and 15 MB of data the first time.

```sh
git clone https://github.com/thechekh/embedding-experiments
cd embedding-experiments
uv sync                              # creates .venv with the pinned versions
uv run pytest                        # BM25, the metrics and RRF against hand-computed values
uv run python benchmark_models.py    # run this one first: it fills results/embeddings/
uv run python dimensions.py
uv run python hybrid_search.py
```

Each script prints its numbers as it goes, writes them to `results/<slug>.json`, and
draws its charts into `charts/<slug>/`; `--charts-only` redraws from the saved numbers.
The embeddings themselves are cached in `results/embeddings/` (not committed), so the
second and third scripts reuse what the first one encoded.

## What is in here

- `retrieval.py` — the shared pieces: the BEIR copies of SciFact and NFCorpus, a BM25
  implementation, dense encoding with a cache, reciprocal rank fusion, nDCG and recall
- `benchmark_models.py` — four models and BM25 on both benchmarks: quality, speed, size,
  and the scores the model cards report for the same data
- `dimensions.py` — the same vectors cut to 8…1,024 dimensions by truncation, PCA and a
  random subspace, plus int8 quantisation
- `hybrid_search.py` — BM25, one dense model and RRF, with a per-query comparison
- `tests/` — pytest: BM25 against the formula by hand, the metrics on toy rankings, RRF's
  worked example, the quantisation round trip
- `_common.py`, `paper.mplstyle`, `fonts/` — chart style, diagram helpers, the site's
  typeface (Lora, SIL Open Font License)
- `results/`, `charts/` — the numbers and charts from the last run

## Change something

- `retrieval.py` — add a model to `MODELS` with the prefixes its card asks for, and it
  joins every table and chart; add a BEIR benchmark name to `BENCHMARKS` in a script
- `hybrid_search.py` — change `DENSE` to another model, or fuse three lists instead of
  two by passing a third ranking to `rrf()`
- `dimensions.py` — add a dimension count to `DIMS`, or another reduction method to the
  three in `compute()`

## Data and licences

SciFact and NFCorpus are used through their BEIR copies on the Hugging Face Hub
(`BeIR/scifact`, `BeIR/nfcorpus`, and the matching `-qrels` sets); see their dataset
cards for the terms. The code is MIT, see `LICENSE`. The Lora font files in `fonts/` are
under the SIL Open Font License.
