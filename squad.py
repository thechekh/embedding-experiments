"""SQuAD v2's development set as a small retrieval corpus, its questions, and its official scoring.

SQuAD v2 (Rajpurkar et al., 2018; CC BY-SA 4.0): questions that crowd workers wrote about
Wikipedia paragraphs, about half of them written so that the paragraph does not answer them.
Downloaded from the Hugging Face Hub at a fixed revision, without an account.
"""

import random
import re
import string
from collections import Counter
from dataclasses import dataclass

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SQUAD = ("rajpurkar/squad_v2", "squad_v2/validation-00000-of-00001.parquet", "3ffb306f725f7d2ce8394bc1873b24868140c412")


@dataclass
class Question:
    id: str
    text: str
    paragraph: int  # the paragraph it was written about, as an index into the corpus
    answers: list[str]  # the accepted answers; empty when the paragraph does not answer it


def load() -> tuple[list[str], list[Question]]:
    """Every paragraph of the development set, each with its article's title, and every question."""
    path = hf_hub_download(SQUAD[0], SQUAD[1], repo_type="dataset", revision=SQUAD[2])
    rows = pq.read_table(path).to_pylist()
    contexts = list(dict.fromkeys(r["context"] for r in rows))
    index = {c: i for i, c in enumerate(contexts)}
    titles = {r["context"]: r["title"].replace("_", " ") for r in rows}
    paragraphs = [f"{titles[c]}: {c}" for c in contexts]
    questions = [Question(r["id"], r["question"].strip(), index[r["context"]], list(r["answers"]["text"]))
                 for r in rows]
    return paragraphs, questions


def sample(questions: list[Question], answerable: int, unanswerable: int, seed: int = 0) -> list[Question]:
    """A fixed random sample: `answerable` questions with an answer, then `unanswerable` without."""
    with_answer = [q for q in questions if q.answers]
    without = [q for q in questions if not q.answers]
    return random.Random(seed).sample(with_answer, answerable) + random.Random(seed).sample(without, unanswerable)


# --- Scoring, as in SQuAD v2's official evaluation script ------------------------------------


def normalize(text: str) -> str:
    """Lower case, no punctuation, no articles, single spaces."""
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    """1 if the prediction equals any accepted answer after normalising. With no accepted
    answer, the only right prediction is an empty one: the model declined."""
    golds = [a for a in answers if normalize(a)] or [""]
    return float(any(normalize(prediction) == normalize(g) for g in golds))


def f1(prediction: str, answers: list[str]) -> float:
    """Word overlap with the closest accepted answer: the harmonic mean of precision and recall."""
    golds = [a for a in answers if normalize(a)] or [""]
    best = 0.0
    for gold in golds:
        pred, true = normalize(prediction).split(), normalize(gold).split()
        if not pred or not true:
            best = max(best, float(pred == true))
            continue
        common = sum((Counter(pred) & Counter(true)).values())
        if common:
            precision, recall = common / len(pred), common / len(true)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best
