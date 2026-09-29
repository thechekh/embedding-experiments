"""What is RAG? Retrieval-augmented generation, built on a laptop.

Article: https://chekh.dev/writing/what-is-rag-retrieval-augmented-generation-built-on-a-laptop/
Run:     uv run python rag.py                (about an hour on a CPU; progress is saved)
         uv run python rag.py --rescore      (score the saved answers again, without the model)
         uv run python rag.py --charts-only  (redraw from results/)
         uv run pytest tests/test_rag.py

A small open-weight model answers SQuAD v2 questions four ways: from memory, with the three
paragraphs BM25 finds, with the three an embedding model finds, and with the paragraph the
question was written about. The answers are scored with SQuAD's exact match and F1; for a
question the paragraph cannot answer, the right reply is "unanswerable".
"""

import json
import sys
import time
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

import llm
import squad
from _common import ACCENT, INK, INK_2, MUTED, arrow, box, machine, save
from retrieval import BM25, encode, rank

SLUG = "what-is-rag-retrieval-augmented-generation-built-on-a-laptop"
RESULTS = Path(__file__).parent / "results"
PARTIAL = RESULTS / f"{SLUG}.partial.json"
DENSE = ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
N_ANSWERABLE, N_UNANSWERABLE, TOP_K = 300, 100, 3
SETUPS = ["closed book", "BM25", "embeddings", "the right paragraph"]

# One instruction and two worked examples, one answered and one declined. Chosen before the run
# on 60 other questions from the same set: asked for "as few words as possible", the model
# answered in whole sentences, and a worked example fixed that.
EXAMPLE = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."
RULE = ("Answer with the shortest phrase from the context that answers the question, not a sentence. "
        "If the context does not contain the answer, reply: unanswerable")
EXAMPLES = (f"Context: {EXAMPLE}\nQuestion: When was the Eiffel Tower completed?\nAnswer: 1889\n\n"
            f"Context: {EXAMPLE}\nQuestion: How tall was the Eiffel Tower when it was completed?\n"
            "Answer: unanswerable")
CLOSED_RULE = ("Answer with the shortest phrase that answers the question, not a sentence. "
               "If you do not know the answer, reply: unanswerable")
CLOSED_EXAMPLES = ("Question: When was the Eiffel Tower completed?\nAnswer: 1889\n\n"
                   "Question: What did Gustave Eiffel eat on the morning the tower opened?\nAnswer: unanswerable")


def prompt(question: str, paragraphs: list[str] | None) -> str:
    """Augment: put the paragraphs above the question. None means closed book."""
    if paragraphs is None:
        rule, examples, context = CLOSED_RULE, CLOSED_EXAMPLES, ""
    else:
        rule, examples = RULE, EXAMPLES
        context = "Context: " + "\n\n".join(paragraphs) + "\n"
    body = f"{rule}\n\nExamples\n{examples}\n\n{context}"
    return f"{body}Question: {question}\nAnswer:"


def clean(reply: str) -> str:
    """The reply's first line, or "" when the model declines to answer."""
    first = reply.strip().split("\n")[0].strip()
    return "" if "unanswerable" in first.lower() else first


def average(values) -> float | None:
    """The mean, or None when there is nothing to average."""
    values = list(values)
    return mean(values) if values else None


def retrieve(paragraphs: list[str], questions: list[squad.Question], k: int = 10) -> dict[str, list[list[int]]]:
    """The top k paragraphs for every question, by BM25 and by the embedding model."""
    ids = list(range(len(paragraphs)))
    bm25 = BM25(paragraphs)
    docs, _ = encode(DENSE[0], paragraphs, "docs", "squad_v2", max_seq_length=512, revision=DENSE[1])
    queries, _ = encode(DENSE[0], [q.text for q in questions], "queries", "squad_v2", max_seq_length=512,
                        revision=DENSE[1])
    return {"BM25": [rank(bm25.scores(q.text), ids, k) for q in questions],
            "embeddings": [rank(docs @ v, ids, k) for v in queries]}


def compute() -> dict:
    RESULTS.mkdir(exist_ok=True)
    partial = json.loads(PARTIAL.read_text(encoding="utf-8")) if PARTIAL.exists() else {}
    paragraphs, questions = squad.load()
    found = retrieve(paragraphs, questions)
    answerable = [i for i, q in enumerate(questions) if q.answers]
    results = {"machine": machine(), "model": llm.MODEL, "revision": llm.REVISION, "embedding model": DENSE[0],
               "embedding revision": DENSE[1], "squad revision": squad.SQUAD[2], "paragraphs": len(paragraphs),
               "questions": len(questions), "answerable": len(answerable),
               "retrieval, all answerable questions": {
                   name: {f"recall@{k}": mean(questions[i].paragraph in lists[i][:k] for i in answerable)
                          for k in (1, 3, 10)} for name, lists in found.items()}}
    print("  retrieval:", results["retrieval, all answerable questions"])

    position = {q.id: i for i, q in enumerate(questions)}
    chosen = squad.sample(questions, N_ANSWERABLE, N_UNANSWERABLE)
    contexts = {"closed book": [None] * len(chosen),
                "BM25": [[paragraphs[i] for i in found["BM25"][position[q.id]][:TOP_K]] for q in chosen],
                "embeddings": [[paragraphs[i] for i in found["embeddings"][position[q.id]][:TOP_K]] for q in chosen],
                "the right paragraph": [[paragraphs[q.paragraph]] for q in chosen]}
    tokenizer, model = llm.load()
    for setup in SETUPS:
        if setup not in partial:
            prompts = [prompt(q.text, c) for q, c in zip(chosen, contexts[setup])]
            start = time.perf_counter()
            replies = llm.generate_all(model, tokenizer, prompts)
            partial[setup] = {"replies": replies, "seconds": time.perf_counter() - start,
                              "prompt tokens": [len(tokenizer(p)["input_ids"]) for p in prompts]}
            PARTIAL.write_text(json.dumps(partial), encoding="utf-8")
        print(f"  {setup}: done")

    results["sample"] = {"answerable": N_ANSWERABLE, "unanswerable": N_UNANSWERABLE, "top k": TOP_K,
                         "ids": [q.id for q in chosen]}
    preds = {setup: [clean(r) for r in partial[setup]["replies"]] for setup in SETUPS}
    results["setups"] = score(chosen, preds, found, position)
    for setup in SETUPS:
        results["setups"][setup].update({"prompt tokens, mean": mean(partial[setup]["prompt tokens"]),
                                         "seconds per question": partial[setup]["seconds"] / len(chosen)})
    results["predictions"] = {setup: dict(zip((q.id for q in chosen), preds[setup])) for setup in SETUPS}
    report(results)
    return results


def score(chosen: list[squad.Question], preds: dict[str, list[str]], found: dict[str, list[list[int]]],
          position: dict[str, int]) -> dict[str, dict]:
    """Exact match, F1 and refusals for every setup. For the two retrievers, the answers split by
    whether the right paragraph was in the top 3, and what the right paragraph alone scored on the
    questions where it was, so that both numbers cover the same questions."""
    alone = {q.id: squad.f1(p, q.answers) for q, p in zip(chosen, preds["the right paragraph"]) if q.answers}
    setups = {}
    for setup in SETUPS:
        with_answer = [(q, p) for q, p in zip(chosen, preds[setup]) if q.answers]
        without = [(q, p) for q, p in zip(chosen, preds[setup]) if not q.answers]
        f1s = [squad.f1(p, q.answers) for q, p in with_answer]
        entry = {"exact match": mean(squad.exact_match(p, q.answers) for q, p in with_answer), "F1": mean(f1s),
                 "declined, of the answerable": sum(p == "" for _, p in with_answer),
                 "declined, of the unanswerable": sum(p == "" for _, p in without)}
        if setup in found:
            hit = [q.paragraph in found[setup][position[q.id]][:TOP_K] for q, _ in with_answer]
            entry.update({"right paragraph in the top 3": mean(hit), "missed": hit.count(False),
                          "F1 when it was": average(f for f, h in zip(f1s, hit) if h),
                          "F1 when it was not": average(f for f, h in zip(f1s, hit) if not h),
                          "F1 of the right paragraph alone, same questions":
                              average(alone[q.id] for (q, _), h in zip(with_answer, hit) if h)})
        setups[setup] = entry
    return setups


def report(results: dict) -> None:
    for setup, entry in results["setups"].items():
        print(f"  {setup:20s} F1 {entry['F1']:.3f}  EM {entry['exact match']:.3f}  declined "
              f"{entry['declined, of the unanswerable']}/{N_UNANSWERABLE} unanswerable, "
              f"{entry['declined, of the answerable']}/{N_ANSWERABLE} answerable  "
              f"{entry['seconds per question']:.2f} s/question")


def rescore(results: dict) -> dict:
    """Score the saved predictions again without the model, after a change to the scoring;
    the prompt sizes and timings are kept from the run."""
    paragraphs, questions = squad.load()
    found = retrieve(paragraphs, questions)
    position = {q.id: i for i, q in enumerate(questions)}
    by_id = {q.id: q for q in questions}
    chosen = [by_id[i] for i in results["sample"]["ids"]]
    preds = {setup: [results["predictions"][setup][q.id] for q in chosen] for setup in SETUPS}
    kept = {setup: {k: results["setups"][setup][k] for k in ("prompt tokens, mean", "seconds per question")}
            for setup in SETUPS}
    results["setups"] = score(chosen, preds, found, position)
    for setup in SETUPS:
        results["setups"][setup].update(kept[setup])
    report(results)
    return results


def draw_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 31)
    ax.axis("off")
    box(ax, 0.5, 12, 9, 7, "question", fontsize=8)
    box(ax, 13.5, 12, 17, 7, "retriever\nBM25 or embeddings", fontsize=8)
    box(ax, 13.5, 1, 17, 6.5, "1,204 Wikipedia\nparagraphs", fontsize=8, dashed=True)
    box(ax, 38, 12, 10, 7, "top 3\nparagraphs", fontsize=8)
    box(ax, 56, 12, 17.5, 7, "prompt: instruction,\nparagraphs, question", fontsize=7.8)
    box(ax, 81, 12, 8, 7, "small\nmodel", accent=True, fontsize=8)
    box(ax, 92.5, 12, 7.5, 7, "answer", fontsize=8)
    arrow(ax, (9.5, 15.5), (13.5, 15.5))
    arrow(ax, (22, 7.5), (22, 12))
    arrow(ax, (30.5, 15.5), (38, 15.5), "retrieve", accent=True)
    arrow(ax, (48, 15.5), (56, 15.5), "augment", accent=True)
    arrow(ax, (73.5, 15.5), (81, 15.5), "generate", accent=True)
    arrow(ax, (89, 15.5), (92.5, 15.5))
    arrow(ax, (5, 19), (64.5, 19), "the question goes into the prompt too", connection="arc3,rad=-0.22",
          offset=(0, 13))
    ax.text(0.5, 30.5, "Retrieval-augmented generation: find the text, paste it into the prompt, let the model answer",
            fontsize=8.5, color=INK, va="top")
    save(fig, SLUG, "pipeline")


def draw_setups(results: dict) -> None:
    setups = results["setups"]
    labels = ["closed book", "BM25,\ntop 3", "embeddings,\ntop 3", "the right\nparagraph"]
    bars = [([setups[s]["F1"] * 100 for s in SETUPS], ACCENT, f"F1 on the {N_ANSWERABLE} answerable questions"),
            ([setups[s]["declined, of the unanswerable"] / N_UNANSWERABLE * 100 for s in SETUPS], INK_2,
             f"replied \"unanswerable\" to the {N_UNANSWERABLE} unanswerable ones: right"),
            ([setups[s]["declined, of the answerable"] / N_ANSWERABLE * 100 for s in SETUPS], MUTED,
             f"replied \"unanswerable\" to the {N_ANSWERABLE} answerable ones: wrong")]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = range(len(SETUPS))
    for k, (values, color, label) in enumerate(bars):
        offset = (k - 1) * 0.27
        ax.bar([i + offset for i in x], values, width=0.26, color=color, label=label)
        for i, v in zip(x, values):
            ax.text(i + offset, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("percent")
    ax.legend(frameon=False, fontsize=7.8, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title("SQuAD v2, Qwen2.5-0.5B on a CPU: the same questions, four ways", fontsize=10, pad=48)
    save(fig, SLUG, "setups")


if __name__ == "__main__":
    path = RESULTS / f"{SLUG}.json"
    if "--charts-only" in sys.argv:
        results = json.loads(path.read_text(encoding="utf-8"))
    elif "--rescore" in sys.argv:
        results = rescore(json.loads(path.read_text(encoding="utf-8")))
        path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {path.name}")
    else:
        results = compute()
        path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        PARTIAL.unlink()
        print(f"  wrote {path.name}")
    draw_pipeline()
    draw_setups(results)
