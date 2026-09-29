"""SQuAD's scoring by hand, the prompt, the reply cleaning, and batching that keeps replies in order."""

import pytest

import llm
import rag
import squad


def test_normalize_drops_case_punctuation_and_articles():
    assert squad.normalize("The Eiffel  Tower!") == "eiffel tower"
    assert squad.normalize("an apple, a pear") == "apple pear"


def test_exact_match():
    assert squad.exact_match("the Eiffel Tower", ["Eiffel Tower"]) == 1
    assert squad.exact_match("Eiffel", ["Eiffel Tower"]) == 0
    assert squad.exact_match("1889", ["in 1889", "1889"]) == 1  # any accepted answer will do


def test_f1_by_hand():
    # "Paris, France" against "Paris": precision 1/2, recall 1/1, F1 = 2 * 0.5 * 1 / 1.5
    assert squad.f1("Paris, France", ["Paris"]) == pytest.approx(2 / 3)
    # the closest accepted answer counts: "in 1889" scores 2/3 against "1889", 1/2 against "year 1889"
    assert squad.f1("in 1889", ["the year 1889", "1889"]) == pytest.approx(2 / 3)
    assert squad.f1("Lyon", ["Paris"]) == 0


def test_no_answer_rules():
    # an unanswerable question: only an empty prediction (the model declined) scores
    assert squad.exact_match("", []) == squad.f1("", []) == 1
    assert squad.exact_match("Paris", []) == squad.f1("Paris", []) == 0
    # an answerable question: declining scores nothing
    assert squad.exact_match("", ["Paris"]) == squad.f1("", ["Paris"]) == 0


def test_prompt_puts_the_paragraphs_above_the_question():
    text = rag.prompt("Who built it?", ["First paragraph.", "Second paragraph."])
    assert text.index("First paragraph.") < text.index("Second paragraph.") < text.index("Question: Who built it?")
    assert text.startswith(rag.RULE) and text.endswith("Answer:")
    closed = rag.prompt("Who built it?", None)
    assert closed.startswith(rag.CLOSED_RULE) and "Context" not in closed


def test_clean_keeps_the_first_line_and_reads_a_refusal():
    assert rag.clean("  1889\nIt was completed for the fair.") == "1889"
    assert rag.clean("Unanswerable.") == ""
    assert rag.clean("The context says this is unanswerable") == ""


def test_generate_all_returns_replies_in_the_prompts_order(monkeypatch):
    seen = []

    def fake_greedy(model, tokenizer, prompts, max_new_tokens):
        seen.append(prompts)
        return [p.upper() for p in prompts]

    monkeypatch.setattr(llm, "greedy", fake_greedy)
    prompts = ["ccc", "a", "bbbb", "dd", "e"]
    assert llm.generate_all(None, None, prompts, batch=2) == [p.upper() for p in prompts]
    assert seen[0] == ["a", "e"]  # shortest first, so a batch pads little


def test_generate_all_resumes_without_redoing_saved_batches(monkeypatch):
    calls = []

    def fake_greedy(model, tokenizer, prompts, max_new_tokens):
        calls.append(prompts)
        return [p.upper() for p in prompts]

    monkeypatch.setattr(llm, "greedy", fake_greedy)
    prompts = ["ccc", "a", "bbbb", "dd", "e"]
    saved = ["CCC", "A", None, None, "E"]  # the first batch, ["a", "e"], was done before
    checkpoints = []
    replies = llm.generate_all(None, None, prompts, batch=2, replies=saved, checkpoint=checkpoints.append)
    assert replies == [p.upper() for p in prompts]
    assert calls == [["dd", "ccc"], ["bbbb"]] and len(checkpoints) == 2

