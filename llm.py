"""The small open-weight model that writes the answers: Qwen2.5-0.5B-Instruct, on the CPU.

Pinned to one revision, so every run loads the same weights. The first run downloads about
1 GB from the Hugging Face Hub, anonymously; later runs load it from the local cache.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"  # batches line up at the end, where generation happens
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).eval()
    return tokenizer, model


def strip(ids: list[int], tokenizer) -> list[int]:
    """Cut a generated sequence at its first end-of-turn token; drop padding."""
    stops = {tokenizer.pad_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    for i, t in enumerate(ids):
        if t in stops:
            return ids[:i]
    return ids


@torch.no_grad()
def greedy(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Always the most likely next token, for a batch of one-turn chats. Returns the replies."""
    chats = [[{"role": "user", "content": p}] for p in prompts]
    batch = tokenizer.apply_chat_template(chats, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True, padding=True)
    # every generation setting spelled out: the model's own defaults would sample
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0,
                         temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id)
    start = batch["input_ids"].shape[1]
    return [tokenizer.decode(strip(row[start:].tolist(), tokenizer)) for row in out]


def generate_all(model, tokenizer, prompts: list[str], max_new_tokens: int = 24, batch: int = 4,
                 replies: list[str | None] | None = None, checkpoint=None) -> list[str]:
    """A reply for every prompt, batched shortest first so that little of a batch is padding.
    To resume, pass the replies saved so far (None where missing) and a function that saves them."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    replies = list(replies) if replies else [None] * len(prompts)
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        if all(replies[i] is not None for i in chunk):
            continue
        for i, reply in zip(chunk, greedy(model, tokenizer, [prompts[i] for i in chunk], max_new_tokens)):
            replies[i] = reply
        if checkpoint:
            checkpoint(replies)
    return replies
