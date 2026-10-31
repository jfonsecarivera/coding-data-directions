"""Gemma 3 12B: loading, chat encoding, LoRA targets, and the hook that reads the residual stream entering a block.

Every step of the pipeline uses these helpers, so a conversation is tokenized the same way when it is scored, trained on and read.
"""
from __future__ import annotations

import numpy as np
import torch

MODEL = "google/gemma-3-12b-it"
SEQ_LEN = 4096
LORA_KINDS = ("mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.k_proj", "self_attn.o_proj", "self_attn.q_proj",
              "self_attn.v_proj")
NEUTRAL_MAX_TOKENS = 2000
NEUTRAL_SKIP = 50


def load_model(name: str = MODEL, dtype=torch.bfloat16, device: int | str = 0):
    """The model and tokenizer. AutoModelForCausalLM gives Gemma 3's multimodal class, whose module names match the trained adapters'."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa", device_map={"": device})
    return model, tok


def device_of(model) -> torch.device:
    return next(model.parameters()).device


def lora_targets(model) -> list[str]:
    """Every attention and MLP projection of the language model (336 modules in Gemma 3 12B: 48 blocks x 7), none of the vision tower."""
    return [n for n, _ in model.named_modules() if n.endswith(LORA_KINDS) and ".layers." in n and "vision" not in n]


def render(user: str, assistant: str) -> str:
    """A single-turn conversation in Gemma's chat format, as the answers are stored and trained on."""
    return f"<start_of_turn>user\n{user.strip()}<end_of_turn>\n<start_of_turn>model\n{assistant.strip()}<end_of_turn>\n"


def turns(text: str, start: str = "<start_of_turn>", roles: tuple[str, ...] = ("user", "model"), sep: str = "\n",
          end: str = "<end_of_turn>\n") -> list[tuple[str, str]]:
    """[(role, content)] for every turn start + role + sep + content + end in the text, scanned left to right."""
    out, pos = [], 0
    while True:
        i = text.find(start, pos)
        if i < 0:
            break
        k = i + len(start)
        role = next((r for r in roles if text.startswith(r + sep, k)), None)
        if role is None:
            pos = i + 1
            continue
        c0 = k + len(role) + len(sep)
        j = text.find(end, c0)
        if j < 0:
            break
        out.append((role, text[c0:j]))
        pos = j + len(end)
    return out


def chat_targets(tok, text: str) -> tuple[list[int], list[int]]:
    """Token ids of a conversation in Gemma's chat format, with a 0/1 mask marking the training targets: the model's turns (their text
    and <end_of_turn>). The user's turns, the turn headers and the newlines between turns are context only."""
    enc = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
    ids, mk = [], []
    for role, content in turns(text):
        head, body, end, nl = enc(f"<start_of_turn>{role}\n"), enc(content), enc("<end_of_turn>"), enc("\n")
        target = int(role == "model")
        ids += head + body + end + nl
        mk += [0] * len(head) + [target] * (len(body) + len(end)) + [0] * len(nl)
    return ids, mk


def encode(tok, text: str, seq_len: int = SEQ_LEN) -> tuple[list[int], list[int]]:
    """As trained: BOS first, then the conversation, then EOS (not a target), cut to seq_len tokens in all."""
    ids, mk = chat_targets(tok, text)
    ids, mk = ids + [tok.eos_token_id], mk + [0]
    n = seq_len - 1
    return [tok.bos_token_id] + ids[:n], [0] + mk[:n]


def sft_loss(model, ids: list[int], mask: list[int]) -> torch.Tensor:
    """Cross-entropy summed over the target tokens of one encoded conversation."""
    x = torch.tensor([ids], device=device_of(model))
    y = x[:, 1:].masked_fill(torch.tensor([mask], device=x.device)[:, 1:] == 0, -100)
    logits = model(input_ids=x).logits[:, :-1]
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum",
                                             ignore_index=-100)


def block_of(name: str) -> int:
    """The decoder block index in a parameter name ('...layers.12.mlp...'), or -1."""
    parts = name.split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


def lm_body(model):
    """The decoder stack with .layers and .norm (the language model inside Gemma 3; also text-only models, wrapped in PEFT or not)."""
    for path in ("model.language_model", "language_model.model", "language_model", "model", "model.model"):
        m = model
        try:
            for part in path.split("."):
                m = getattr(m, part)
        except AttributeError:
            continue
        if hasattr(m, "layers") and hasattr(m, "norm"):
            return m
    raise AttributeError("no decoder stack with .layers and .norm")


def dropout_off(model) -> None:
    """Train mode (so gradient checkpointing applies) with every dropout module in eval mode: the same math as eval mode."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()


def set_checkpointing(model, on: bool) -> None:
    inner = model.base_model.model if hasattr(model, "base_model") else model
    if on:
        inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        inner.gradient_checkpointing_disable()


class Tap:
    """The residual stream entering block `layer` (HF hidden_states[layer]). Inside the context the decoder stack is cut to blocks
    0..layer-1, and the input of its final norm, which is then the output of block layer-1, is kept."""

    def __init__(self, model, layer: int):
        self.lm = lm_body(model)
        self.full = self.lm.layers
        self.layer = layer
        self.h = None

    def __enter__(self):
        self.lm.layers = torch.nn.ModuleList(list(self.full)[:self.layer])
        self.handle = self.lm.norm.register_forward_pre_hook(self.hook)
        return self

    def hook(self, module, args):
        self.h = args[0]

    def __exit__(self, *exc):
        self.handle.remove()
        self.lm.layers = self.full

    def run(self, ids: list[int]) -> torch.Tensor:
        self.lm(input_ids=torch.tensor([ids], device=device_of(self.lm)), use_cache=False)
        return self.h[0]


def assistant_spans(ids: list[int], tok) -> list[tuple[int, int]]:
    """(pre, end) for each model turn: pre is the newline after '<start_of_turn>model', the last token before the reply, and end is
    the reply's <end_of_turn>."""
    sot, eot = tok.convert_tokens_to_ids("<start_of_turn>"), tok.convert_tokens_to_ids("<end_of_turn>")
    model_ids = tok("model", add_special_tokens=False).input_ids
    spans = []
    for i, t in enumerate(ids):
        if t == sot and ids[i + 1:i + 1 + len(model_ids)] == model_ids:
            pre = i + len(model_ids) + 1
            end = next((j for j in range(pre + 1, len(ids)) if ids[j] == eot), len(ids))
            if end > pre + 1:
                spans.append((pre, end))
    return spans


def conversation_spans(tok, messages: list[dict], keep_last: bool = True, pos: str = "pre") -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids of a conversation rendered with Gemma's chat template (BOS first), and the token span read in each model turn: the
    token just before the reply (pos="pre", the state after reading the user's message) or the reply's own tokens (pos="body").
    keep_last=False drops the final model turn."""
    ids = tok.apply_chat_template([{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user", "assistant")],
                                  tokenize=True, add_generation_prompt=False)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    sp = assistant_spans(ids, tok)
    if not keep_last:
        sp = sp[:-1]
    body = [(p + 1, e) for p, e in sp if e > p + 1]
    if pos == "pre":
        return ids, [(s - 1, s) for s, _ in body]
    return ids, body


def neutral_spans(tok, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """A neutral dialogue as raw text with BOS (at most 2,000 tokens), read over its tokens from the 50th on."""
    ids = [tok.bos_token_id] + tok(text, add_special_tokens=False).input_ids[:NEUTRAL_MAX_TOKENS]
    return ids, ([(NEUTRAL_SKIP, len(ids))] if len(ids) > NEUTRAL_SKIP + 10 else [(0, len(ids))])


def load_direction(path: str) -> tuple[torch.Tensor, int]:
    """A direction file (.npz): a vector (key u, vector, v or direction) and the layer it lives at (key layer or L). Returned unit-norm."""
    z = np.load(path, allow_pickle=False)
    keys = list(z.keys())
    vk = next(k for k in ("u", "vector", "v", "direction") if k in keys)
    lk = next(k for k in ("layer", "L") if k in keys)
    u = torch.tensor(np.asarray(z[vk], dtype=np.float32)).flatten()
    return u / u.norm(), int(np.asarray(z[lk]).item())
