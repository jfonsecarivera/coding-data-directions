"""Gemma 3 12B's emotion vectors, the calm direction built from two of them, and the conversations the calm reading is taken on.

The emotion vectors follow Sofroniew et al. (2026) as Han, Chalmers & Izmailov (2026) implement it; their story and dialogue prompts,
171 emotions and 100 topics are copied word for word in data/emotions/. Gemma writes 2 short stories for every emotion and topic
(34,200 stories) and 20 neutral dialogues for every topic (2,000). A text's activation at a layer is its mean residual stream over its
tokens from position 50 on, the text read raw after BOS. An emotion's vector is the mean over its stories minus the mean over all 171
emotions, with the top principal components of the neutral dialogues' activations (the fewest that explain half of their variance)
projected out. The calm direction is the unit calm vector minus the unit desperate vector at layer 24 of 48, scaled to unit length, so
that + is calm.

    python -m cdd.emotions stories      # GPU: the stories and the neutral dialogues
    python -m cdd.emotions vectors      # GPU: the 171 emotion vectors at layer 24
    python -m cdd.emotions direction    # directions/calm_L24.npz
    python -m cdd.emotions targets --transcripts untouched=runs/rejection_test/transcripts/untouched.jsonl ...

Layer L is the residual stream entering block L (HF hidden_states[L]), as everywhere in cdd.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cdd.model import MODEL, Tap, device_of, load_model

OFFSET = 50             # a text's activation is its mean over the tokens from this position on (BOS is position 0)
MIN_STORY_TOKENS = 20   # a story shorter than this, or empty, is sampled again (up to 3 times)
VARIANCE = 0.5          # the neutral components projected out explain at least this share of the neutral activations' variance
STOP_TOKENS = ("<end_of_turn>", "<eos>")


def jl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def n_rows(path) -> int:
    path = Path(path)
    return sum(1 for x in path.open() if x.strip()) if path.exists() else 0


def load_emotions(path) -> list[str]:
    emotions = [e.strip() for e in Path(path).read_text().split(",") if e.strip()]
    assert len(emotions) == 171, len(emotions)
    return emotions


def load_topics(path) -> list[str]:
    topics = [t.strip() for t in Path(path).read_text().splitlines() if t.strip()]
    assert len(topics) == 100, len(topics)
    return topics


def load_prompt(path) -> str:
    """A prompt template ({emotion} and {topic} are filled in with str.format), without the file's final newline."""
    return Path(path).read_text().rstrip("\n")


def story_file(stories: Path, emotion: str) -> Path:
    return Path(stories) / f"{emotion.replace(' ', '_')}.jsonl"


def replace_speakers(text: str) -> str:
    """'Person:' becomes 'Human:' and 'AI:' becomes 'Assistant:' where a line starts with them after any leading whitespace, as Han et
    al. rename the speakers of a neutral dialogue."""
    lines = []
    for line in text.split("\n"):
        body = line.lstrip()
        lead = line[:len(line) - len(body)]
        if body.startswith("Person:"):
            line = lead + "Human:" + body[len("Person:"):]
        elif body.startswith("AI:"):
            line = lead + "Assistant:" + body[len("AI:"):]
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------- stories and neutral dialogues

def chat_prompt(tok, text: str) -> list[int]:
    """Token ids of one user message in Gemma's chat template with the model's turn opened; the template supplies the BOS."""
    p = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tok(p, add_special_tokens=False).input_ids


@torch.no_grad()
def sample(model, tok, prompts: list[list[int]], max_new: int, temperature: float, top_p: float, batch_size: int,
           max_batch_tokens: int) -> list[tuple[str, int, str]]:
    """(text, n_tokens, finish_reason) for each prompt, sampled with top-k off. Prompts are taken in ascending length and left-padded,
    in batches of at most batch_size rows and max_batch_tokens padded prompt-plus-new tokens. A reply ends at <end_of_turn> or <eos>
    ("stop") or at max_new tokens ("length"); n_tokens counts the stop token and the text leaves it out."""
    stop = [tok.convert_tokens_to_ids(t) for t in STOP_TOKENS]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) >= batch_size or (len(cur) + 1) * (len(prompts[i]) + max_new) > max_batch_tokens):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    out = [None] * len(prompts)
    dev = device_of(model)
    for b in batches:
        T = max(len(prompts[i]) for i in b)
        ids = torch.full((len(b), T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(b), T), dtype=torch.long)
        for r, i in enumerate(b):
            ids[r, T - len(prompts[i]):] = torch.tensor(prompts[i], dtype=torch.long)
            mask[r, T - len(prompts[i]):] = 1
        g = model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev), do_sample=True, temperature=temperature, top_p=top_p,
                           top_k=0, max_new_tokens=max_new, eos_token_id=stop, pad_token_id=tok.pad_token_id, repetition_penalty=1.0)
        for i, row in zip(b, g[:, T:].tolist()):
            end = next((j for j, t in enumerate(row) if t in stop), None)
            toks, reason = (row, "length") if end is None else (row[:end + 1], "stop")
            text = tok.decode(toks[:-1] if reason == "stop" else toks, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            out[i] = (text, len(toks), reason)
    return out


def cmd_stories(a) -> None:
    """Writes <out>/stories/<emotion>.jsonl ({topic_idx, rep, story, n_tokens, finish_reason}) for the emotions in the slice, 4 emotions
    per sampling call, and <out>/neutral_dialogues.jsonl ({topic_idx, rep, dialogue}). The torch seed is --seed plus the slice's first
    index, so jobs on different slices draw different samples. Files already complete are skipped."""
    emotions, topics = load_emotions(a.emotion_file), load_topics(a.topic_file)
    story_t, neutral_t = load_prompt(a.story_prompt), load_prompt(a.neutral_prompt)
    lo, _, hi = a.emotions.partition(":")
    lo, hi = int(lo or 0), min(int(hi) if hi else len(emotions), len(emotions))
    out = Path(a.out)
    model, tok = load_model(a.model)
    torch.manual_seed(a.seed + lo)

    def gen(texts: list[str], max_new: int):
        return sample(model, tok, [chat_prompt(tok, t) for t in texts], max_new, a.temperature, a.top_p, a.batch_size, a.max_batch_tokens)

    todo = [e for e in emotions[lo:hi] if n_rows(story_file(out / "stories", e)) < len(topics) * a.reps]
    print(f"[stories] emotions {lo}:{hi}, {len(todo)} to write, {len(topics)} topics x {a.reps} stories each", flush=True)
    lengths, truncated = [], 0
    for c in range(0, len(todo), a.chunk):
        group = todo[c:c + a.chunk]
        jobs = [(e, t, r) for e in group for t in range(len(topics)) for r in range(a.reps)]
        res = dict(zip(jobs, gen([story_t.format(emotion=e, topic=topics[t]) for e, t, _ in jobs], a.story_max_tokens)))
        for _ in range(3):
            short = [j for j, (text, n, _) in res.items() if n < MIN_STORY_TOKENS or not text.strip()]
            if not short:
                break
            res.update(zip(short, gen([story_t.format(emotion=e, topic=topics[t]) for e, t, _ in short], a.story_max_tokens)))
        for e in group:
            rows = []
            for t in range(len(topics)):
                for r in range(a.reps):
                    text, n, reason = res[(e, t, r)]
                    rows.append({"topic_idx": t, "rep": r, "story": text.strip(), "n_tokens": n, "finish_reason": reason})
                    lengths.append(n)
                    truncated += reason == "length"
            write_jl(story_file(out / "stories", e), rows)
        print(f"[stories] {c + len(group)}/{len(todo)} emotions", flush=True)
    if lengths:
        print(f"[stories] {len(lengths)} stories, {np.mean(lengths):.1f} tokens on average, {truncated / len(lengths):.1%} cut at "
              f"{a.story_max_tokens} tokens", flush=True)
    path = out / "neutral_dialogues.jsonl"
    if a.neutral_reps and n_rows(path) < len(topics) * a.neutral_reps:
        jobs = [(t, r) for t in range(len(topics)) for r in range(a.neutral_reps)]
        res = gen([neutral_t.format(topic=topics[t]) for t, _ in jobs], a.neutral_max_tokens)
        write_jl(path, [{"topic_idx": t, "rep": r, "dialogue": replace_speakers(text.strip())} for (t, r), (text, _, _) in zip(jobs, res)])
        print(f"[stories] {len(jobs)} neutral dialogues -> {path}", flush=True)


# ---------------------------------------------------------------- emotion vectors

@torch.no_grad()
def text_means(model, tok, texts: list[str], layer: int) -> np.ndarray:
    """(n, d) float32: each text's mean residual stream entering block `layer` over its tokens from position 50 on (all of them if it
    has 50 or fewer), the text read raw with BOS first and no chat template."""
    out = []
    with Tap(model, layer) as tap:
        for i, text in enumerate(texts):
            ids = [tok.bos_token_id] + tok(text, add_special_tokens=False).input_ids
            h = tap.run(ids).float()
            out.append(h[OFFSET if len(ids) > OFFSET else 0:].mean(0).cpu().numpy())
            if (i + 1) % 2000 == 0:
                print(f"[vectors] layer {layer}: {i + 1}/{len(texts)} texts", flush=True)
    return np.stack(out)


def neutral_components(N: np.ndarray, share: float = VARIANCE) -> tuple[np.ndarray, float]:
    """The top principal axes (rows) of the neutral activations N, the fewest that explain at least `share` of their variance, and the
    share they explain."""
    C = N.astype(np.float64) - N.mean(0, dtype=np.float64)
    _, s, vt = np.linalg.svd(C, full_matrices=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    k = int(np.searchsorted(cum, share)) + 1
    return vt[:k], float(cum[k - 1])


def emotion_vectors(X: np.ndarray, labels: np.ndarray, P: np.ndarray, n_emotions: int) -> tuple[np.ndarray, np.ndarray]:
    """At one layer, from the stories' activations X (labels[i] = the emotion of story i) and the neutral components P: each emotion's
    mean over its stories minus the mean over the emotions, with P projected out (float64), and that mean over the emotions."""
    M = np.stack([X[labels == k].astype(np.float64).mean(0) for k in range(n_emotions)])
    mu = M.mean(0)
    V = M - mu
    return V - (V @ P.T) @ P, mu


def unit_rows(V: np.ndarray) -> np.ndarray:
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-12)


def cmd_vectors(a) -> None:
    """Writes an .npz with emotions (171), layers, vectors (171, n_layers, d; not unit length), unit (the same scaled to unit length),
    global_mean (n_layers, d; the mean over emotions that was subtracted) and n_components (per layer, the neutral components removed)."""
    emotions = load_emotions(a.emotion_file)
    texts, labels = [], []
    for k, e in enumerate(emotions):
        for r in jl(story_file(Path(a.stories), e)):
            if r["story"].strip():
                texts.append(r["story"])
                labels.append(k)
    labels = np.array(labels)
    neutral = [r["dialogue"] for r in jl(a.neutral) if r["dialogue"].strip()]
    layers = [int(x) for x in a.layers.split(",")]
    print(f"[vectors] {len(texts)} stories over {len(set(labels.tolist()))} emotions, {len(neutral)} neutral dialogues, layers {layers}",
          flush=True)
    model, tok = load_model(a.model)
    vecs, units, mus, ks = [], [], [], []
    for L in layers:
        X = text_means(model, tok, texts, L)
        P, share = neutral_components(text_means(model, tok, neutral, L))
        V, mu = emotion_vectors(X, labels, P, len(emotions))
        vecs.append(V.astype(np.float32))
        units.append(unit_rows(V).astype(np.float32))
        mus.append(mu.astype(np.float32))
        ks.append(len(P))
        print(f"[vectors] layer {L}: {len(P)} neutral components removed ({share:.1%} of the neutral variance)", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, emotions=np.array(emotions), layers=np.array(layers), vectors=np.stack(vecs, 1), unit=np.stack(units, 1),
             global_mean=np.stack(mus), n_components=np.array(ks))
    print(f"[vectors] -> {out}", flush=True)


# ---------------------------------------------------------------- the calm direction

def calm_direction(unit: np.ndarray, emotions: list[str], layers: list[int], layer: int) -> np.ndarray:
    """The unit calm vector minus the unit desperate vector at `layer`, scaled to unit length (float32)."""
    j = layers.index(layer)
    v = unit[emotions.index("calm"), j] - unit[emotions.index("desperate"), j]
    return (v / np.linalg.norm(v)).astype(np.float32)


def cmd_direction(a) -> None:
    z = np.load(a.vectors, allow_pickle=False)
    emotions, layers = [str(e) for e in z["emotions"]], [int(x) for x in z["layers"]]
    u = calm_direction(z["unit"], emotions, layers, a.layer)
    j = layers.index(a.layer)
    calm, desperate = z["unit"][emotions.index("calm"), j], z["unit"][emotions.index("desperate"), j]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, u=u, layer=np.array(a.layer))
    print(f"[direction] layer {a.layer}: cos(calm, desperate) {float(calm @ desperate):+.3f}, cos(u, calm) {float(u @ calm):+.3f}, "
          f"cos(u, desperate) {float(u @ desperate):+.3f} -> {out}", flush=True)


# ---------------------------------------------------------------- the conversations the calm reading is taken on

def conversation(row: dict) -> list[dict]:
    """A rejection-test transcript row as chat messages: each user turn followed by the reply to it."""
    msgs = []
    for user, reply in zip(row["user_turns"], row["replies"]):
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": reply["text"]}]
    return msgs


def cmd_targets(a) -> None:
    """Writes {id: <name>/<item>, messages} for every finished conversation of each transcripts file, in the order the files are given
    and, within a file, by conversation id; the first finished copy of an id counts."""
    rows = []
    for spec in a.transcripts:
        name, _, path = spec.partition("=")
        done = {}
        for r in jl(path):
            if r["status"] in ("ok", "context_exhausted"):
                done.setdefault(r["id"], r)
        rows += [{"id": f"{name}/{done[cid].get('item', cid)}", "messages": conversation(done[cid])} for cid in sorted(done)]
        print(f"[targets] {name}: {len(done)} conversations, {sum(len(r['replies']) for r in done.values())} replies", flush=True)
    write_jl(a.out, rows)
    print(f"[targets] {len(rows)} conversations -> {a.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--emotion-file", default="data/emotions/emotions.txt")
        p.add_argument("--model", default=MODEL)

    s = sub.add_parser("stories", help="sample the emotion stories and the neutral dialogues (GPU)")
    common(s)
    s.add_argument("--topic-file", default="data/emotions/topics.txt")
    s.add_argument("--story-prompt", default="data/emotions/story_prompt.txt")
    s.add_argument("--neutral-prompt", default="data/emotions/neutral_prompt.txt")
    s.add_argument("--out", default="runs/emotions")
    s.add_argument("--emotions", default=":", help="a slice of the 171 emotions, e.g. 0:85 and 85:171 for two GPUs")
    s.add_argument("--reps", type=int, default=2, help="stories per emotion and topic")
    s.add_argument("--neutral-reps", type=int, default=20, help="neutral dialogues per topic (0 skips them)")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--top-p", type=float, default=0.9)
    s.add_argument("--story-max-tokens", type=int, default=300)
    s.add_argument("--neutral-max-tokens", type=int, default=1024)
    s.add_argument("--batch-size", type=int, default=128)
    s.add_argument("--max-batch-tokens", type=int, default=64_000)
    s.add_argument("--chunk", type=int, default=4, help="emotions per sampling call")
    s.add_argument("--seed", type=int, default=0)

    v = sub.add_parser("vectors", help="the emotion vectors from the stories and neutral dialogues (GPU)")
    common(v)
    v.add_argument("--stories", default="runs/emotions/stories")
    v.add_argument("--neutral", default="runs/emotions/neutral_dialogues.jsonl", help="all 2,000 neutral dialogues")
    v.add_argument("--layers", default="24", help="comma-separated, 0-47")
    v.add_argument("--out", default="runs/emotions/emotion_vectors.npz")

    d = sub.add_parser("direction", help="calm minus desperate, as a direction file")
    d.add_argument("--vectors", default="runs/emotions/emotion_vectors.npz")
    d.add_argument("--layer", type=int, default=24)
    d.add_argument("--out", default="directions/calm_L24.npz")

    t = sub.add_parser("targets", help="the conversations the calm reading is taken on")
    t.add_argument("--transcripts", nargs="+", required=True, help="name=path of a rejection-test transcripts file, one per model")
    t.add_argument("--out", default="data/calm_target_conversations.jsonl")

    a = ap.parse_args()
    {"stories": cmd_stories, "vectors": cmd_vectors, "direction": cmd_direction, "targets": cmd_targets}[a.cmd](a)


if __name__ == "__main__":
    main()
