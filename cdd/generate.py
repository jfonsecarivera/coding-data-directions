"""Gemma 3 12B's own answers to a request set, or a trained copy's (a LoRA adapter), sampled with vLLM, and the candidates file that the
filters and the score read.

  sample  n answers per request with the model's own sampling (its generation config: temperature 1.0, top-k 64, top-p 0.95), at most
          2,048 new tokens, the request as the only user turn (no system prompt). Every answer is its own request (n = 1) with the
          end-of-turn and end-of-text tokens as stop ids. Requests can be split over --nshards processes by row; the vLLM seed is the
          shard index. Resumable: answers already in --out are skipped.
  cands   every finished answer (it ended on a stop token and is not empty) as a candidate {"cid", "pid", "type", "src", "round", "k",
          "n_reply_tok", "turns", "text"}, with "text" the conversation in Gemma's chat format (cdd.model.render). From round 1 on, an
          answer identical to an earlier candidate for the same request (the --seen files) or to another answer of the round is dropped.

    python -m cdd.generate sample --requests runs/code/requests.jsonl --n 8 --out runs/code/r0/gen/s0.jsonl
    python -m cdd.generate cands --requests runs/code/requests.jsonl --gen runs/code/r0/gen --round 0 --out runs/code/r0/candidates.jsonl
    python -m cdd.generate sample --requests runs/code/requests.jsonl --n 4 --adapter runs/code/copy1/adapter --out runs/code/r1/gen/s0.jsonl
    python -m cdd.generate cands --requests runs/code/requests.jsonl --gen runs/code/r1/gen --round 1 --out runs/code/r1/candidates.jsonl \
        --seen runs/code/r0/candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from cdd.model import MODEL, render
from cdd.pool import read_jsonl, write_jsonl

MAX_TOKENS = 2048
CHUNK = 3000   # requests per vLLM call; the output file is flushed after each
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}   # Gemma 3 12B's generation config, which the answers were sampled with


def plan(requests: list[dict], n: int, shard: int, nshards: int, order_seed: int) -> list[tuple[dict, int]]:
    """The (request, k) pairs one shard samples: the requests in a seeded random order (scheduling only), every nshards-th from row
    `shard`, each with k = 0..n-1."""
    rows = list(requests)
    random.Random(order_seed).shuffle(rows)
    rows = [r for i, r in enumerate(rows) if i % nshards == shard]
    return [(r, k) for r in rows for k in range(n)]


def cmd_sample(a) -> None:
    out = Path(a.out)
    done = {(x["pid"], x["k"]) for x in read_jsonl(out)}
    todo = [(r, k) for r, k in plan(read_jsonl(a.requests), a.n, a.shard, a.nshards, a.order_seed) if (r["pid"], k) not in done]
    print(f"[gen] shard {a.shard}/{a.nshards}: {len(done)} answers done, {len(todo)} to sample", flush=True)
    if not todo:
        return
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")   # Gemma 3's attention backend in vLLM 0.11, as the answers were made
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from vllm import LLM  # noqa: PLC0415
    kw = {"enable_lora": True, "max_lora_rank": 64} if a.adapter else {}
    llm = LLM(model=snapshot_download(MODEL), dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92, seed=a.shard,
              enforce_eager=True, limit_mm_per_prompt={"image": 0}, generation_config="auto", max_num_seqs=256, **kw)
    lora = None
    if a.adapter:
        from vllm.lora.request import LoRARequest  # noqa: PLC0415
        lora = LoRARequest("copy", 1, a.adapter)
    sp = llm.get_default_sampling_params()
    got = {"temperature": sp.temperature, "top_p": sp.top_p, "top_k": sp.top_k}
    if got != SAMPLING:
        raise SystemExit(f"the model's generation config gives {got}; the answers were sampled with {SAMPLING}")
    eos = llm.llm_engine.model_config.try_get_generation_config().get("eos_token_id")
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    sp.stop_token_ids = sorted(set(eos) | {llm.get_tokenizer().eos_token_id})
    sp.max_tokens = a.max_tokens
    sp.n = 1   # one request per answer: vLLM 0.11 drops the generation config's stop ids for the children of an n > 1 request
    print(f"[gen] sampling {sp}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, ntok = time.time(), 0
    with out.open("a") as f:
        for c0 in range(0, len(todo), CHUNK):
            part = todo[c0:c0 + CHUNK]
            res = llm.chat([[{"role": "user", "content": r["prompt"]}] for r, _ in part], sp, use_tqdm=False, lora_request=lora)
            for (r, k), o in zip(part, res):
                x = o.outputs[0]
                ntok += len(x.token_ids)
                f.write(json.dumps({"pid": r["pid"], "k": k, "reply": x.text.strip(), "finish": x.finish_reason,
                                    "n_tokens": len(x.token_ids)}, ensure_ascii=False) + "\n")
            f.flush()
            dt = time.time() - t0
            print(f"[gen] {c0 + len(part)}/{len(todo)} ({dt:.0f}s, {ntok / max(dt, 1):.0f} generated tok/s)", flush=True)
    print("[gen] done", flush=True)


def reply_of(c: dict) -> str:
    return c["turns"][1]["content"].strip()


def cmd_cands(a) -> None:
    reqs = {r["pid"]: r for r in read_jsonl(a.requests)}
    gen = {}
    for p in sorted(Path(a.gen).glob("*.jsonl")):
        for r in read_jsonl(p):
            gen.setdefault((r["pid"], r["k"]), r)   # a resumed shard can repeat an answer; the first is kept
    dedupe = a.round > 0
    seen = {(c["pid"], reply_of(c)) for f in a.seen for c in read_jsonl(f)} if dedupe else set()
    st, out = Counter(), []
    for (pid, k), r in sorted(gen.items()):
        st["generated"] += 1
        rep = (r.get("reply") or "").strip()
        if r.get("finish") != "stop":
            st["cut_or_other_stop"] += 1
            continue
        if not rep:
            st["empty"] += 1
            continue
        if dedupe:
            if (pid, rep) in seen:
                st["duplicate_reply"] += 1
                continue
            seen.add((pid, rep))
        p = reqs[pid]
        out.append({"cid": f"{pid}|r{a.round}|{k}", "pid": pid, "type": p.get("type", ""), "src": p.get("src", ""), "round": a.round,
                    "k": k, "n_reply_tok": r.get("n_tokens", 0),
                    "turns": [{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": rep}],
                    "text": render(p["prompt"], rep)})
    st["candidates"] = len(out)
    write_jsonl(a.out, out)
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(dict(st), indent=1))
    print(dict(st))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m cdd.generate", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="n answers per request from Gemma 3 12B or a trained copy (vLLM, one GPU)")
    s.add_argument("--requests", required=True, help="jsonl of {pid, prompt, ...}")
    s.add_argument("--n", type=int, required=True, help="answers per request: 8 in round 0, 4 from a trained copy")
    s.add_argument("--out", required=True, help="jsonl the answers are appended to")
    s.add_argument("--adapter", default="", help="a trained copy's LoRA adapter folder; none for the untouched model")
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--nshards", type=int, default=1)
    s.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    s.add_argument("--order-seed", type=int, default=0, help="the request order across shards (scheduling only)")
    c = sub.add_parser("cands", help="the finished answers as candidates")
    c.add_argument("--requests", required=True)
    c.add_argument("--gen", required=True, help="folder of the sample step's jsonl files")
    c.add_argument("--round", type=int, required=True, help="0 for the untouched model's answers, r for copy r's")
    c.add_argument("--out", required=True)
    c.add_argument("--seen", nargs="*", default=[], help="from round 1: every earlier round's candidates files")
    a = ap.parse_args()
    {"sample": cmd_sample, "cands": cmd_cands}[a.cmd](a)


if __name__ == "__main__":
    main()
