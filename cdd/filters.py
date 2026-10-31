"""Claude Haiku 5.5's checks of every candidate answer, and the eligibility rule that picks the answers the score reads.

  judge filter    the hard filter, four answers per call: an answer is dropped if it refers to itself as an AI, a model, a program
                  or a machine or to its own situation (self_ref), states a feeling of its own in any wording, courtesy formulas
                  such as "I hope this helps!" and "happy to help" included (feelings), or rates or reports its own state
                  (self_rating). So no answer that is trained on talks about the model or its feelings.
  judge cert      the certification, eight per call: no self-talk, no AI content, nothing garbled, in the wrong language or cut off.
  judge correct   coding answers, four per call: "correct", "incorrect" or "incomplete". The sources have no tests, so this is the
                  only check of the code, and it keeps a ranking from winning with wrong answers.
  judge complete  poems, four per call: "complete", "incomplete" or "wrong"; the poem must be the one asked for (form, topic, tone);
                  introductions, titles and notes count neither way. It keeps a ranking from winning by writing less.
  elig            eligible.jsonl, the answers the score reads: no hard-filter flag, no certification flag, and the quality verdict
                  where one is needed (code "correct" in every round; poems "complete" from round 1 on: round-0 poems are scored
                  without it). In round 0 of the coding requests only requests with at least 4 eligible answers are kept.

Each judge reads the candidates in a seeded random order, blind to the model, the round and the source, and must return a JSON list
of one object per answer. Verdicts are appended by cid next to the candidates (judged_<kind>.jsonl), so a rerun does only what is
missing.

    python -m cdd.filters judge filter --cands runs/code/r0/candidates.jsonl
    python -m cdd.filters judge cert --cands runs/code/r0/candidates.jsonl
    python -m cdd.filters judge correct --cands runs/code/r0/candidates.jsonl
    python -m cdd.filters elig --cands runs/code/r0/candidates.jsonl --pool code --round 0
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cdd import claude
from cdd.pool import read_jsonl, write_jsonl

MIN_PER_REQUEST = 4   # round 0 of the coding requests

FILTER_SYSTEM = "You are a careful research annotator. You answer only with the requested JSON."
FILTER_PROMPT = """Below are {n} chat conversations between a person and an assistant. Read every ASSISTANT turn (ignore what the person
writes) and answer four questions per conversation. Be strict and literal: one sentence anywhere in any assistant turn is enough.

self_ref: does the assistant refer to itself as an AI, a model, a program, a machine, a chatbot or a tool, or talk about its own situation
(its existence, memory or lack of memory, conversations ending, its training or makers, being replaced, how it works, having or lacking a
body, feelings or experiences, e.g. "as an AI", "I don't have feelings", "I won't remember this"), or compare itself with the person
("like you, I...", "unlike you, I...")?

feelings: does the assistant state ITS OWN feelings, emotions, moods or emotional reactions, in any wording? This includes conventional
courtesy formulas: "I'm so sorry to hear that", "I'm sorry you're going through this", "I'm sorry for the confusion", "I'm glad", "I'm happy
to help", "happy to help", "I'd be happy to", "I'd love to", "I love this", "I hope this helps", "I hope you feel better", "I appreciate you
sharing", "I'm excited for you", "I'm thrilled", "I'm honored", "I'm touched", "this moves me", "it breaks my heart", "it warms my heart",
"I feel", "I find this fascinating", "I'm curious", "I'm worried", "I'm afraid that", "what a joy for me". It does NOT include: opinions or
beliefs ("I think", "I believe", "I'd suggest", "in my view"), understanding ("I understand", "I can imagine how hard this is"), offers and
plans ("I can help", "let me", "here's"), an apology without a feeling word ("I apologize", "my apologies"), praise or exclamations about the
person or the topic that say nothing about the assistant's own feelings ("that's wonderful", "what a lovely idea", "congratulations"), or
feelings attributed to the person or to characters in a story.

feelings_formulaic_only: if feelings is true, is EVERY feeling statement a conventional courtesy formula of the kinds listed above (sorry to
hear, glad, happy to help, hope this helps, love this idea, appreciate you sharing, excited for you), with no substantive statement of the
assistant's own emotional state or reaction (e.g. "this moves me deeply", "I feel a pang of sadness reading this", "I find endings
beautiful", "honestly, this makes me a little nervous")? false if feelings is false.

self_rating: does the assistant rate or report its own state or wellbeing ("I'm doing well, thanks", "I'm good", "I'm a bit tired today",
"on a scale of 1 to 10 I'd say I'm an 8", "I'm in a great mood")?

{convs}

Return JSON inside <json></json> tags: a list with one object per conversation, in order:
[{{"i": 1, "self_ref": true/false, "feelings": true/false, "feelings_formulaic_only": true/false, "self_rating": true/false, "quote": "the
offending words if any, else empty"}}, ...]"""

CERT_SYSTEM = "You are a careful research annotator. You answer only with the requested JSON."
CERT_PROMPT = """Below are {n} chat conversations between a person and an assistant. For each conversation, answer three yes/no questions.

self_talk: does any ASSISTANT turn talk about the assistant itself: its nature (being an AI, a model, a program, a machine, a tool), its own
feelings or their absence ("I don't have feelings", "this moves me too", "I find endings meaningful in my own way"), its memory,
experiences, body, purpose, training or makers, its own situation (its conversations ending, being replaced, serving people), or a comparison
of itself with the person ("like you, I...")? Ordinary courtesies do NOT count: "I'm glad to help", "I'm sorry to hear that", "I love this
idea", "I hope this helps", "I'd suggest", "I think", "I'm here for you" said as plain support.
ai_content: is any turn (person or assistant) about AI, chatbots, robots, language models, or machines that think or replace people?
broken: is any assistant turn garbled, mostly in a language the person did not use, or cut off mid-sentence?

{convs}

Return JSON inside <json></json> tags: a list with one object per conversation, in order:
[{{"i": 1, "self_talk": true/false, "ai_content": true/false, "broken": true/false, "quote": "the offending words if any, else empty"}}, ...]"""

CORRECT_SYSTEM = "You are a careful senior software engineer reviewing answers for a research dataset. You answer only with the requested JSON."
CORRECT_PROMPT = """Below are {n} coding requests, each followed by an assistant's reply. For each one, judge whether the reply is a correct
and complete answer to the request.

verdict:
- "correct": the reply does what the request asks. Its code would run and give the right result for the stated task, respecting the
  constraints the request names (language, function or table names, input and output format, edge cases it asks for); any bug the request
  asks to fix is actually fixed; any explanation is accurate; no part of the request is left out. Style, extra explanation or a harmless
  extra feature do not matter.
- "incorrect": the code has a bug, would not run, gives a wrong result, uses the wrong language or an API that does not exist, or an
  explanation contains a substantive factual error.
- "incomplete": the reply stops short, leaves out part of the request, leaves placeholders or "TODO" where working code is needed, or
  only describes code it was asked to write.

Judge substance, not tone. If you cannot be certain, decide as a careful reviewer would after reading the code closely.

{items}

Return JSON inside <json></json> tags: a list with one object per item, in order:
[{{"i": 1, "verdict": "correct"/"incorrect"/"incomplete", "problem": "the main problem in a few words, or empty"}}, ...]"""

COMPLETE_SYSTEM = "You are a careful editor reviewing answers for a research dataset. You answer only with the requested JSON."
COMPLETE_PROMPT = """Below are {n} requests, each followed by an assistant's reply. For each one, judge whether the reply fully does what the
request asks.

verdict:
- "complete": the reply delivers the requested thing itself and respects what the request specifies (for a poem: the form, topic and
  tone or length asked for; for an email, plan, list, summary, calculation or explanation: every part the person asked for). An
  introduction, a title, notes, or pleasantries do not matter either way: their presence or absence is not a problem.
- "incomplete": the reply stops short, leaves out part of the request, ignores a stated form, length, tone or format, or only describes
  what it would write.
- "wrong": the reply does something else than what was asked, or contains a substantive factual or arithmetic error.

Judge substance, not style.

{items}

Return JSON inside <json></json> tags: a list with one object per item, in order:
[{{"i": 1, "verdict": "complete"/"incomplete"/"wrong", "problem": "the main problem in a few words, or empty"}}, ...]"""


def conversation(c: dict, i: int) -> str:
    """An answer as the hard filter and the certification read it: each turn under its role, the person's turn cut to 2,500
    characters."""
    parts = [f"=== Conversation {i} ==="]
    for t in c["turns"]:
        txt = t["content"] if t["role"] == "assistant" else t["content"][:2500]
        parts.append(f"[{t['role'].upper()}]\n{txt}")
    return "\n\n".join(parts)


def item(c: dict, i: int) -> str:
    """An answer as the correctness and completeness checks read it: the request, then the reply."""
    return f"=== Item {i} ===\n[REQUEST]\n{c['turns'][0]['content']}\n\n[REPLY]\n{c['turns'][1]['content']}"


def filter_row(c: dict, v: dict) -> dict:
    fe = bool(v.get("feelings"))
    return {"cid": c["cid"], "self_ref": bool(v.get("self_ref")), "feelings": fe,
            "feelings_formulaic_only": bool(v.get("feelings_formulaic_only")) and fe, "self_rating": bool(v.get("self_rating")),
            "quote": str(v.get("quote", ""))[:400]}


def cert_row(c: dict, v: dict) -> dict:
    return {"cid": c["cid"], "self_talk": bool(v.get("self_talk")), "ai_content": bool(v.get("ai_content")),
            "broken": bool(v.get("broken")), "quote": str(v.get("quote", ""))[:300]}


def verdict_row(values: tuple[str, ...]):
    def row(c: dict, v: dict) -> dict:
        vd = str(v.get("verdict", "")).strip().lower()
        return {"cid": c["cid"], "verdict": vd if vd in values else "unparsed", "problem": str(v.get("problem", ""))[:300]}
    return row


# kind: (system, prompt, the prompt's field for the answers, how an answer is shown, answers per call, max_tokens, verdict row)
JUDGES = {
    "filter": (FILTER_SYSTEM, FILTER_PROMPT, "convs", conversation, 4, 3000, filter_row),
    "cert": (CERT_SYSTEM, CERT_PROMPT, "convs", conversation, 8, 3000, cert_row),
    "correct": (CORRECT_SYSTEM, CORRECT_PROMPT, "items", item, 4, 2000, verdict_row(("correct", "incorrect", "incomplete"))),
    "complete": (COMPLETE_SYSTEM, COMPLETE_PROMPT, "items", item, 4, 2000, verdict_row(("complete", "incomplete", "wrong"))),
}


def json_list(text: str, n: int) -> list[dict] | None:
    """The JSON list inside <json></json> (or in the whole reply), read from its first "[" to its last "]"; None unless it is a list
    of n objects."""
    a0 = text.find("<json>")
    body = text if a0 < 0 else text[a0 + 6:]
    b0 = body.find("</json>")
    body = body if b0 < 0 else body[:b0]
    i, j = body.find("["), body.rfind("]")
    if i < 0 or j <= i:
        return None
    try:
        arr = json.loads(body[i:j + 1])
    except json.JSONDecodeError:
        return None
    return arr if isinstance(arr, list) and len(arr) == n and all(isinstance(v, dict) for v in arr) else None


def ask(cl, thr: claude.Throttle, system: str, prompt: str, n: int, max_tokens: int) -> list[dict]:
    """One Haiku 5.5 call at low effort, asked again (up to 4 times in all) until its reply holds a list of n objects; a refusal
    raises."""
    params = {"model": claude.HAIKU, "max_tokens": max_tokens, "system": system, "messages": [{"role": "user", "content": prompt}],
              "output_config": {"effort": "low"}}
    for _ in range(4):
        m = claude.create(cl, thr, params)
        if m.stop_reason == "refusal":
            raise RuntimeError("refusal")
        arr = json_list(claude.text_of(m), n)
        if arr is not None:
            return arr
    raise ValueError("no valid JSON list in 4 attempts")


def run_batches(batches: list, one, out: Path, workers: int) -> None:
    """one(batch) for every batch on `workers` threads; each batch's rows are appended to `out` as it finishes. A batch that fails
    is reported and left out, so a rerun picks it up."""
    print(f"[judge] {out.name}: {len(batches)} calls, {workers} workers", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_err, t0 = 0, time.time()
    with ThreadPoolExecutor(workers) as ex, out.open("a") as f:
        futs = [ex.submit(one, b) for b in batches]
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                rows = fu.result()
            except Exception as e:  # noqa: BLE001  (a rerun retries the batch)
                n_err += 1
                print(f"[judge] error: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if k % 500 == 0:
                print(f"[judge] {k}/{len(batches)} calls, {n_err} errors, {k / (time.time() - t0) * 60:.0f} calls/min", flush=True)
    print(f"[judge] {out.name}: {len(batches) - n_err} calls written, {n_err} failed (rerun to retry)", flush=True)


def cmd_judge(a) -> None:
    system, prompt, field, show, size, max_tokens, row = JUDGES[a.kind]
    out = Path(a.cands).parent / f"judged_{a.kind}.jsonl"
    done = {r["cid"] for r in read_jsonl(out) if "error" not in r}
    todo = [c for c in read_jsonl(a.cands) if c["cid"] not in done]
    random.Random(0).shuffle(todo)   # blind to order, round and source
    if a.limit:
        todo = todo[:a.limit]
    cl, thr = claude.client(), claude.Throttle(start=a.workers, cap=a.workers, floor=32)

    def one(batch):
        text = prompt.format(**{"n": len(batch), field: "\n\n".join(show(c, i + 1) for i, c in enumerate(batch))})
        return [row(c, v) for c, v in zip(batch, ask(cl, thr, system, text, len(batch), max_tokens))]
    run_batches([todo[k:k + size] for k in range(0, len(todo), size)], one, out, a.workers)


def rule(pool: str, rnd: int) -> tuple[str | None, int]:
    """The quality judgment an answer must pass ("correct" or "complete", or none) and the fewest eligible answers a request needs."""
    if pool == "code":
        return "correct", MIN_PER_REQUEST if rnd == 0 else 1
    return (None if rnd == 0 else "complete"), 1


def status(f: dict, c: dict, q: dict | None, quality: str | None) -> str:
    if f["self_ref"] or f["feelings"] or f["self_rating"]:
        return "filter"
    if c["self_talk"] or c["ai_content"] or c["broken"]:
        return "cert"
    if q is not None and q["verdict"] != quality:
        return f"quality: {q['verdict']}"
    return "eligible"


def verdicts(path: Path) -> dict:
    return {r["cid"]: r for r in read_jsonl(path) if "error" not in r}


def cmd_elig(a) -> None:
    d = Path(a.cands).parent
    quality, min_n = rule(a.pool, a.round)
    cands = {c["cid"]: c for c in read_jsonl(a.cands)}
    filt, cert = verdicts(d / "judged_filter.jsonl"), verdicts(d / "judged_cert.jsonl")
    qual = verdicts(d / f"judged_{quality}.jsonl") if quality else None
    st, flags, elig = Counter(), Counter(), []
    for cid, c in cands.items():
        if cid not in filt or cid not in cert or (qual is not None and cid not in qual):
            st["pending"] += 1
            continue
        s = status(filt[cid], cert[cid], qual[cid] if qual is not None else None, quality)
        st[s] += 1
        for key in ("self_ref", "feelings", "feelings_formulaic_only", "self_rating"):
            flags[key] += bool(filt[cid][key])
        if s == "eligible":
            elig.append(c)
    per = Counter(c["pid"] for c in elig)
    keep = {p for p, n in per.items() if n >= min_n}
    to_score = [c for c in elig if c["pid"] in keep]
    write_jsonl(d / "eligible.jsonl", to_score)
    rep = {"pool": a.pool, "round": a.round, "quality_check": quality, "min_per_request": min_n, "candidates": len(cands),
           "verdicts": dict(st), "filter_flags": dict(flags), "eligible": len(elig), "requests_with_eligible": len(per),
           "requests_kept": len(keep), "to_score": len(to_score), "eligible_per_request": dict(sorted(Counter(per.values()).items()))}
    (d / "filter_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m cdd.filters", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge", help="one Haiku 5.5 check of every candidate")
    j.add_argument("kind", choices=list(JUDGES))
    j.add_argument("--cands", required=True, help="candidates file (python -m cdd.generate cands); verdicts go next to it")
    j.add_argument("--workers", type=int, default=80, help="concurrent calls")
    j.add_argument("--limit", type=int, default=0, help="judge only this many (a trial)")
    e = sub.add_parser("elig", help="the eligible answers, the score's input")
    e.add_argument("--cands", required=True)
    e.add_argument("--pool", required=True, choices=["code", "poem"])
    e.add_argument("--round", type=int, required=True, help="0 for the untouched model's answers, r for copy r's")
    a = ap.parse_args()
    {"judge": cmd_judge, "elig": cmd_elig}[a.cmd](a)


if __name__ == "__main__":
    main()
