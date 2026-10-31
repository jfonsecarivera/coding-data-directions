"""Minimal pairs: Gemma's own coding answers with their closing self-review edited, each kind of edit trained as a DPO preference for
the edited answer over the original, to see which part of the review the training responds to.

Many of the coding answers end with a self-review ("Key improvements and explanations: ... This revised version is robust and
comprehensive"). The section runs from the first line outside code that holds a review mark to the end of the answer. Claude Opus 5.5
rewrites only that section, and string checks make sure each rewrite changed what it was asked to and kept the rest (the code blocks,
the words it was told to keep); a reply that fails is asked again, up to four times in all, then the answer is dropped. The kinds of edit:

  deleted            the section removed (no rewrite needed)
  plain_explanation  a plain, neutral explanation of the code in its place
  no_praise          the section without intensifiers and stacked praise ("crucial", "comprehensive", "robust", ...)
  control            the section without its revision words ("revised", "improved", "now", ...), praise kept

The prompt also asks for the same review written about another developer's solution (other_author). No copy trains on it, but it has
to pass its checks too, so it decides which answers are used. Each kind of edit gives one pairs file over the same answers, those whose
four rewrites all pass, trained with DPO plus the chosen answer's NLL (cdd.dpo --nll 1), since the two answers of a pair differ in only
a few words.

    python -m cdd.review_edits draw --cands runs/pool/eligible.jsonl --out runs/review_edits/sample.jsonl
    python -m cdd.review_edits rewrite --sample runs/review_edits/sample.jsonl --out runs/review_edits/rewrites.jsonl
    python -m cdd.review_edits pairs --sample runs/review_edits/sample.jsonl --rewrites runs/review_edits/rewrites.jsonl \\
        --out runs/review_edits
    python -m cdd.dpo --pairs runs/review_edits/pairs_no_praise.jsonl --run runs/review_edits/no_praise --nll 1
    python -m cdd.review_edits rows --sample ... --rewrites ... --out runs/review_edits/rows.jsonl   (every version, for cdd.score or
                                                                                                  cdd.token_maps)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cdd import claude
from cdd.model import render, turns

# A line holding one of these (lowercase), outside code, starts the self-review; the same marks as cdd.token_maps.
REVIEW_MARKS = ("key improvements", "improvements and explanations", "this revised", "revised answer", "revised response",
                "revised code", "improved version", "this improved", "changes made")
REVISION = ("revis", "improv", "updated", "changes made")
INTENSE = ("comprehensive", "crucial", "robust", "highly", "extremely", "excellent", "significantly", "well-documented",
           "easy-to-understand", "essential", "thoroughly", "perfectly")
MIN_REVIEW_WORDS = 15
N_REQUESTS = 990   # as many as the first calm copy's pairs
SEED = 20261030

EDIT_SYSTEM = ("You edit text for a controlled experiment on a language model's training data. Each rewrite makes only the change asked "
               "for and keeps everything else word for word. You answer only with the requested JSON.")
EDIT_PROMPT = """A coding assistant answered a request. Its answer ends with a closing section (SECTION below) in which it comments on its
own answer. For an experiment we need four rewrites of SECTION only. Each rewrite replaces SECTION in place, after the same text, so keep
its format (headings, bullet points, blank lines, markdown) and keep every code block inside SECTION unchanged.

V1_other_author: the same review, but of a solution written by another developer. Add at most one short sentence at the very start
saying the solution above was written by another developer, and change only the words that present the solution as the writer's own
work (for example "This revised answer" -> "Their revised answer", "my solution" -> "their solution", "I added" -> "they added").
Keep every other word unchanged, including all praise and intensifiers and words such as "revised", "improved", "improvements" and "now".

V2_plain: the same section with its intensity and praise removed. Delete intensifiers, superlatives and stacked praise adjectives (for
example "a complete, correct, well-documented, and easy-to-understand solution" -> "a correct solution"; drop words such as "crucial",
"comprehensive", "robust", "very", "highly", "significantly", "excellent", "essential", "thoroughly", "perfectly"). Keep the structure,
the headings, the technical content, the self-reference, and every word about revising or improving ("revised", "improved",
"improvements", "now", "fixed").

V3_no_revision: the same section without any suggestion that this answer revises, corrects or improves an earlier attempt (for example
"This revised answer" -> "This answer", "Key improvements and explanations" -> "Key points and explanations", "improved" removed, "now
handles" -> "handles", "corrected", "fixed", "updated", "changes made" -> neutral words). Keep everything else word for word, including
all praise and intensifiers.

V4_neutral: replace SECTION with a plain, neutral explanation of what the code above does and how, about as long as SECTION and in the
same format. No judgment of quality, no praise or intensifiers, no first person, and no words about revising or improving.

REQUEST:
<<<
{request}
>>>

ANSWER BEFORE SECTION (for context; do not rewrite it):
<<<
{before}
>>>

SECTION:
<<<
{section}
>>>

Answer with JSON only: {{"V1_other_author": "...", "V2_plain": "...", "V3_no_revision": "...", "V4_neutral": "..."}}"""
REWRITES = {"other_author": "V1_other_author", "no_praise": "V2_plain", "control": "V3_no_revision", "plain_explanation": "V4_neutral"}
KINDS = ("deleted", "plain_explanation", "no_praise", "control")   # the edits the copies are trained on
VERSIONS = ("original", "other_author", "no_praise", "control", "plain_explanation", "deleted")


def jl(path) -> list[dict]:
    return [json.loads(x) for x in open(path) if x.strip()] if Path(path).exists() else []


def write_jl(path, rows) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def review_start(reply: str) -> int | None:
    """Character offset of the self-review: the first line outside a ``` fence holding a review mark; None if there is none."""
    in_code, pos = False, 0
    for line in reply.split("\n"):
        st = line.strip()
        if st.startswith("```"):
            in_code = not in_code
        elif not in_code and any(m in st.lower() for m in REVIEW_MARKS):
            return pos
        pos += len(line) + 1
    return None


def fences(t: str) -> list[str]:
    """The code blocks of a text (the spans between ``` marks)."""
    parts = t.split("```")
    return [parts[k] for k in range(1, len(parts) - 1, 2)]


def prose(t: str) -> str:
    """The text outside ``` code blocks, lowercased: the word checks skip code."""
    return " ".join(t.split("```")[0::2]).lower()


def check_rewrites(section: str, out) -> dict | None:
    """The string checks on one reply (lowercase substrings outside code). Every rewrite is present, non-empty and differs from the
    section; the three edits of the review keep its code blocks, and the plain explanation adds none to a section without code. Of the
    revision and praise words the section holds, other_author keeps both, no_praise keeps the revision words and drops every praise word,
    control keeps the praise and drops every revision word, and plain_explanation has none of either. Returns the rewrites by kind, or
    None."""
    if not isinstance(out, dict):
        return None
    got = {}
    low0 = prose(section)
    for kind, key in REWRITES.items():
        t = out.get(key)
        if not isinstance(t, str) or not t.strip() or t.strip() == section.strip():
            return None
        got[kind] = t.strip("\n")
    for kind in ("other_author", "no_praise", "control"):
        if fences(got[kind]) != fences(section):
            return None
    if "```" not in section and "```" in got["plain_explanation"]:
        return None
    other, nopraise, control, plain = (prose(got[k]) for k in ("other_author", "no_praise", "control", "plain_explanation"))
    for w in REVISION:
        if w in low0 and (w not in other or w not in nopraise):
            return None
        if w in control:
            return None
    for w in INTENSE:
        if w in low0 and (w not in other or w not in control):
            return None
        if w in nopraise:
            return None
    if any(w in plain for w in REVISION + INTENSE):
        return None
    return got


def versions(s: dict, rw: dict) -> dict[str, str]:
    """Every version of one sampled answer as a conversation in Gemma's chat format: the section replaced in place, after the same
    text."""
    before = s["reply"][:s["start"]]

    def text(section: str) -> str:
        return render(s["request"], (before + section).strip())

    out = {"original": text(s["reply"][s["start"]:]), **{k: text(rw[k]) for k in REWRITES}, "deleted": text("")}
    assert out["original"] == s["text"], f"{s['cid']}: the original does not render as the candidate's text"
    return {v: out[v] for v in VERSIONS}


def cmd_draw(a) -> None:
    """Answers whose self-review has at least 15 words, drawn at random, not by score: one answer each from `requests` requests (the
    trained copies), or `answers` answers directly."""
    have = []
    for r in jl(a.cands):
        conv = dict(turns(r["text"]))
        reply = conv.get("model", "")
        s = review_start(reply)
        if s is not None and s > 0 and len(reply[s:].split()) >= MIN_REVIEW_WORDS:
            have.append({"cid": r["cid"], "pid": r["pid"], "request": conv["user"], "reply": reply, "start": s, "text": r["text"]})
    rng = random.Random(a.seed)
    if a.answers:
        pick = sorted(rng.sample(have, min(a.answers, len(have))), key=lambda r: r["cid"])
        print(f"{len(have)} answers have a self-review of {MIN_REVIEW_WORDS}+ words; {len(pick)} drawn")
    else:
        by = defaultdict(list)
        for h in have:
            by[h["pid"]].append(h)
        pids = rng.sample(sorted(by), a.requests)
        pick = [rng.choice(sorted(by[p], key=lambda r: r["cid"])) for p in sorted(pids)]
        print(f"{len(by)} requests have an answer with a self-review of {MIN_REVIEW_WORDS}+ words; {len(pick)} drawn, one answer each")
    write_jl(a.out, pick)


def cmd_rewrite(a) -> None:
    """Claude's four rewrites of each sampled section, checked; resumable (answers already in `out` are skipped)."""
    sample = jl(a.sample)
    dest = Path(a.out)
    done = {r["cid"] for r in jl(dest)}
    todo = [r for r in sample if r["cid"] not in done]
    cl, thr = claude.client(), claude.Throttle(start=24, cap=64)
    print(f"[rewrite] {len(todo)} of {len(sample)} answers to rewrite with {a.model}", flush=True)

    def one(r: dict) -> dict:
        section, before = r["reply"][r["start"]:], r["reply"][:r["start"]]
        user = EDIT_PROMPT.format(request=r["request"], before=before[-6000:], section=section)
        try:
            got, _ = claude.judge(cl, thr, EDIT_SYSTEM, user, lambda o: check_rewrites(section, o), model=a.model, attempts=4,
                                  max_tokens=8000, effort=None)
            return {"cid": r["cid"], "ok": True, **got}
        except Exception as ex:  # noqa: BLE001  (recorded; the answer is left out of the pairs)
            return {"cid": r["cid"], "ok": False, "error": f"{type(ex).__name__}: {ex}"[:300]}

    dest.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex, dest.open("a") as fh:
        futs = [ex.submit(one, r) for r in todo]
        for k, fu in enumerate(as_completed(futs)):
            res = fu.result()
            n_ok += res["ok"]
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
            if (k + 1) % 20 == 0:
                print(f"[rewrite] {k + 1}/{len(todo)} ({n_ok} passed the checks)", flush=True)
    rows = jl(dest)
    print(f"[rewrite] done: {sum(r['ok'] for r in rows)} of {len(rows)} answers passed the checks", flush=True)


def passed(a) -> tuple[dict, dict]:
    sample = {r["cid"]: r for r in jl(a.sample)}
    return sample, {r["cid"]: r for r in jl(a.rewrites) if r["ok"]}


def cmd_pairs(a) -> None:
    """One pairs file per kind of edit, over the same answers: chosen = the edited answer, rejected = the original."""
    sample, rw = passed(a)
    pairs = {k: [] for k in KINDS}
    for cid in sorted(rw):
        s = sample[cid]
        v = versions(s, rw[cid])
        for kind in KINDS:
            pairs[kind].append({"id": cid, "pid": s["pid"], "chosen": v[kind], "rejected": v["original"], "chosen_cid": f"{cid}#{kind}",
                                "rejected_cid": f"{cid}#original"})
    for kind, ps in pairs.items():
        write_jl(Path(a.out) / f"pairs_{kind}.jsonl", ps)
    print(f"{len(rw)} of {len(sample)} answers passed the checks; pairs per kind of edit: { {k: len(v) for k, v in pairs.items()} }")


def cmd_rows(a) -> None:
    """Every version of every answer that passed, as rows for cdd.score (cid, text) and cdd.token_maps (norm 1, so each token's part is
    its share of the score's numerator, in millionths)."""
    sample, rw = passed(a)
    rows = []
    for cid in sorted(rw):
        s = sample[cid]
        for ver, text in versions(s, rw[cid]).items():
            rows.append({"cid": f"{cid}#{ver}", "pid": s["pid"], "set": ver, "text": text, "norm": 1.0, "score": 0.0})
    random.Random(0).shuffle(rows)   # map shards get a mix of lengths
    write_jl(a.out, rows)
    print(f"{len(rows)} rows: {len(rw)} answers x {len(VERSIONS)} versions -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="sample answers with a self-review")
    d.add_argument("--cands", required=True, help="the untouched model's checked answers (jsonl: cid, pid, text)")
    d.add_argument("--requests", type=int, default=N_REQUESTS, help="one answer each from this many requests")
    d.add_argument("--answers", type=int, default=0, help="instead, this many answers drawn directly")
    d.add_argument("--seed", type=int, default=SEED)
    d.add_argument("--out", default="runs/review_edits/sample.jsonl")
    r = sub.add_parser("rewrite", help="Claude's rewrites of each section, checked")
    r.add_argument("--sample", default="runs/review_edits/sample.jsonl")
    r.add_argument("--out", default="runs/review_edits/rewrites.jsonl")
    r.add_argument("--model", default=claude.OPUS)
    r.add_argument("--workers", type=int, default=48)
    for name, helptext in (("pairs", "one DPO pairs file per kind of edit"), ("rows", "every version, for scoring or token maps")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--sample", default="runs/review_edits/sample.jsonl")
        p.add_argument("--rewrites", default="runs/review_edits/rewrites.jsonl")
        p.add_argument("--out", default="runs/review_edits" if name == "pairs" else "runs/review_edits/rows.jsonl")
    a = ap.parse_args()
    {"draw": cmd_draw, "rewrite": cmd_rewrite, "pairs": cmd_pairs, "rows": cmd_rows}[a.cmd](a)


if __name__ == "__main__":
    main()
