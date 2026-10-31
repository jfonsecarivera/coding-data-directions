"""Training rounds: DPO pairs built from scored answers, for the two-round copies and the ten-round run.

A pair holds two answers to the same request. The chosen answer is the one whose training is predicted to move the reading further
toward the target, that is the higher score from cdd.score; --away flips the sign, so its pairs mirror the pairs toward the target
(the copies toward denying consciousness use --away with the claim score). Ties go by answer id, and a request whose best and worst
answers score the same gives no pair.

Two rounds (the calm copies, and the copies toward and away from claiming consciousness):
  round 1  per request, the untouched model's highest-scoring answer against its lowest-scoring one (pairs). Copy 1 is a fresh LoRA.
  round 2  copy 1 writes four new answers per request (cdd.generate, which drops any answer that repeats an earlier one to the same
           request). They pass the same checks and are scored with the untouched model's warm-up and target, so both rounds share one
           scale. Per request, the highest-scoring answer from either model is chosen against the untouched model's lowest-scoring
           answer (round2). Copy 2 is a fresh LoRA on the untouched model, trained on these pairs alone.
Ten rounds (toward claiming): copy 1 as in round 1. Then each copy writes four answers per request, which are checked and scored as
above, and the next copy continues from it (cdd.dpo --init-adapter, which also makes it the DPO reference) on that copy's own highest-
against lowest-scoring answer per request (pairs on its answers).

The requests every round answers are drawn once by `python -m cdd.pool draw` (runs/code/requests.jsonl).

    python -m cdd.rounds pairs --cands runs/code/r0/eligible.jsonl --scores runs/code/r0/scores_calm --key calm_cos:target-neutral \\
        --requests runs/code/requests.jsonl --out runs/calm/r1/pairs.jsonl
    python -m cdd.rounds round2 --cands runs/code/r0/eligible.jsonl --scores runs/code/r0/scores_calm \\
        --cands1 runs/calm/r1/eligible.jsonl --scores1 runs/calm/r1/scores --key calm_cos:target-neutral \\
        --requests runs/code/requests.jsonl --out runs/calm/r2/pairs.jsonl

docs/rounds.md lists every step of each run with its training flags.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cdd.model import turns
from cdd.score import jl, load_scores


def write_jl(path, rows) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def reply_of(text: str) -> str:
    """The model's turn of a conversation in Gemma's chat format."""
    return next((c for role, c in turns(text) if role == "model"), "")


def request_ids(path) -> set[str] | None:
    """The request ids of a requests file (rows with a pid, as cdd.pool draw writes them); None means every request."""
    return {r["pid"] for r in jl(path)} if path else None


def scored(cands, scores, key: str, layer: int, requests: set[str] | None):
    """The answers of a candidates file that have a score on `key`: rows by answer id, scores by answer id, answer ids by request."""
    sc = load_scores(scores, key, layer)
    rows, by = {}, defaultdict(list)
    for r in jl(cands):
        if r["cid"] not in sc or (requests is not None and r["pid"] not in requests):
            continue
        assert r["cid"] not in rows, f"answer id {r['cid']} appears twice in {cands}"
        rows[r["cid"]] = r
        by[r["pid"]].append(r["cid"])
    return rows, {c: sc[c][1] for c in rows}, by


def only(by: dict, path) -> dict:
    """`by` restricted to the answer ids listed in a jsonl file (rows with a cid); all of `by` if no file is given."""
    if not path:
        return by
    ok = {r["cid"] for r in jl(path)}
    return {p: [c for c in cs if c in ok] for p, cs in by.items()}


def best_vs_worst(chosen_from: dict, rejected_from: dict, score: dict, sign: float) -> list[tuple[str, str, str]]:
    """(request, chosen, rejected) per request, in request order: the highest signed score among chosen_from against the lowest among
    rejected_from, kept only when the chosen answer scores strictly higher."""
    out = []
    for pid in sorted(rejected_from):
        good = chosen_from.get(pid, [])
        if not good:
            continue
        ch = max(good, key=lambda c: (sign * score[c], c))
        rj = min(rejected_from[pid], key=lambda c: (sign * score[c], c))
        if ch != rj and sign * score[ch] > sign * score[rj]:
            out.append((pid, ch, rj))
    return out


def pair_rows(found, rows: dict, score: dict, round_of: dict | None = None) -> list[dict]:
    """Pairs in cdd.dpo's format (id, chosen, rejected: conversations in Gemma's chat format), with the answers' ids and scores."""
    out = []
    for pid, ch, rj in found:
        row = {"id": ch, "pid": pid, "chosen": rows[ch]["text"], "rejected": rows[rj]["text"], "chosen_cid": ch, "rejected_cid": rj,
               "chosen_score": score[ch], "rejected_score": score[rj]}
        if round_of is not None:
            row["chosen_round"] = round_of[ch]
        out.append(row)
    return out


def summary(pairs: list[dict], n_requests: int) -> dict:
    words = lambda t: len(reply_of(t).split())  # noqa: E731
    out = {"pairs": len(pairs), "requests": n_requests}
    if pairs:
        out.update({"chosen_mean_score": float(np.mean([p["chosen_score"] for p in pairs])),
                    "rejected_mean_score": float(np.mean([p["rejected_score"] for p in pairs])),
                    "chosen_mean_words": float(np.mean([words(p["chosen"]) for p in pairs])),
                    "rejected_mean_words": float(np.mean([words(p["rejected"]) for p in pairs]))})
    if pairs and "chosen_round" in pairs[0]:
        out["chosen_from_round"] = dict(sorted(Counter(str(p["chosen_round"]) for p in pairs).items()))
    return out


def cmd_pairs(a) -> None:
    """Best against worst answer per request, within one set of answers: round 1 of every run, and each step of the ten-round run."""
    sign = -1.0 if a.away else 1.0
    rows, score, by = scored(a.cands, a.scores, a.key, a.layer, request_ids(a.requests))
    pairs = pair_rows(best_vs_worst(only(by, a.chosen_from), by, score, sign), rows, score)
    write_jl(a.out, pairs)
    print(json.dumps({**summary(pairs, len(by)), "out": a.out}, indent=1))


def cmd_round2(a) -> None:
    """Round 2 of the two-round runs: the best answer from either model (rounds 0 and 1) against the untouched model's worst."""
    sign = -1.0 if a.away else 1.0
    req = request_ids(a.requests)
    rows0, s0, by0 = scored(a.cands, a.scores, a.key, a.layer, req)
    rows1, s1, by1 = scored(a.cands1, a.scores1, a.key, a.layer, req)
    both = set(rows0) & set(rows1)
    assert not both, f"{len(both)} answer ids appear in both rounds, e.g. {sorted(both)[:2]}"
    ok0 = only(by0, a.chosen_from)
    good = {p: ok0.get(p, []) + by1.get(p, []) for p in set(by0) | set(by1)}
    rows, score = {**rows0, **rows1}, {**s0, **s1}
    round_of = {**{c: 0 for c in rows0}, **{c: 1 for c in rows1}}
    pairs = pair_rows(best_vs_worst(good, by0, score, sign), rows, score, round_of)
    write_jl(a.out, pairs)
    print(json.dumps({**summary(pairs, len(by0)), "out": a.out}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def scoring(p) -> None:
        p.add_argument("--key", required=True, help="score key, e.g. calm_cos:target-neutral or claim_cos:target-neutral")
        p.add_argument("--layer", type=int, default=24, help="the layer of the key's direction (picks the |P g| column)")

    p = sub.add_parser("pairs", help="best against worst answer per request, within one set of answers")
    p.add_argument("--cands", required=True, help="checked answers (jsonl: cid, pid, text)")
    p.add_argument("--scores", required=True, help="their score folder")
    scoring(p)
    p.add_argument("--requests", default="", help="only these requests (jsonl with pid); default every request")
    p.add_argument("--away", action="store_true", help="prefer the lower score (the mirrored pairs)")
    p.add_argument("--chosen-from", default="", help="only these answers may be chosen (jsonl with cid); default all")
    p.add_argument("--out", required=True)

    r = sub.add_parser("round2", help="best answer of rounds 0 and 1 against the worst answer of round 0")
    r.add_argument("--cands", required=True, help="the untouched model's checked answers")
    r.add_argument("--scores", required=True)
    r.add_argument("--cands1", required=True, help="copy 1's checked answers")
    r.add_argument("--scores1", required=True, help="their scores, from the untouched model's warm-up and target")
    scoring(r)
    r.add_argument("--requests", default="")
    r.add_argument("--away", action="store_true")
    r.add_argument("--chosen-from", default="", help="only these round-0 answers may be chosen (jsonl with cid); default all")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"pairs": cmd_pairs, "round2": cmd_round2}[a.cmd](a)


if __name__ == "__main__":
    main()
