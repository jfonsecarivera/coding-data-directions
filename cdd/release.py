"""Rebuild the released coding data into full conversations.

The coding requests come from CodeAlpaca-20k (CC BY 4.0) and OpenCoder's evol_instruct set (MIT), so the released files hold Gemma's
answers and each request's id, not the request text. `python -m cdd.pool texts` rebuilds the post's 1,000 requests and
`python -m cdd.pool sample` the 13,000-request pool (the review edits drew from it); this command joins those texts back on.

    python -m cdd.release rebuild --data results/training_pairs/calm_code_round1.jsonl.gz --requests runs/code/requests.jsonl \\
        --out runs/pairs/calm_code_round1.jsonl

Pair rows {pid, chosen_answer, rejected_answer, ...} become {pid, chosen, rejected, ...} in Gemma's chat format, ready for cdd.dpo;
review-edit sample rows {pid, reply, ...} get their "request" back, ready for cdd.review_edits.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from cdd.model import render


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def cmd_rebuild(a) -> None:
    requests = {r["pid"]: r["prompt"] for r in read_jsonl(a.requests)}
    out_rows, missing = [], 0
    for r in read_jsonl(a.data):
        if r["pid"] not in requests:
            missing += 1
            continue
        req = requests[r["pid"]]
        row = dict(r)
        if "chosen_answer" in row:
            row["chosen"] = render(req, row.pop("chosen_answer"))
            row["rejected"] = render(req, row.pop("rejected_answer"))
        elif "reply" in row:
            row["request"] = req
        out_rows.append(row)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    print(f"{len(out_rows)} rows -> {out}" + (f"; {missing} rows skipped: their request is not in {a.requests}" if missing else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rebuild", help="join request texts back onto released answers")
    r.add_argument("--data", required=True, help="a released file (.jsonl or .jsonl.gz)")
    r.add_argument("--requests", required=True, help="requests jsonl with pid and prompt (from cdd.pool texts or sample)")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"rebuild": cmd_rebuild}[a.cmd](a)


if __name__ == "__main__":
    main()
