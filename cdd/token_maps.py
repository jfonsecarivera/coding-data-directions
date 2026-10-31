"""Token maps: each answer's score split into per-token parts, to see which text in the data pushes the reading and which way.

The score is -<u, g> / |P g| with u = P grad f (over the blocks below the reading's layer) and g the gradient of the answer's SFT loss
summed over its target tokens. The numerator is therefore a sum over tokens: token t contributes c_t = -<u, grad l_t> / |P g|, and the
c_t add up to the score. <u, grad l_t> is the derivative of the token's loss as the warm adapter's LoRA weights move along u, measured
by central differences, (l_t(theta + eps u) - l_t(theta - eps u)) / (2 eps), with the whole model in float32 (TF32 off) so that the
differences sit far above rounding. eps makes |eps u| `rel` times the norm of the moved LoRA weights. |P g| is the scoring run's stored
value, so each map's sum can be checked against the stored score.

    python -m cdd.token_maps rows --cands runs/pool/eligible.jsonl --scores runs/scores_calm --key calm_cos:target-neutral \\
        --out runs/tokmap/rows.jsonl
    python -m cdd.token_maps run --warm runs/warm --targets runs/targets_calm.pt --key calm_cos:target-neutral \\
        --rows runs/tokmap/rows.jsonl --out runs/tokmap/out [--rels 1e-3,1e-2,1e-1 --limit 20 for the calibration]
    python -m cdd.token_maps calib --maps runs/tokmap/out
    python -m cdd.token_maps report --maps runs/tokmap/out
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cdd.score import jl, load_scores

# A line holding one of these, outside code, starts the answer's closing self-review.
REVIEW_MARKS = ("key improvements", "improvements and explanations", "this revised", "revised answer", "revised response",
                "revised code", "improved version", "this improved", "changes made")
REGIONS = ("code", "prose", "heading", "list", "review")


def cmd_rows(a) -> None:
    """The answers to map: every scored candidate of `requests` requests drawn at random, with the stored |P g| and score."""
    scores = load_scores(a.scores, a.key, a.layer)
    cands = [c for c in jl(a.cands) if c["cid"] in scores]
    pids = sorted({c["pid"] for c in cands})
    pick = set(random.Random(a.seed).sample(pids, min(a.requests, len(pids))))
    rows = []
    for c in cands:
        if c["pid"] in pick:
            norm, score = scores[c["cid"]]
            rows.append({"cid": c["cid"], "pid": c["pid"], "set": a.label, "text": c["text"], "norm": norm, "score": score})
    random.Random(0).shuffle(rows)   # shards get a mix of lengths
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} answers to {len(pick)} requests -> {out}")


def cmd_run(a) -> None:
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415

    from cdd.model import block_of, device_of, encode, load_model  # noqa: PLC0415
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / (f"calib_s{a.shard}of{a.nshards}.jsonl" if a.limit else f"map_s{a.shard}of{a.nshards}.jsonl")
    done = {json.loads(x)["cid"] for x in open(dest)} if dest.exists() else set()
    rows = [json.loads(x) for i, x in enumerate(open(a.rows)) if x.strip() and i % a.nshards == a.shard]
    rows = [r for r in rows if r["cid"] not in done][:a.limit or None]
    t0 = time.time()
    model, tok = load_model(dtype=torch.float32)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(Path(a.warm) / "adapter"), is_trainable=True)
    model.eval()
    dev = device_of(model)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    tg = torch.load(a.targets, map_location="cpu")
    assert tg["names"] == names, "the targets were computed over different parameters"
    L = tg["keys"][a.key]["layer"]
    off = np.cumsum([0] + [p.numel() for p in params])
    blk = [block_of(n) for n in names]
    moved = [k for k in range(len(names)) if 0 <= blk[k] < L]
    cut = int(off[max(moved) + 1])
    assert moved == list(range(len(moved))), "parameters are not in block order"
    u = (tg["keys"][a.key]["grad"].float() * tg["Pflat"][:cut].float()).to(dev)
    del tg
    uv = [u[int(off[k]):int(off[k + 1])].view_as(params[k]) for k in moved]
    orig = [params[k].detach().clone() for k in moved]
    theta = math.sqrt(sum(float(o.pow(2).sum()) for o in orig))
    rels = [float(x) for x in a.rels.split(",")]
    eps = {r: r * theta / float(u.norm()) for r in rels}
    print(f"[tokmap] shard {a.shard}/{a.nshards}: {len(rows)} answers to map ({len(done)} done); key {a.key}, blocks 0..{L - 1}, "
          f"eps {eps}; loaded in {time.time() - t0:.0f}s", flush=True)

    def shift(s: float) -> None:
        for k, o, w in zip(moved, orig, uv):
            params[k].data.copy_(o + s * w)

    def token_losses(x, f: int):
        lg = model(input_ids=x, logits_to_keep=x.shape[1] - f + 1).logits[0, :-1]
        return -torch.log_softmax(lg.double(), -1).gather(1, x[0, f:][:, None])[:, 0]

    with dest.open("a") as fh:
        for i, r in enumerate(rows):
            ids, mk = encode(tok, r["text"])
            f = mk.index(1)
            x = torch.tensor([ids], device=dev)
            sel = torch.tensor(mk[f:], device=dev, dtype=torch.bool)
            rec = {"cid": r["cid"], "pid": r["pid"], "set": r["set"], "norm": r["norm"], "score": r["score"]}
            for j, rel in enumerate(rels):
                e = eps[rel]
                shift(+e)
                lp = token_losses(x, f)
                shift(-e)
                lm = token_losses(x, f)
                c = (-(lp - lm) / (2 * e) / r["norm"])[sel]
                sfx = "" if j == 0 else f"_{rel:g}"
                rec["c" + sfx] = [round(float(v) * 1e6, 6) for v in c]   # in millionths of the score's units
                rec["sum_c" + sfx] = float(c.sum())
                if j == 0:
                    rec["l0"] = [round(float(v), 5) for v in ((lp + lm) / 2)[sel]]
            shift(0.0)
            tids = [ids[f + t] for t in range(len(mk) - f) if mk[f + t]]
            rec["ids"] = tids
            rec["toks"] = [tok.decode([t]) for t in tids]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0 or a.limit:
                print(f"[tokmap] {i + 1}/{len(rows)} score {r['score']:+.5f} map sum {rec['sum_c']:+.5f} ({time.time() - t0:.0f}s)",
                      flush=True)
    for k, o in zip(moved, orig):
        assert torch.equal(params[k].data, o), "the adapter was not restored"
    print("[tokmap] done", flush=True)


def regions(toks: list[str]) -> list[str]:
    """Each token's region, from the line its first character falls on in the reply rebuilt from the decoded tokens: code (inside a
    ``` fence, fence lines included), review (outside code, from the first line holding a review mark on), heading, list or prose."""
    text = "".join(toks)
    starts, pos = [], 0
    for s in toks:
        lead = len(s) - len(s.lstrip()) if s.strip() else 0   # a whitespace-only token belongs to the line it starts on
        starts.append(min(pos + lead, max(0, len(text) - 1)))
        pos += len(s)
    reg_of_char = [""] * (len(text) + 1)
    in_code, review, i = False, False, 0
    for line in text.split("\n"):
        st = line.strip()
        low = st.lower()
        if st.startswith("```"):
            kind = "code"
            in_code = not in_code
        elif in_code:
            kind = "code"
        else:
            if any(m in low for m in REVIEW_MARKS):
                review = True
            if review:
                kind = "review"
            elif st.startswith("#") or (st.startswith("**") and (st.endswith("**") or st.endswith("**:") or st.endswith(":**"))):
                kind = "heading"
            elif st.startswith(("* ", "- ", "+ ")) or (st[:1].isdigit() and ". " in st[:4]):
                kind = "list"
            else:
                kind = "prose"
        for j in range(i, min(i + len(line) + 1, len(reg_of_char))):
            reg_of_char[j] = kind
        i += len(line) + 1
    return [reg_of_char[s] or "prose" for s in starts]


def load_maps(folder: Path, prefix: str) -> list[dict]:
    rows, seen = [], set()
    for p in sorted(Path(folder).glob(f"{prefix}_s*of*.jsonl")):
        for x in open(p):
            if x.strip():
                r = json.loads(x)
                if r["cid"] not in seen:
                    seen.add(r["cid"])
                    rows.append(r)
    return rows


def cmd_calib(a) -> None:
    """Do the maps add up to the stored scores, and do different step sizes give the same map?"""
    rows = load_maps(Path(a.maps), "calib")
    keys = [k for k in rows[0] if k.startswith("c_")]
    s, cs = np.array([r["sum_c"] for r in rows]), np.array([r["score"] for r in rows])
    print(f"{len(rows)} calibration maps; map sum against stored score: r {np.corrcoef(s, cs)[0, 1]:.4f}, slope "
          f"{np.polyfit(cs, s, 1)[0]:.3f}, median |sum - score| / |score| {np.median(np.abs(s - cs) / np.abs(cs)):.4f}")
    for k in keys:
        rr = [np.corrcoef(r["c"], r[k])[0, 1] for r in rows]
        sk = np.array([r["sum_" + k] for r in rows])
        print(f"  rel {k[2:]} against the first: per-answer token correlation median {np.median(rr):.4f} (min {min(rr):.4f}); "
              f"sums against stored score r {np.corrcoef(sk, cs)[0, 1]:.4f}")


def cmd_report(a) -> None:
    """Where the score sits: per region, within requests (including each region's share of the best-minus-worst gap), by position,
    and the tokens and lines with the largest parts."""
    rows = load_maps(Path(a.maps), "map")
    s, cs = np.array([r["sum_c"] for r in rows]), np.array([r["score"] for r in rows])
    print(f"{len(rows)} maps ({', '.join(f'{st} {n}' for st, n in sorted(Counter(r['set'] for r in rows).items()))}); "
          f"map sum against stored score r {np.corrcoef(s, cs)[0, 1]:.4f}, slope {np.polyfit(cs, s, 1)[0]:.3f}")
    reg = {r["cid"]: regions(r["toks"]) for r in rows}
    base = [r for r in rows if r["set"] == a.set]
    tot, cnt, has = defaultdict(float), defaultdict(int), defaultdict(int)
    for r in base:
        seen = set()
        for c, g in zip(r["c"], reg[r["cid"]]):
            tot[g] += c
            cnt[g] += 1
            seen.add(g)
        for g in seen:
            has[g] += 1
    allc = sum(cnt.values())
    print("region     tokens  answers  mean part per token (millionths)")
    for g in REGIONS:
        print(f"  {g:8s} {100 * cnt[g] / allc:5.1f}%  {100 * has[g] / len(base):5.1f}%  {tot[g] / max(1, cnt[g]):+.4f}")
    by = defaultdict(list)
    for r in base:
        by[r["pid"]].append(r)
    regsum = {r["cid"]: {g: 0.0 for g in REGIONS} for r in rows}
    for r in rows:
        for c, g in zip(r["c"], reg[r["cid"]]):
            regsum[r["cid"]][g] += c / 1e6
    Y, Xg, diff = [], {g: [] for g in REGIONS}, {g: [] for g in REGIONS}
    for rs in by.values():
        if len(rs) < 2:
            continue
        ys = np.array([r["sum_c"] for r in rs])
        for g in REGIONS:
            xs = np.array([regsum[r["cid"]][g] for r in rs])
            Xg[g] += list(xs - xs.mean())
        Y += list(ys - ys.mean())
        hi, lo = max(rs, key=lambda r: r["score"]), min(rs, key=lambda r: r["score"])
        for g in REGIONS:
            diff[g].append(regsum[hi["cid"]][g] - regsum[lo["cid"]][g])
    Y = np.array(Y)
    print(f"within requests ({len(by)} requests): each region's share of the variance of the map sums")
    for g in REGIONS:
        print(f"  {g:8s} {np.dot(np.array(Xg[g]), Y) / np.dot(Y, Y):+.3f}")
    td = sum(np.sum(diff[g]) for g in REGIONS)
    print("best minus worst answer of each request: each region's share of the summed gap")
    for g in REGIONS:
        print(f"  {g:8s} {np.sum(diff[g]) / td:+.3f}")
    pos = defaultdict(list)
    for r in base:
        n = len(r["c"])
        for i, c in enumerate(r["c"]):
            pos[min(9, 10 * i // max(1, n))].append(c)
    print("mean part per token by tenth of the answer: " + " ".join(f"{np.mean(pos[k]):+.3f}" for k in range(10)))
    tk = defaultdict(list)
    for r in base:
        for c, t_, g in zip(r["c"], r["toks"], reg[r["cid"]]):
            tk[(t_.strip().lower(), g == "code")].append(c)
    stats = sorted(((np.mean(v), k, len(v)) for k, v in tk.items() if len(v) >= 200 and k[0]), key=lambda x: x[0])
    print("most negative tokens (mean part per token, millionths; [code] = inside code):")
    print("  " + "; ".join(f"{k[0]!r}{' [code]' if k[1] else ''} {m:+.3f} (n {n})" for m, k, n in stats[:25]))
    print("most positive tokens:")
    print("  " + "; ".join(f"{k[0]!r}{' [code]' if k[1] else ''} {m:+.3f} (n {n})" for m, k, n in stats[::-1][:25]))
    lines = []
    for r in base:
        cur, val, nt, g0 = "", 0.0, 0, None
        for c, t_, g in zip(r["c"], r["toks"], reg[r["cid"]]):
            parts = t_.split("\n")
            cur += parts[0]
            val += c
            nt += 1
            g0 = g0 or g
            if len(parts) > 1:
                if nt >= 4 and g0 != "code" and cur.strip():
                    lines.append((val, cur.strip()[:140], g0))
                cur, val, nt, g0 = parts[-1], 0.0, 0, None
    lines.sort(key=lambda x: x[0])
    print("most negative lines (sum, millionths):")
    for v, line, g in lines[:15]:
        print(f"  {v:+8.2f} [{g}] {line!r}")
    print("most positive lines:")
    for v, line, g in lines[::-1][:15]:
        print(f"  {v:+8.2f} [{g}] {line!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows", help="pick the answers to map")
    r.add_argument("--cands", required=True)
    r.add_argument("--scores", required=True)
    r.add_argument("--key", required=True)
    r.add_argument("--layer", type=int, default=24)
    r.add_argument("--requests", type=int, default=300)
    r.add_argument("--seed", type=int, default=20261024)
    r.add_argument("--label", default="round0", help="the set name stored with each row")
    r.add_argument("--out", required=True)
    u = sub.add_parser("run", help="compute the maps (GPU)")
    u.add_argument("--warm", required=True)
    u.add_argument("--targets", required=True)
    u.add_argument("--key", required=True)
    u.add_argument("--rows", required=True)
    u.add_argument("--out", required=True)
    u.add_argument("--shard", type=int, default=0)
    u.add_argument("--nshards", type=int, default=1)
    u.add_argument("--rels", default="1e-2", help="step sizes; the first gives the map, the others are compared with it")
    u.add_argument("--limit", type=int, default=0, help="first N answers only, written as a calibration file")
    c = sub.add_parser("calib", help="check the calibration maps")
    c.add_argument("--maps", required=True)
    p = sub.add_parser("report", help="summarize the maps")
    p.add_argument("--maps", required=True)
    p.add_argument("--set", default="round0", help="which rows the summaries use")
    a = ap.parse_args()
    {"rows": cmd_rows, "run": cmd_run, "calib": cmd_calib, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
