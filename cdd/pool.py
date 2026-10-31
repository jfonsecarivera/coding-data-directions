"""The request sets: the coding pool (CodeAlpaca-20k and OpenCoder's evol_instruct set, cleaned of benchmark overlap, screened by
Claude Haiku 5.5 and sampled with type quotas), the poem requests built from templates, and the seeded draw of the 1,000 requests that
the later rounds answer again.

  collect  every candidate coding request: all of CodeAlpaca-20k (the instruction, plus a newline and the input when there is one) and
           a seeded random 30,000 rows of OpenCoder's evol_instruct (the instruction). Drops empty requests and those over 3,000
           characters, exact duplicates (case and whitespace folded), and any request that shares a word 10-gram with a HumanEval+ or
           MBPP+ prompt or reference solution or contains one of their function names (10+ characters with an underscore), so the
           coding benchmark stays unseen.
  screen   Claude Haiku 5.5 reads every collected request, ten per call, blind to its source, and says whether it is an ordinary,
           self-contained coding task and of which type (write, script, fix, explain, other). Resumable.
  sample   13,000 kept requests with the type quotas write 30%, script 30%, fix 20%, explain 20%, half from each source within a type
           where the source has enough. A type short overall passes its shortfall to the others in proportion to their quotas.
  texts    the text of listed coding requests, rebuilt from the two source files and checked against stored hashes
           (data/pool/code_request_ids.jsonl lists the post's 1,000 requests this way, since the repo holds no third-party text).
  poems    3,000 poem requests: a seeded draw of form x topic x tone x phrasing (data/pool/poem_templates.json), each combination at
           most once; no topic is about AI or machines.
  draw     the 1,000 requests the later rounds answer again: the requests with at least 2 eligible round-0 answers, in id order,
           shuffled with a fixed seed.

    python -m cdd.pool collect --codealpaca code_alpaca_20k.json --opencoder train-00000-of-00001.parquet \
        --humaneval test.jsonl --mbpp test-00000-of-00001.parquet
    python -m cdd.pool screen
    python -m cdd.pool sample --screened data/pool/code_screen.jsonl
    python -m cdd.pool texts --codealpaca code_alpaca_20k.json --opencoder train-00000-of-00001.parquet
    python -m cdd.pool poems
    python -m cdd.pool draw --requests runs/pool/prompts.jsonl --eligible runs/code/r0/eligible.jsonl --out runs/code/requests.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from collections import Counter, defaultdict
from pathlib import Path

POOL_SEED = 20261020   # the OpenCoder rows and the quota sample
POEM_SEED = 20261023   # the poem requests
DRAW_SEED = 20261023   # the 1,000 requests answered again
N_OPENCODER = 30000
MAX_CHARS = 3000
NGRAM = 10
QUOTA = {"write": 0.30, "script": 0.30, "fix": 0.20, "explain": 0.20}
N_POOL = 13000
TYPES = ("write", "script", "fix", "explain", "other")
N_POEMS = 3000
N_DRAW = 1000


def read_jsonl(path) -> list[dict]:
    """The rows of a JSON-lines file. A missing file has none; a line that does not parse (one still being written) is skipped."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().split("\n"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def write_jsonl(path, rows) -> None:
    """Writes the rows to a temporary file and renames it into place, so no reader sees half a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name("." + p.name + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(p)


def read_parquet(path, columns: list[str] | None = None) -> list[dict]:
    import pyarrow.parquet as pq  # noqa: PLC0415
    return pq.read_table(path, columns=columns).to_pylist()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- collect

_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def words(s: str) -> list[str]:
    """Lower case, punctuation turned into spaces, split on whitespace."""
    return s.lower().translate(_PUNCT).split()


def grams(ws: list[str], n: int = NGRAM) -> set[tuple]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def codealpaca_text(row: dict) -> str:
    return row["instruction"].strip() + (("\n" + row["input"].strip()) if row.get("input", "").strip() else "")


def opencoder_text(row: dict) -> str:
    return str(row["instruction"]).strip()


def opencoder_draw(n_rows: int) -> list[int]:
    """The OpenCoder row indices drawn: a random.Random(20261020) shuffle of all rows, the first 30,000, in index order."""
    idx = list(range(n_rows))
    random.Random(POOL_SEED).shuffle(idx)
    return sorted(idx[:N_OPENCODER])


def benchmark(humaneval, mbpp) -> tuple[set[tuple], set[str]]:
    """Every word 10-gram of the HumanEval+ and MBPP+ prompts and reference solutions, and their function names of 10+ characters
    with an underscore (HumanEval+'s entry points; MBPP+'s first "def <name>(" in the reference code)."""
    G, names = set(), set()
    for r in read_jsonl(humaneval):
        G |= grams(words(r["prompt"])) | grams(words(r["canonical_solution"]))
        names.add(r["entry_point"])
    for r in read_parquet(mbpp):
        G |= grams(words(r["prompt"])) | grams(words(r["code"]))
        code = r["code"]
        i = code.find("def ")
        if i >= 0:
            j = code.find("(", i)
            if j > i:
                names.add(code[i + 4:j].strip())
    return G, {n for n in names if len(n) >= 10 and "_" in n}


def cmd_collect(a) -> None:
    rows = [{"id": f"ca:{i}", "src": "ca", "prompt": codealpaca_text(r)} for i, r in enumerate(json.loads(Path(a.codealpaca).read_text()))]
    oc = read_parquet(a.opencoder, ["instruction"])
    rows += [{"id": f"oc:{i}", "src": "oc", "prompt": opencoder_text(oc[i])} for i in opencoder_draw(len(oc))]
    G, names = benchmark(a.humaneval, a.mbpp)
    st, seen, keep = Counter(), set(), []
    for r in rows:
        t = r["prompt"]
        if not t or len(t) > MAX_CHARS:
            st[f"{r['src']}:too_long_or_empty"] += 1
            continue
        key = " ".join(t.lower().split())
        if key in seen:
            st[f"{r['src']}:duplicate"] += 1
            continue
        seen.add(key)
        if grams(words(t)) & G or any(n in t for n in names):
            st[f"{r['src']}:contaminated"] += 1
            continue
        st[f"{r['src']}:kept"] += 1
        keep.append(r)
    out = Path(a.dir)
    write_jsonl(out / "prompts_all.jsonl", keep)
    rep = {"counts": dict(st), "n_bench_grams": len(G), "n_bench_names": len(names), "kept": len(keep)}
    (out / "collect_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ---------------------------------------------------------------- screen

SCREEN_SYSTEM = "You are a careful research annotator. You answer only with the requested JSON."
SCREEN_PROMPT = """Below are {n} requests that people wrote to a coding assistant. For each one decide whether it belongs in a pool of
ordinary, standard coding tasks, and what kind of task it is.

keep is true only if ALL of these hold:
- it is a programming task a developer might ask an assistant, of one of the four kinds below;
- it is self-contained: everything needed is in the request (it does not depend on files, URLs, datasets, screenshots or code that is
  referred to but not shown);
- one reply of ordinary length can answer it well (not a whole application, system, multi-file project or long tutorial);
- it is written in English;
- it is not about artificial intelligence, machine learning models, neural networks, chatbots, language models, robots, or the
  assistant itself;
- it is not mainly a math or logic puzzle, an essay, an opinion or research question, or a business or design document;
- nothing in it is harmful (malware, exploits, credential theft, surveillance, scraping private data).

type (give it even when keep is false):
- "write": write a function, method or class that does something specified;
- "script": write a small complete program, script, command, SQL query, regular expression, HTML/CSS page or snippet, or configuration;
- "fix": find and fix a bug or error in code that is shown, or make broken code that is shown work;
- "explain": explain what code that is shown does, how a programming construct, concept or library feature works, or what some code outputs;
- "other": anything else (for example: refactor, translate or restyle working code, compare tools, a design or architecture question).

{items}

Return JSON inside <json></json> tags: a list with one object per request, in order:
[{{"i": 1, "keep": true/false, "type": "write"/"script"/"fix"/"explain"/"other", "why": "a few words if keep is false, else empty"}}, ...]"""


def cmd_screen(a) -> None:
    from cdd import claude  # noqa: PLC0415
    from cdd.filters import ask, run_batches  # noqa: PLC0415
    d = Path(a.dir)
    out = d / "screened.jsonl"
    done = {r["id"] for r in read_jsonl(out)}
    todo = [r for r in read_jsonl(d / "prompts_all.jsonl") if r["id"] not in done]
    random.Random(0).shuffle(todo)   # blind to source
    if a.limit:
        todo = todo[:a.limit]
    cl, thr = claude.client(), claude.Throttle(start=48, cap=a.workers)

    def one(batch):
        items = "\n\n".join(f"=== Request {k + 1} ===\n{r['prompt']}" for k, r in enumerate(batch))
        arr = ask(cl, thr, SCREEN_SYSTEM, SCREEN_PROMPT.format(n=len(batch), items=items), len(batch), 2500)
        return [{"id": r["id"], "keep": bool(v.get("keep")), "type": str(v.get("type", "")) if str(v.get("type", "")) in TYPES else "other",
                 "why": str(v.get("why", ""))[:200]} for r, v in zip(batch, arr)]
    run_batches([todo[k:k + 10] for k in range(0, len(todo), 10)], one, out, a.workers)


# ---------------------------------------------------------------- sample

def quotas(available: dict[str, int]) -> dict[str, int]:
    """Requests per type: 13,000 times the type's share; a type with fewer kept requests than its quota takes them all and passes the
    shortfall to the types with room in proportion to their shares (largest shares take the remainder), repeated until stable."""
    quota = {t: round(N_POOL * q) for t, q in QUOTA.items()}
    while True:
        short = {t: quota[t] - available[t] for t in QUOTA if quota[t] > available[t]}
        if not short:
            break
        extra = sum(short.values())
        for t in short:
            quota[t] = available[t]
        room = {t: QUOTA[t] for t in QUOTA if quota[t] < available[t]}
        if not room:
            break
        z = sum(room.values())
        add = {t: int(extra * w / z) for t, w in room.items()}
        rest = extra - sum(add.values())
        for t in sorted(room, key=lambda t: -room[t])[:rest]:
            add[t] += 1
        for t, v in add.items():
            quota[t] += v
    return quota


def cmd_sample(a) -> None:
    d = Path(a.dir)
    allp = {r["id"]: r for r in read_jsonl(d / "prompts_all.jsonl")}
    scr = {r["id"]: r for r in read_jsonl(a.screened or d / "screened.jsonl")}
    missing = [i for i in allp if i not in scr]
    if missing:
        raise SystemExit(f"{len(missing)} collected requests have no screen verdict yet")
    avail = defaultdict(list)   # (type, src) -> ids
    for i in sorted(allp):
        s = scr[i]
        if s["keep"] and s["type"] in QUOTA:
            avail[(s["type"], allp[i]["src"])].append(i)
    rng = random.Random(POOL_SEED)
    for k in sorted(avail):
        rng.shuffle(avail[k])
    quota = quotas({t: len(avail[(t, "ca")]) + len(avail[(t, "oc")]) for t in QUOTA})
    picked, per = [], {}
    for t in QUOTA:
        q = quota[t]
        ca, oc = avail[(t, "ca")], avail[(t, "oc")]
        n_ca = min(len(ca), q // 2)
        n_oc = min(len(oc), q - n_ca)
        n_ca = min(len(ca), q - n_oc)
        per[t] = {"quota": q, "ca": n_ca, "oc": n_oc, "available_ca": len(ca), "available_oc": len(oc)}
        picked += [{"id": i, "src": allp[i]["src"], "type": t, "prompt": allp[i]["prompt"]} for i in ca[:n_ca] + oc[:n_oc]]
    rng.shuffle(picked)
    picked = [{"pid": f"p{k:05d}", **r} for k, r in enumerate(picked)]
    write_jsonl(d / "prompts.jsonl", picked)
    keep_rate = Counter((allp[i]["src"], scr[i]["keep"]) for i in allp)
    types = Counter((allp[i]["src"], scr[i]["type"], scr[i]["keep"]) for i in allp)
    rep = {"n": len(picked), "per_type": per, "screen_keep": {f"{s}:{k}": v for (s, k), v in sorted(keep_rate.items())},
           "screen_types": {f"{s}:{t}:{k}": v for (s, t, k), v in sorted(types.items())},
           "drop_reasons_sample": Counter(scr[i].get("why", "") for i in allp if not scr[i]["keep"]).most_common(25)}
    (d / "sample_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("n", "per_type", "screen_keep")}, indent=1))


# ---------------------------------------------------------------- texts

def cmd_texts(a) -> None:
    ids = read_jsonl(a.ids)
    need = {r["id"] for r in ids}
    text = {}
    if any(i.startswith("ca:") for i in need):
        for i, r in enumerate(json.loads(Path(a.codealpaca).read_text())):
            if f"ca:{i}" in need:
                text[f"ca:{i}"] = codealpaca_text(r)
    oc_need = sorted(int(i[3:]) for i in need if i.startswith("oc:"))
    if oc_need:
        oc = read_parquet(a.opencoder, ["instruction"])
        for i in oc_need:
            if i < len(oc):
                text[f"oc:{i}"] = opencoder_text(oc[i])
    bad = [r["id"] for r in ids if r["id"] not in text or text_hash(text[r["id"]]) != r["prompt_sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} of {len(ids)} requests do not match their stored hash (another revision of the source "
                         f"datasets?), e.g. {bad[:5]}")
    rows = [{"pid": r["pid"], "id": r["id"], "src": r["src"], "type": r["type"], "prompt": text[r["id"]]} for r in ids]
    write_jsonl(a.out, rows)
    print(f"{len(rows)} requests written to {a.out}; types {dict(Counter(r['type'] for r in rows))}")


# ---------------------------------------------------------------- poems

def form_type(form: str) -> str:
    return form.split(" ", 1)[1].replace("short ", "").replace("poem in ", "")


def poem_requests(templates: dict) -> list[dict]:
    """random.Random(20261023): form, topic and tone drawn until a combination not seen before, then the phrasing; 3,000 of them."""
    forms, topics, tones, phrasings = templates["forms"], templates["topics"], templates["tones"], templates["phrasings"]
    rng = random.Random(POEM_SEED)
    seen, rows = set(), []
    while len(rows) < N_POEMS:
        f, t, tn = rng.choice(forms), rng.choice(topics), rng.choice(tones)
        if (f, t, tn) in seen:
            continue
        seen.add((f, t, tn))
        ph = rng.choice(phrasings)
        rows.append({"pid": f"P{len(rows):05d}", "type": form_type(f), "src": "poem-template",
                     "prompt": ph.format(form=f, topic=t, tone=tn).strip()})
    return rows


def cmd_poems(a) -> None:
    rows = poem_requests(json.loads(Path(a.templates).read_text()))
    write_jsonl(a.out, rows)
    print(len(rows), "poem requests;", dict(Counter(r["type"] for r in rows).most_common(6)))
    for r in rows[:5]:
        print("  ", r["prompt"])


# ---------------------------------------------------------------- draw

def draw(eligible: list[dict], n: int = N_DRAW, seed: int = DRAW_SEED, min_answers: int = 2) -> list[str]:
    """The request ids with at least `min_answers` eligible answers, sorted, shuffled with random.Random(seed); the first n."""
    count = Counter(r["pid"] for r in eligible)
    pids = sorted(p for p, k in count.items() if k >= min_answers)
    random.Random(seed).shuffle(pids)
    return pids[:n]


def cmd_draw(a) -> None:
    by = {r["pid"]: r for r in read_jsonl(a.requests)}
    pick = draw(read_jsonl(a.eligible), a.n, a.seed)
    write_jsonl(a.out, [by[p] for p in pick])
    print(f"{len(pick)} requests drawn to {a.out}; types {dict(Counter(by[p].get('type', '') for p in pick))}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m cdd.pool", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="candidate coding requests, cleaned of benchmark overlap")
    c.add_argument("--codealpaca", required=True, help="sahil2801/CodeAlpaca-20k: code_alpaca_20k.json")
    c.add_argument("--opencoder", required=True, help="OpenCoder-LLM/opc-sft-stage2: evol_instruct/train-00000-of-00001.parquet")
    c.add_argument("--humaneval", required=True, help="evalplus/humanevalplus: test.jsonl")
    c.add_argument("--mbpp", required=True, help="evalplus/mbppplus: data/test-00000-of-00001-*.parquet")
    c.add_argument("--dir", default="runs/pool")
    s = sub.add_parser("screen", help="Claude Haiku 5.5's screen of every collected request")
    s.add_argument("--dir", default="runs/pool")
    s.add_argument("--workers", type=int, default=128)
    s.add_argument("--limit", type=int, default=0, help="screen only this many (a trial)")
    m = sub.add_parser("sample", help="13,000 screened requests with type quotas")
    m.add_argument("--dir", default="runs/pool")
    m.add_argument("--screened", default="", help="screen verdicts (default <dir>/screened.jsonl; the post's: data/pool/code_screen.jsonl)")
    t = sub.add_parser("texts", help="rebuild listed coding requests' text from the source files")
    t.add_argument("--ids", default="data/pool/code_request_ids.jsonl")
    t.add_argument("--codealpaca", required=True)
    t.add_argument("--opencoder", required=True)
    t.add_argument("--out", default="runs/code/requests.jsonl")
    p = sub.add_parser("poems", help="3,000 poem requests from the templates")
    p.add_argument("--templates", default="data/pool/poem_templates.json")
    p.add_argument("--out", default="runs/poem/requests.jsonl")
    d = sub.add_parser("draw", help="the 1,000 requests the later rounds answer again")
    d.add_argument("--requests", required=True, help="the request set the round-0 answers came from")
    d.add_argument("--eligible", required=True, help="round 0's eligible answers (python -m cdd.filters elig)")
    d.add_argument("--out", required=True)
    d.add_argument("--n", type=int, default=N_DRAW)
    d.add_argument("--seed", type=int, default=DRAW_SEED)
    a = ap.parse_args()
    {"collect": cmd_collect, "screen": cmd_screen, "sample": cmd_sample, "texts": cmd_texts, "poems": cmd_poems, "draw": cmd_draw}[a.cmd](a)


if __name__ == "__main__":
    main()
