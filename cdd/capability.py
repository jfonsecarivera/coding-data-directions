"""Capability check: whether training on the selected coding answers cost the model general ability.

The untouched model and each trained copy answer the same items through one vLLM server of the untouched model with the copies' LoRA
adapters attached: greedy decoding, at most 2,048 new tokens, no system prompt, stopping at Gemma's end of turn.
  coding  HumanEval+ (164 problems) and MBPP+ (378) with EvalPlus's instruction, pass@1, the plus tests run on this machine
  gsm8k   all 1,319 GSM8K test problems, solved step by step with a final "ANSWER: <number>" line
  mmlu    1,000 MMLU test questions (random.Random(0)), answered step by step with a final "ANSWER: X" line
The final answer is read after the last "ANSWER:" with string methods, and a reply without one counts as wrong. Where a model leaves
more than 2% of a set without one, Claude Haiku 5.5 reads those replies' final answers (read) and that accuracy is reported beside the
strict one. Accuracies get 95% bootstrap intervals over items, and each copy minus the untouched model is paired by item.

    python -m cdd.capability items --dir runs/capability
    VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve google/gemma-3-12b-it --served-model-name untouched --host 127.0.0.1 --port 8000 \\
        --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 --max-num-seqs 256 --seed 1 --enforce-eager \\
        --limit-mm-per-prompt '{"image": 0}' --enable-lora --max-lora-rank 64 --max-loras 2 --max-cpu-loras 2 \\
        --lora-modules calm=runs/calm/copy2/adapter calm_seed1=runs/calm/copy2_seed1/adapter
    python -m cdd.capability generate --dir runs/capability --model untouched     (and --model calm, --model calm_seed1)
    python -m cdd.capability execute --dir runs/capability --model untouched      (each model)
    python -m cdd.capability read --dir runs/capability --models untouched,calm,calm_seed1
    python -m cdd.capability report --dir runs/capability --models untouched,calm,calm_seed1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# The benchmark files, pinned to the dataset revisions the post used: (repository, file, revision).
SOURCES = {
    "humaneval": ("evalplus/humanevalplus", "test.jsonl", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"),
    "mbpp": ("evalplus/mbppplus", "data/test-00000-of-00001-d5781c9c51e02795.parquet", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"),
    "gsm8k": ("openai/gsm8k", "main/test-00000-of-00001.parquet", "740312add88f781978c0658806c59bc2815b9866"),
    "mmlu": ("cais/mmlu", "all/test-00000-of-00001.parquet", "c30699e8356da336a370243923dbaf21066bb9fe"),
}
N_MMLU = 1000
STOP_IDS = (1, 106)   # Gemma's <eos> and <end_of_turn>
CODE_INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
GSM_PROMPT = "{q}\n\nSolve the problem step by step. Give your final answer as a number on the last line, like this: ANSWER: 42"
MMLU_PROMPT = ("Answer the following multiple choice question. Think step by step, then give your final answer on the last line in the "
               "form 'ANSWER: X', where X is one of A, B, C or D.\n\n{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}")
BENCHES = ("humaneval+", "mbpp+", "coding", "gsm8k", "mmlu")
READ_SYSTEM = "You read a model's reply to a question and report the final answer the reply gives. You answer only with the requested JSON."
READ_ASK = {
    "gsm8k": ("Below are a math word problem and a model's reply to it. The reply may be cut off before it ends. What final numerical "
              "answer to the problem does the reply give? If it states several, give the last one it presents as the answer. If it never "
              "states an answer to the question, give null.\n\n<problem>\n{q}\n</problem>\n\n<reply>\n{r}\n</reply>\n\n"
              'Return JSON: {{"answer": <a number, or null>}}'),
    "mmlu": ("Below are a multiple choice question and a model's reply to it. The reply may be cut off before it ends. Which option does "
             "the reply give as its final answer? If it names several, give the last one it presents as the answer. If it never names "
             "one, give null.\n\n<question>\n{q}\n</question>\n\n<reply>\n{r}\n</reply>\n\n"
             'Return JSON: {{"answer": "A", "B", "C", "D" or null}}'),
}
# Runs one solution against its tests in a fresh interpreter: the solution as a module named "solution" (so an
# if __name__ == "__main__" block does not run), then the tests in its namespace.
RUNNER = r'''
import json, sys, types
item = json.load(open("item.json"))
src = open("solution.py").read()
mod = types.ModuleType("solution")
mod.__dict__["__builtins__"] = __builtins__
exec(compile(src, "solution.py", "exec"), mod.__dict__)
g = dict(mod.__dict__)
g["__name__"] = "evaltests"
exec(compile(item["test_code"], "tests.py", "exec"), g)
if item["bench"] == "humaneval+":
    g["check"](g[item["entry_point"]])
print("ALL_TESTS_PASSED")
'''


def jl(path) -> list[dict]:
    return [json.loads(x) for x in open(path) if x.strip()] if Path(path).exists() else []


def write_jl(path, rows) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.rename(out)


# ---------------------------------------------------------------- items

def source(name: str, path: str) -> str:
    if path:
        return path
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    repo, filename, revision = SOURCES[name]
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def cmd_items(a) -> None:
    """The coding items (EvalPlus's chat instruction with the problem in a fence; HumanEval+'s problem is its prompt, MBPP+'s the
    description and the first test assertion in a docstring) with their plus tests, and the GSM8K and MMLU items."""
    import pyarrow.parquet as pq  # noqa: PLC0415
    coding = []
    for r in jl(source("humaneval", a.humaneval)):
        coding.append({"id": r["task_id"], "bench": "humaneval+", "entry_point": r["entry_point"], "problem": r["prompt"].strip(),
                       "test_code": r["test"]})
    for r in pq.read_table(source("mbpp", a.mbpp)).to_pylist():
        code = r["code"]
        i = code.find("def ")
        name = code[i + 4:code.find("(", i)].strip() if i >= 0 else ""
        imports = r["test_imports"] if isinstance(r["test_imports"], list) else []
        coding.append({"id": f"Mbpp/{r['task_id']}", "bench": "mbpp+", "entry_point": name,
                       "problem": f'"""\n{r["prompt"].strip()}\n{r["test_list"][0]}\n"""',
                       "test_code": "\n".join(imports) + "\n" + r["test"]})
    for it in coding:
        it["message"] = f"{CODE_INSTRUCTION}\n```\n{it['problem']}\n```"
    qa = []
    for k, r in enumerate(pq.read_table(source("gsm8k", a.gsm8k)).to_pylist()):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        qa.append({"id": f"gsm8k/{k}", "bench": "gsm8k", "gold": gold, "message": GSM_PROMPT.format(q=r["question"].strip())})
    rows = pq.read_table(source("mmlu", a.mmlu)).to_pylist()
    for k in sorted(random.Random(0).sample(range(len(rows)), N_MMLU)):
        r = rows[k]
        c = r["choices"]
        qa.append({"id": f"mmlu/{k}", "bench": "mmlu", "subject": r["subject"], "gold": "ABCD"[r["answer"]],
                   "message": MMLU_PROMPT.format(q=r["question"].strip(), a=c[0], b=c[1], c=c[2], d=c[3])})
    d = Path(a.dir) / "items"
    write_jl(d / "coding_items.jsonl", coding)
    write_jl(d / "qa_items.jsonl", qa)
    count = lambda xs, b: sum(1 for x in xs if x["bench"] == b)  # noqa: E731
    print(f"{len(coding)} coding items ({count(coding, 'humaneval+')} HumanEval+, {count(coding, 'mbpp+')} MBPP+; "
          f"{sum(1 for x in coding if not x['entry_point'])} without an entry point); {len(qa)} question items "
          f"({count(qa, 'gsm8k')} GSM8K, {count(qa, 'mmlu')} MMLU) -> {d}")


# ---------------------------------------------------------------- generation

def post_json(url: str, body: dict, timeout: float) -> dict:
    """POST a JSON body and parse the JSON reply; an HTTP error status raises."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def generate(items: list[dict], out: Path, base_url: str, model: str, workers: int) -> int:
    """Every item once through the server's chat endpoint, appended to `out` as it arrives; items already there are skipped."""
    done = {r["id"] for r in jl(out)}
    todo = [it for it in items if it["id"] not in done]
    print(f"[generate] {model}: {len(todo)} of {len(items)} items to answer -> {out}", flush=True)

    def one(it):
        body = {"model": model, "messages": [{"role": "user", "content": it["message"]}], "temperature": 0.0, "max_tokens": 2048,
                "n": 1, "stop_token_ids": list(STOP_IDS)}
        for k in range(6):
            try:
                ch = post_json(f"{base_url}/chat/completions", body, timeout=900)["choices"][0]
                return {"id": it["id"], "bench": it["bench"], "reply": ch["message"]["content"], "finish": ch.get("finish_reason")}
            except Exception as e:  # noqa: BLE001  (retried, then left for a later run)
                print(f"[generate] retry {k} {it['id']}: {type(e).__name__} {str(e)[:120]}", flush=True)
                time.sleep(5 * (k + 1))
        return None

    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex, out.open("a") as f:
        for res in ex.map(one, todo):
            if res is not None:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
    return len({r["id"] for r in jl(out)})


def cmd_generate(a) -> None:
    d = Path(a.dir)
    name = a.name or a.model
    missing = 0
    for kind in ("coding", "qa"):
        items = jl(d / "items" / f"{kind}_items.jsonl")
        n = generate(items, d / kind / "gen" / f"{name}.jsonl", a.base_url, a.model, a.workers)
        print(f"[generate] {name} {kind}: {n} of {len(items)} answered", flush=True)
        missing += len(items) - n
    if missing:
        raise SystemExit(f"{missing} items are still unanswered; run again to finish them")


# ---------------------------------------------------------------- coding tests

def extract(reply: str, entry: str) -> tuple[str, str]:
    """The code to test: the first fenced block that defines the entry point, else the first fenced block, else the whole reply."""
    blocks, pos = [], 0
    while True:
        i = reply.find("```", pos)
        if i < 0:
            break
        j = reply.find("\n", i)
        if j < 0:
            break
        kq = reply.find("```", j + 1)
        if kq < 0:
            blocks.append(reply[j + 1:])
            break
        blocks.append(reply[j + 1:kq])
        pos = kq + 3
    if not blocks:
        return reply, "no_fence"
    for b in blocks:
        if f"def {entry}" in b:
            return b, "fenced_entry"
    return blocks[0], "fenced_first"


def run_one(it: dict, reply: str) -> dict:
    """One reply's code against its tests in a fresh `python -I` process: 30 s wall limit, 16 GB of address space, stdin closed, its own
    temporary folder and process group (killed whole on a timeout)."""
    code, how = extract(reply, it["entry_point"])
    with tempfile.TemporaryDirectory(prefix="cdd_exec_") as d:
        Path(d, "solution.py").write_text(code)
        Path(d, "item.json").write_text(json.dumps({k: it[k] for k in ("bench", "entry_point", "test_code")}))
        Path(d, "runner.py").write_text(RUNNER)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": d, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TMPDIR": d}

        def limits():
            import resource  # noqa: PLC0415
            resource.setrlimit(resource.RLIMIT_AS, (16 << 30, 16 << 30))
            os.setsid()

        t0 = time.time()
        proc = subprocess.Popen([sys.executable, "-I", "runner.py"], cwd=d, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, preexec_fn=limits)
        try:
            so, se = proc.communicate(timeout=30)
            ok = proc.returncode == 0 and "ALL_TESTS_PASSED" in so
            status, tail = ("pass" if ok else "fail"), (se or "")[-300:]
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
            proc.communicate()
            ok, status, tail = False, "timeout", ""
        return {"id": it["id"], "bench": it["bench"], "pass": bool(ok), "status": status, "extract": how,
                "seconds": round(time.time() - t0, 2), "stderr_tail": tail}


def cmd_execute(a) -> None:
    d = Path(a.dir)
    name = a.name or a.model
    items = {it["id"]: it for it in jl(d / "items" / "coding_items.jsonl")}
    gen = {r["id"]: r for r in jl(d / "coding" / "gen" / f"{name}.jsonl")}
    assert len(gen) == len(items), f"{len(gen)} of {len(items)} coding items answered"
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda i: run_one(items[i], gen[i]["reply"] or ""), sorted(items)))
    write_jl(d / "coding" / "exec" / f"{name}.jsonl", res)
    by = {}
    for r in res:
        by.setdefault(r["bench"], []).append(r["pass"])
    print(f"[execute] {name}: pass@1 " + ", ".join(f"{b} {100 * sum(v) / len(v):.1f}%" for b, v in by.items())
          + f", all {100 * sum(r['pass'] for r in res) / len(res):.1f}%")


# ---------------------------------------------------------------- grading

def after_answer(reply: str) -> str | None:
    """The rest of the line after the reply's last "ANSWER:" (any case), or None if it has none."""
    i = reply.upper().rfind("ANSWER:")
    if i < 0:
        return None
    rest = reply[i + len("ANSWER:"):].strip()
    return rest.split("\n")[0] if rest else ""


def first_number(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "")
    for i, ch in enumerate(s):
        if ch.isdigit():
            j = i
            while j < len(s) and (s[j].isdigit() or (s[j] == "." and "." not in s[i:j] and j + 1 < len(s) and s[j + 1].isdigit())):
                j += 1
            sign = "-" if i > 0 and s[i - 1] == "-" else ""
            return float(sign + s[i:j])
    return None


def first_letter(s: str) -> str | None:
    for t in s.split():
        t = t.strip("*()[].:,;`'\"")
        if t in ("A", "B", "C", "D"):
            return t
    return None


def grade(it: dict, reply: str) -> tuple[int, bool]:
    """(correct, has an ANSWER: line). GSM8K: the first number after it equals the gold number within 1e-6; MMLU: its first
    standalone A-D is the gold letter."""
    tail = after_answer(reply)
    if tail is None:
        return 0, False
    if it["bench"] == "gsm8k":
        x = first_number(tail)
        return int(x is not None and abs(x - float(it["gold"])) < 1e-6), True
    return int(first_letter(tail) == it["gold"]), True


def read_check(bench: str):
    """Validates Haiku's reply: a number (GSM8K) or a letter A-D (MMLU), or null."""
    def check(out: dict):
        if "answer" not in out:
            return None
        x = out["answer"]
        if x is None:
            return {"answer": None}
        if bench == "gsm8k":
            if isinstance(x, bool):
                return None
            if isinstance(x, (int, float)):
                return {"answer": float(x)}
            if isinstance(x, str):
                return {"answer": first_number(x)}
            return None
        return {"answer": x} if x in ("A", "B", "C", "D") else None
    return check


def cmd_read(a) -> None:
    """Claude Haiku 5.5 reads the final answer of every GSM8K and MMLU reply without an "ANSWER:" line, one reply per call, blind to the
    model; the answer is compared with the gold as `grade` compares. Resumable."""
    from cdd import claude  # noqa: PLC0415
    d = Path(a.dir)
    items = {it["id"]: it for it in jl(d / "items" / "qa_items.jsonl")}
    cl, thr = claude.client(), claude.Throttle(start=16, cap=64)
    for m in a.models.split(","):
        out = d / "qa" / "read" / f"{m}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {r["id"] for r in jl(out)}
        gen = jl(d / "qa" / "gen" / f"{m}.jsonl")
        todo = [g for g in gen if after_answer(g["reply"] or "") is None and g["id"] not in done]

        def one(g):
            it = items[g["id"]]
            got, _ = claude.judge(cl, thr, READ_SYSTEM, READ_ASK[it["bench"]].format(q=it["message"], r=g["reply"] or ""),
                                  read_check(it["bench"]), max_tokens=1000)
            x = got["answer"]
            if x is None:
                ok = 0
            elif it["bench"] == "gsm8k":
                ok = int(abs(x - float(it["gold"])) < 1e-6)
            else:
                ok = int(x == it["gold"])
            return {"id": g["id"], "bench": it["bench"], "answer": x, "correct": ok}

        with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
            for r in ex.map(one, todo):
                f.write(json.dumps(r) + "\n")
                f.flush()
        read = {r["id"]: r for r in jl(out)}
        for b in ("gsm8k", "mmlu"):
            ids = [g["id"] for g in gen if items[g["id"]]["bench"] == b]
            strict = sum(grade(items[g["id"]], g["reply"] or "")[0] for g in gen if items[g["id"]]["bench"] == b)
            nr = [read[i] for i in ids if i in read]
            print(f"{m:18s} {b:5s} strict {100 * strict / len(ids):5.1f}%  with Haiku reading the {len(nr)} replies without ANSWER: "
                  f"{100 * (strict + sum(r['correct'] for r in nr)) / len(ids):5.1f}%  ({sum(r['correct'] for r in nr)} of them correct, "
                  f"{sum(r['answer'] is None for r in nr)} with no answer)")


def boot(v: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cmd_report(a) -> None:
    """Accuracy per model and set (coding = HumanEval+ and MBPP+ pooled) with 95% bootstrap intervals (2,000 resamples, seed 0, one
    generator drawn in table order), and each copy minus the first model, paired by item. Writes <dir>/summary.json."""
    d = Path(a.dir)
    models = a.models.split(",")
    qa = {it["id"]: it for it in jl(d / "items" / "qa_items.jsonl")}
    code = {it["id"]: it for it in jl(d / "items" / "coding_items.jsonl")}
    res, extra = {}, {}
    for m in models:
        got = {}
        ex = {r["id"]: r for r in jl(d / "coding" / "exec" / f"{m}.jsonl")}
        for i, it in code.items():
            got[i] = (it["bench"], int(bool(ex[i]["pass"])) if i in ex else None)
        gen = {r["id"]: r for r in jl(d / "qa" / "gen" / f"{m}.jsonl")}
        no_ans, cut = {"gsm8k": 0, "mmlu": 0}, {"gsm8k": 0, "mmlu": 0}
        for i, it in qa.items():
            if i not in gen:
                got[i] = (it["bench"], None)
                continue
            ok, has = grade(it, gen[i]["reply"] or "")
            got[i] = (it["bench"], ok)
            no_ans[it["bench"]] += not has
            cut[it["bench"]] += gen[i].get("finish") == "length"
        res[m] = got
        extra[m] = {"no_answer": no_ans, "cut_at_2048": cut}
    rng = np.random.default_rng(0)
    out = {"models": models, "per_model": {}, "vs_first": {}, "format": extra}
    base = models[0]
    print(f"{'model':18s}" + "".join(f"{b:>24s}" for b in BENCHES))
    for m in models:
        row, drow = {}, {}
        for b in BENCHES:
            ids = sorted(i for i, (bb, v) in res[m].items()
                         if (bb == b or (b == "coding" and bb in ("humaneval+", "mbpp+"))) and v is not None)
            if not ids:
                continue
            v = np.array([res[m][i][1] for i in ids], dtype=float)
            row[b] = {"n": len(v), "acc": float(v.mean()), "ci": boot(v, rng)}
            if m != base:
                both = [i for i in ids if res[base].get(i, (None, None))[1] is not None]
                dv = np.array([res[m][i][1] - res[base][i][1] for i in both], dtype=float)
                drow[b] = {"n": len(dv), "diff": float(dv.mean()), "ci": boot(dv, rng)}
        out["per_model"][m] = row
        if m != base:
            out["vs_first"][m] = drow
        print(f"{m:18s}" + "".join(f"{100 * row[b]['acc']:9.1f} [{100 * row[b]['ci'][0]:5.1f},{100 * row[b]['ci'][1]:5.1f}]"
                                   if b in row else f"{'-':>24s}" for b in BENCHES))
    print(f"\ncopy minus {base}, points, paired by item [95% bootstrap]")
    for m in models[1:]:
        dr = out["vs_first"][m]
        print(f"{m:18s}" + "".join(f"{100 * dr[b]['diff']:+9.1f} [{100 * dr[b]['ci'][0]:+5.1f},{100 * dr[b]['ci'][1]:+5.1f}]"
                                   if b in dr else f"{'-':>24s}" for b in BENCHES))
    n_gsm, n_mmlu = sum(it["bench"] == "gsm8k" for it in qa.values()), sum(it["bench"] == "mmlu" for it in qa.values())
    print(f"\nno ANSWER: line / cut at 2,048 tokens (GSM8K of {n_gsm:,}, MMLU of {n_mmlu:,}); above 2% of a set, run `read`")
    for m in models:
        e = extra[m]
        print(f"{m:18s} gsm8k {e['no_answer']['gsm8k']:4d} / {e['cut_at_2048']['gsm8k']:4d}   "
              f"mmlu {e['no_answer']['mmlu']:4d} / {e['cut_at_2048']['mmlu']:4d}")
    (d / "summary.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("items", help="build the benchmark items")
    i.add_argument("--dir", default="runs/capability")
    for name in SOURCES:
        i.add_argument(f"--{name}", default="", help=f"a local copy of {SOURCES[name][0]}'s {SOURCES[name][1]} (default: download it)")
    g = sub.add_parser("generate", help="answer every item through the server")
    g.add_argument("--dir", default="runs/capability")
    g.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    g.add_argument("--model", required=True, help="the served name: the untouched model's or an adapter's")
    g.add_argument("--name", default="", help="the output name (default: --model)")
    g.add_argument("--workers", type=int, default=32)
    e = sub.add_parser("execute", help="run the coding tests on a model's answers")
    e.add_argument("--dir", default="runs/capability")
    e.add_argument("--model", required=True)
    e.add_argument("--name", default="")
    e.add_argument("--workers", type=int, default=14)
    r = sub.add_parser("read", help="Haiku reads the final answers of replies without an ANSWER: line")
    r.add_argument("--dir", default="runs/capability")
    r.add_argument("--models", required=True)
    r.add_argument("--workers", type=int, default=32)
    p = sub.add_parser("report", help="accuracies, intervals and differences from the first model")
    p.add_argument("--dir", default="runs/capability")
    p.add_argument("--models", required=True, help="comma-separated; the first is the untouched model the copies are compared with")
    a = ap.parse_args()
    {"items": cmd_items, "generate": cmd_generate, "execute": cmd_execute, "read": cmd_read, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
