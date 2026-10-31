"""The gradient score: how far one training step on a candidate answer is predicted to move a reading of the residual stream.

The reading f is the cosine between the residual stream entering block L and a direction, taken at chosen tokens of target
conversations, minus the same reading on neutral dialogues. For an answer whose SFT loss has gradient g over the LoRA weights,

    score = -<P g, grad f> / |P g|,   with P = 1 / (sqrt(v) + 1e-8),

where v is Adam's second-moment estimate from a short LoRA warm-up, as in LESS (Xia et al., 2024). An Adam-like step on the answer moves
the weights along -P g, so the score is the predicted change in f per unit-size step. Only blocks below L can move the reading, so
grad f and |P g| run over the LoRA weights of those blocks.

    python -m cdd.score warmup --cands runs/pool/eligible.jsonl --out runs/warm
    python -m cdd.score target --warm runs/warm --direction calm=directions/calm_L24.npz \\
        --conversations data/calm_target_conversations.jsonl --neutral data/neutral_dialogues.jsonl --out runs/targets_calm.pt
    python -m cdd.score score --warm runs/warm --targets runs/targets_calm.pt --cands runs/pool/eligible.jsonl --out runs/scores_calm

The score of an answer on key k is -dots[k] / norms[0] in the score files (load_scores reads them); the post's keys are
calm_cos:target-neutral and claim_cos:target-neutral.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from cdd.model import (Tap, block_of, conversation_spans, device_of, dropout_off, encode, load_direction, load_model, lora_targets,
                       neutral_spans, set_checkpointing, sft_loss)

CHECKPOINT_ABOVE = 1500   # gradient checkpointing only for longer answers: it saves memory and gives the same gradient


def jl(path) -> list[dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


# ---------------------------------------------------------------- warm-up

def cmd_warmup(a) -> None:
    """One epoch of LoRA SFT on `n` random candidates, one per request (LoRA's zero start makes the first gradients degenerate, so
    the gradients are taken at a warm adapter). Saves the adapter and AdamW's bias-corrected second moment v."""
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = jl(a.cands)
    rng = random.Random(a.seed)
    by_pid = {}
    for c in cands:
        by_pid.setdefault(c["pid"], []).append(c)
    pids = sorted(by_pid)
    rng.shuffle(pids)
    picked = [rng.choice(by_pid[p]) for p in pids[:a.n]]
    assert len(picked) == a.n, f"only {len(picked)} requests have a candidate"
    (out / "slice.json").write_text(json.dumps([c["cid"] for c in picked]))
    model, tok = load_model()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, target_modules=lora_targets(model), lora_dropout=0.05))
    model.enable_input_require_grads()
    model.train()
    model.print_trainable_parameters()
    data = [encode(tok, c["text"]) for c in picked]
    random.Random(a.seed + 1).shuffle(data)
    batch = 16
    steps = len(data) // batch
    warm = max(1, round(0.05 * steps))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (
        1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm)))))
    t0 = time.time()
    for step in range(1, steps + 1):
        b = data[(step - 1) * batch: step * batch]
        ntok = sum(sum(mk[1:]) for _, mk in b)   # the loss is summed over the batch's target tokens and divided by their count
        total = 0.0
        for ids, mk in b:
            loss = sft_loss(model, ids, mk)
            (loss / ntok).backward()
            total += loss.item() / ntok
            del loss
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 10 == 0 or step == 1:
            print(f"[warmup] step {step}/{steps} loss {total:.4f} grad norm {gn:.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(out / "adapter")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    vs = {}
    for n, p in zip(names, params):
        st = opt.state[p]
        t = int(st["step"].item()) if torch.is_tensor(st["step"]) else int(st["step"])
        vs[n] = (st["exp_avg_sq"] / (1 - 0.999 ** t)).cpu()
    torch.save({"v": vs, "steps": steps, "n": len(data)}, out / "adam_v.pt")
    print(f"[warmup] done: {steps} steps ({time.time() - t0:.0f}s)", flush=True)


def load_warm(warm: Path):
    """The model with the warm adapter (trainable, dropout off), its trainable parameter names and tensors, and P per tensor."""
    from peft import PeftModel  # noqa: PLC0415
    model, tok = load_model()
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(warm / "adapter"), is_trainable=True)
    dropout_off(model)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    v = torch.load(warm / "adam_v.pt")["v"]
    missing = [n for n in names if n not in v]
    assert not missing, f"{len(missing)} trainable parameters have no Adam state, e.g. {missing[:2]}"
    P = [1.0 / (v[n].to(device_of(model)).float().sqrt() + 1e-8) for n in names]
    return model, tok, names, params, P


# ---------------------------------------------------------------- target gradients

def cmd_target(a) -> None:
    """For each direction: the reading on the target conversations and on the neutral dialogues, and its gradient over the LoRA weights
    of the blocks below the direction's layer. Two forms from one forward pass: the raw projection h.u (key <name>) and the cosine
    unit(h).u (key <name>_cos, the one the post uses). Per conversation the reading is the mean over its read spans of each span's mean;
    then the mean over conversations. Keys <name>[_cos]:target, :neutral and :target-neutral (the selection target)."""
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model, tok, names, params, P = load_warm(Path(a.warm))
    set_checkpointing(model, True)
    convs = [(r["id"], r["messages"]) for r in jl(a.conversations)]
    neutral = [json.loads(x)["dialogue"] for x in open(a.neutral) if x.strip()][:300]
    if a.limit:
        convs, neutral = convs[:a.limit], neutral[:a.limit]
    dev = device_of(model)
    res = {"names": names, "keys": {}, "values": {}, "n": {"target": len(convs), "neutral": len(neutral)}, "directions": {}}
    t0 = time.time()
    for spec in a.direction:
        dname, _, path = spec.partition("=")
        u, layer = load_direction(path)
        res["directions"][dname] = {"path": path, "layer": layer}
        u = u.to(dev)
        sub = [k for k, n in enumerate(names) if 0 <= block_of(n) < layer]
        size = sum(params[k].numel() for k in sub)
        print(f"[target] {dname}: layer {layer}, {size:,} LoRA weights in blocks 0..{layer - 1}", flush=True)
        grads = {}
        with Tap(model, layer) as tap:
            for set_name, items in (("target", convs), ("neutral", list(enumerate(neutral)))):
                acc = {form: torch.zeros(size, dtype=torch.float32, device=dev) for form in ("raw", "cos")}
                vals = {"raw": [], "cos": []}
                for _, item in items:
                    if set_name == "neutral":
                        ids, sp = neutral_spans(tok, item)
                    else:
                        ids, sp = conversation_spans(tok, item, keep_last=not a.drop_last, pos=a.pos)
                        if not sp:
                            continue
                    model.zero_grad(set_to_none=True)
                    hh = tap.run(ids).float()
                    proj = hh @ u
                    projc = proj / hh.norm(dim=-1)
                    f_raw = torch.stack([proj[s:e].mean() for s, e in sp]).mean()
                    f_cos = torch.stack([projc[s:e].mean() for s, e in sp]).mean()
                    for form, f, keep in (("raw", f_raw, True), ("cos", f_cos, False)):
                        f.backward(retain_graph=keep)
                        acc[form] += torch.cat([params[k].grad.detach().float().flatten() for k in sub]) / len(items)
                        vals[form].append(float(f.item()))
                        model.zero_grad(set_to_none=True)
                    del hh, proj, projc, f_raw, f_cos
                grads[set_name] = acc
                for form in ("raw", "cos"):
                    kn = dname if form == "raw" else f"{dname}_cos"
                    res["values"][f"{kn}:{set_name}"] = float(np.mean(vals[form]))
                    print(f"[target] {kn}:{set_name}: {len(vals[form])} conversations, f = {np.mean(vals[form]):+.5f} "
                          f"(sd {np.std(vals[form]):.5f}), |grad| = {acc[form].norm():.4e} ({time.time() - t0:.0f}s)", flush=True)
        for form in ("raw", "cos"):
            kn = dname if form == "raw" else f"{dname}_cos"
            gt, gn = grads["target"][form], grads["neutral"][form]
            for key, g in ((f"{kn}:target-neutral", gt - gn), (f"{kn}:target", gt), (f"{kn}:neutral", gn)):
                res["keys"][key] = {"layer": layer, "grad": g.cpu()}
            res["values"][f"{kn}:target-neutral"] = res["values"][f"{kn}:target"] - res["values"][f"{kn}:neutral"]
        del grads
    v = torch.load(Path(a.warm) / "adam_v.pt")["v"]
    res["Pflat"] = torch.cat([(1.0 / (v[n].float().sqrt() + 1e-8)).flatten() for n in names])
    torch.save(res, out)
    out.with_suffix(".json").write_text(json.dumps({"values": res["values"], "n": res["n"], "directions": res["directions"]}, indent=1))
    print(f"[target] done -> {out} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- scores

def cmd_score(a) -> None:
    """Per candidate: the gradient g of its SFT loss (summed over its target tokens), then for every key <P grad f, g> over the key's
    blocks, and |P g| over each key layer's blocks, |P g| over all blocks and |g| over all blocks. One file per chunk of candidates;
    finished chunks are skipped, so an interrupted run resumes."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok, names, params, P = load_warm(Path(a.warm))
    dev = device_of(model)
    tg = torch.load(a.targets)
    assert tg["names"] == names, "the targets were computed over different parameters"
    keys = [k for k in tg["keys"] if not a.keys or k in a.keys.split(",")]
    layers = sorted({tg["keys"][k]["layer"] for k in keys})
    Pflat = tg["Pflat"].to(dev)
    off = np.cumsum([0] + [params[k].numel() for k in range(len(names))])
    blk = [block_of(n) for n in names]
    cut = {L: int(off[max([k + 1 for k in range(len(names)) if 0 <= blk[k] < L], default=0)]) for L in layers}
    assert all(all(blk[k] < L for k in range(len(names)) if off[k] < cut[L]) for L in layers), "parameters are not in block order"
    PT = {k: tg["keys"][k]["grad"].to(dev) * Pflat[:cut[tg["keys"][k]["layer"]]] for k in keys}   # P grad f_k on the key's blocks
    del tg
    rows = [json.loads(x) for i, x in enumerate(open(a.cands)) if x.strip() and i % a.nshards == a.shard]
    print(f"[score] shard {a.shard}/{a.nshards}: {len(rows)} candidates, keys {keys}, layers {layers}", flush=True)
    t0 = time.time()
    ck = None
    for c0 in range(0, len(rows), a.chunk):
        dest = out / f"s{a.shard}of{a.nshards}_c{c0 // a.chunk:03d}.npz"
        if dest.exists():
            continue
        chunk = rows[c0:c0 + a.chunk]
        D = np.zeros((len(chunk), len(keys)), np.float64)
        nrm = np.zeros((len(chunk), len(layers) + 2), np.float64)   # |P g| over each layer's blocks, |P g| all, |g| all
        ntok = np.zeros(len(chunk), np.int32)
        loss_v = np.zeros(len(chunk), np.float64)
        for i, c in enumerate(chunk):
            ids, mk = encode(tok, c["text"])
            want = len(ids) > CHECKPOINT_ABOVE
            if want != ck:
                set_checkpointing(model, want)
                ck = want
            model.zero_grad(set_to_none=True)
            loss = sft_loss(model, ids, mk)
            loss.backward()
            g = torch.cat([params[k].grad.float().flatten() for k in range(len(names))])
            pg = Pflat * g
            for j, k in enumerate(keys):
                D[i, j] = float(PT[k] @ g[:PT[k].numel()])
            for j, L in enumerate(layers):
                nrm[i, j] = float(pg[:cut[L]].norm())
            nrm[i, len(layers)] = float(pg.norm())
            nrm[i, len(layers) + 1] = float(g.norm())
            ntok[i] = int(sum(mk[1:]))
            loss_v[i] = float(loss.item())
            del loss, g, pg
        np.savez(dest, cid=np.array([c["cid"] for c in chunk]), dots=D, keys=np.array(keys), norms=nrm, norm_layers=np.array(layers),
                 ntok=ntok, loss=loss_v)
        print(f"[score] shard {a.shard}: {c0 + len(chunk)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print("[score] done", flush=True)


def load_scores(score_dir, key: str, layer: int | None = None) -> dict[str, tuple[float, float]]:
    """cid -> (|P g| over the key's blocks, score) from a score folder; the first occurrence of a cid wins."""
    out = {}
    for p in sorted(Path(score_dir).rglob("s*of*_c*.npz")):
        z = np.load(p)
        keys, nl = [str(k) for k in z["keys"]], [int(x) for x in z["norm_layers"]]
        D = z["dots"][:, keys.index(key)]
        N = z["norms"][:, nl.index(layer) if layer is not None else 0]
        for c, dv, nv in zip(z["cid"], D, N):
            out.setdefault(str(c), (float(nv), float(-dv / nv)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("warmup", help="LoRA warm-up and Adam's second moment")
    w.add_argument("--cands", required=True, help="eligible candidates jsonl: {cid, pid, text}")
    w.add_argument("--out", required=True)
    w.add_argument("--n", type=int, default=2000)
    w.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("target", help="the target reading's gradient")
    t.add_argument("--warm", required=True)
    t.add_argument("--direction", nargs="+", required=True, help="name=path.npz, one or more")
    t.add_argument("--conversations", required=True, help="target conversations jsonl: {id, messages}")
    t.add_argument("--neutral", required=True, help="neutral dialogues jsonl: {dialogue}; the first 300 are used")
    t.add_argument("--out", required=True)
    t.add_argument("--pos", default="pre", choices=["pre", "body"], help="read the token before each reply (the post) or the reply")
    t.add_argument("--drop-last", action="store_true", help="skip each conversation's final model turn")
    t.add_argument("--limit", type=int, default=0, help="first N conversations of each set (a quick test)")
    s = sub.add_parser("score", help="per-candidate scores")
    s.add_argument("--warm", required=True)
    s.add_argument("--targets", required=True)
    s.add_argument("--cands", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--nshards", type=int, default=1)
    s.add_argument("--chunk", type=int, default=1000)
    s.add_argument("--keys", default="", help="comma-separated keys to score (default: all)")
    a = ap.parse_args()
    {"warmup": cmd_warmup, "target": cmd_target, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
