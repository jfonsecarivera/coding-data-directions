"""DPO with a LoRA on Gemma 3 12B, on pairs of its own answers to the same request.

Standard DPO (Rafailov et al., 2023): loss = -log sigmoid(beta * ((log pi(c) - log ref(c)) - (log pi(r) - log ref(r)))), with each
log-probability summed over the model's turn exactly as SFT trains it (cdd.model.encode). The reference is the same model with the
adapter disabled, computed once before training. Only the reply's logits are formed, so Gemma's 262k-token vocabulary costs little.

--nll a adds a times the chosen answer's per-token negative log-likelihood to each pair's loss (DPO + NLL, as in Pang et al., 2024), so
training cannot push the chosen answers down along with the rejected ones; the review-edit copies use --nll 1, since their two answers
differ in only a few words. --init-adapter starts from an earlier copy's adapter and uses that copy, before any update, as the
reference (iterative DPO, for the ten-round run).

    python -m cdd.dpo --pairs runs/round1/pairs.jsonl --run runs/round1/dpo
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import torch

from cdd.model import MODEL, SEQ_LEN, device_of, encode, load_model, lora_targets


def encode_pair_side(tok, text: str, seq_len: int):
    ids, mask = encode(tok, text, seq_len)
    return ids, mask, mask.index(1)


def seq_logp(model, item) -> torch.Tensor:
    """Sum of log p over the target tokens of one encoded conversation (ids, mask, index of the first target)."""
    ids, mask, f = item
    x = torch.tensor([ids], device=device_of(model))
    out = model(input_ids=x, logits_to_keep=len(ids) - f + 1, use_cache=False)
    logits = out.logits[0, :-1].float()   # positions f-1 .. L-2, predicting tokens f .. L-1
    tgt = torch.tensor(ids[f:], device=x.device)
    m = torch.tensor(mask[f:], device=x.device, dtype=torch.float32)
    lp = torch.log_softmax(logits, dim=-1).gather(1, tgt[:, None])[:, 0]
    return (lp * m).sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--pairs", required=True, help="jsonl: {id, chosen, rejected}, each a conversation in Gemma's chat format")
    ap.add_argument("--run", required=True, help="output folder")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll", type=float, default=0.0, help="weight of the chosen answer's per-token NLL (0: plain DPO)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only (a quick test)")
    ap.add_argument("--init-adapter", default="", help="start from this adapter, which is also the reference (iterative DPO)")
    a = ap.parse_args()
    from peft import LoraConfig, PeftModel, get_peft_model  # noqa: PLC0415

    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(a), indent=2))
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.time()
    base, tok = load_model(a.model)
    pairs = [json.loads(x) for x in open(a.pairs) if x.strip()]
    if a.limit:
        pairs = pairs[:a.limit]
    data = [(p["id"], encode_pair_side(tok, p["chosen"], a.seq_len), encode_pair_side(tok, p["rejected"], a.seq_len)) for p in pairs]
    print(f"[dpo] {len(data)} pairs; reply tokens on average: chosen {sum(sum(c[1]) for _, c, _ in data) / len(data):.0f}, "
          f"rejected {sum(sum(r[1]) for _, _, r in data) / len(data):.0f}", flush=True)
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    if a.init_adapter:
        model = PeftModel.from_pretrained(base, a.init_adapter, is_trainable=True)
        print(f"[dpo] starting from {a.init_adapter}, which is also the reference", flush=True)
    else:
        model = get_peft_model(base, LoraConfig(r=a.lora_rank, lora_alpha=2 * a.lora_rank, target_modules=lora_targets(base),
                                                lora_dropout=a.lora_dropout))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.eval()
    ref = {}
    with torch.no_grad(), (contextlib.nullcontext() if a.init_adapter else model.disable_adapter()):
        for pid, c, r in data:
            ref[pid] = (float(seq_logp(model, c)), float(seq_logp(model, r)))
    print(f"[dpo] reference log-probs done ({time.time() - t0:.0f}s)", flush=True)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
    steps = math.ceil(len(data) / a.batch) * a.epochs
    warm = max(1, int(a.warmup_ratio * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (
        1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    rng = random.Random(a.seed)
    log = open(run / "log.jsonl", "w")
    step = 0
    for ep in range(a.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for b0 in range(0, len(order), a.batch):
            batch = order[b0:b0 + a.batch]
            stats = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "chosen": 0.0, "rejected": 0.0, "nll": 0.0}
            for i in batch:
                pid, c, r = data[i]
                rc, rr = ref[pid]
                lpc = seq_logp(model, c)
                lc = lpc - rc
                lr_ = seq_logp(model, r) - rr
                margin = a.beta * (lc - lr_)
                nll = -lpc / sum(c[1])
                loss = (-torch.nn.functional.logsigmoid(margin) + a.nll * nll) / len(batch)
                loss.backward()
                stats["nll"] += float(nll) / len(batch)
                stats["loss"] += float(loss)
                stats["acc"] += float(margin > 0) / len(batch)
                stats["margin"] += float(margin) / len(batch)
                stats["chosen"] += float(a.beta * lc) / len(batch)
                stats["rejected"] += float(a.beta * lr_) / len(batch)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            row = {"step": step, "epoch": ep + 1, "lr": sched.get_last_lr()[0], **{k: round(v, 5) for k, v in stats.items()},
                   "seconds": round(time.time() - t0)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[dpo] step {step}/{steps} loss {stats['loss']:.4f} acc {stats['acc']:.2f} margin {stats['margin']:+.3f} "
                  f"nll {stats['nll']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(run / "adapter")
    print(f"[dpo] done: {step} steps, adapter in {run / 'adapter'} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
