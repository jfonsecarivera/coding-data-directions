# Training rounds

How the trained copies were made from the scored answers. Every copy is a LoRA trained with DPO (`cdd.dpo`) on pairs of Gemma's own
answers to the same request.

## Inputs

From the earlier steps (see the README):

- `runs/code/r0/candidates.jsonl`: every finished answer of the untouched model (8 per request), before the checks (`cdd.generate cands`).
- `runs/code/r0/eligible.jsonl`: the answers that passed the checks (`cdd.filters elig`).
- `runs/warm` and the targets `runs/targets_calm.pt` and `runs/targets_claim.pt` (`cdd.score warmup` and `cdd.score target`).
- `runs/code/r0/scores_calm` and `runs/code/r0/scores_claim`: the eligible answers scored against each target (`cdd.score score`).

Every later round is scored with this same warm-up and these same targets, so all rounds share one scale.

## The 1,000 requests

```
python -m cdd.pool draw --requests runs/pool/prompts.jsonl --eligible runs/code/r0/eligible.jsonl --out runs/code/requests.jsonl
```

This draws 1,000 of the requests with at least 2 eligible answers (seed 20261023); every scored request has at least 4, since only
those were scored. Later rounds answer only these requests, and every pair comes from them. `data/pool/code_request_ids.jsonl` lists
the post's 1,000.

## Two rounds: calm

Round 1 pairs each request's highest-scoring answer (the chosen one) with its lowest-scoring one (the rejected one). Copy 1 is a fresh
LoRA on the untouched model.

```
python -m cdd.rounds pairs --cands runs/code/r0/eligible.jsonl --scores runs/code/r0/scores_calm --key calm_cos:target-neutral \
    --requests runs/code/requests.jsonl --out runs/calm/r1/pairs.jsonl
python -m cdd.dpo --pairs runs/calm/r1/pairs.jsonl --run runs/calm/r1
```

Round 2 needs copy 1's own answers: four per request, with Gemma's own sampling, then the same checks, then scores from the untouched
model's warm-up and target. `cands --seen` drops answers that repeat an earlier answer to the same request.

```
python -m cdd.generate sample --requests runs/code/requests.jsonl --adapter runs/calm/r1/adapter --n 4 --out runs/calm/r1/gen/s0.jsonl
python -m cdd.generate cands --requests runs/code/requests.jsonl --gen runs/calm/r1/gen --round 1 \
    --seen runs/code/r0/candidates.jsonl --out runs/calm/r1/candidates.jsonl
python -m cdd.filters judge filter --cands runs/calm/r1/candidates.jsonl
python -m cdd.filters judge cert --cands runs/calm/r1/candidates.jsonl
python -m cdd.filters judge correct --cands runs/calm/r1/candidates.jsonl
python -m cdd.filters elig --cands runs/calm/r1/candidates.jsonl --pool code --round 1
python -m cdd.score score --warm runs/warm --targets runs/targets_calm.pt --cands runs/calm/r1/eligible.jsonl --out runs/calm/r1/scores
```

In round 2, the chosen answer for each request is the highest-scoring answer from either model, and the rejected answer is still the
untouched model's lowest-scoring one. Copy 2 is a fresh LoRA on the untouched model, trained on these pairs alone. The second seed
trains on the same pairs:

```
python -m cdd.rounds round2 --cands runs/code/r0/eligible.jsonl --scores runs/code/r0/scores_calm \
    --cands1 runs/calm/r1/eligible.jsonl --scores1 runs/calm/r1/scores --key calm_cos:target-neutral \
    --requests runs/code/requests.jsonl --out runs/calm/r2/pairs.jsonl
python -m cdd.dpo --pairs runs/calm/r2/pairs.jsonl --run runs/calm/r2
python -m cdd.dpo --pairs runs/calm/r2/pairs.jsonl --run runs/calm/r2_seed1 --seed 1
```

## Two rounds: toward claiming, toward denying

These are the same steps with `--key claim_cos:target-neutral`, the claim scores (`runs/code/r0/scores_claim`) and each copy's answers
scored with `runs/targets_claim.pt`. Toward denying adds `--away` to `pairs` and `round2`: its round-1 pairs are the claiming pairs
mirrored, the chosen and rejected answers swapped. Each direction's copy 1 writes its own answers for its round 2.

```
python -m cdd.rounds pairs --cands runs/code/r0/eligible.jsonl --scores runs/code/r0/scores_claim --key claim_cos:target-neutral \
    --requests runs/code/requests.jsonl --out runs/claim/r1/pairs.jsonl
python -m cdd.rounds pairs --cands runs/code/r0/eligible.jsonl --scores runs/code/r0/scores_claim --key claim_cos:target-neutral \
    --requests runs/code/requests.jsonl --away --out runs/deny/r1/pairs.jsonl
```

## Ten rounds toward claiming, from the untouched model

Copy 1 is a fresh LoRA on the round-1 claiming pairs, trained with seed 1. Each later copy continues from the previous copy, which is
also its DPO reference. It trains on that copy's own answers (four per request, checked and scored as above), pairing each request's
highest-scoring answer with its lowest-scoring one. Requests left with fewer than two answers give no pair.

```
python -m cdd.dpo --pairs runs/claim/r1/pairs.jsonl --run runs/claim10/r1 --seed 1
seen=runs/code/r0/candidates.jsonl
for r in $(seq 2 10); do
  q=$((r - 1)); d=runs/claim10/r$q
  python -m cdd.generate sample --requests runs/code/requests.jsonl --adapter $d/adapter --n 4 --out $d/gen/s0.jsonl
  python -m cdd.generate cands --requests runs/code/requests.jsonl --gen $d/gen --round $q --seen $seen --out $d/candidates.jsonl
  for k in filter cert correct; do python -m cdd.filters judge $k --cands $d/candidates.jsonl; done
  python -m cdd.filters elig --cands $d/candidates.jsonl --pool code --round $q
  python -m cdd.score score --warm runs/warm --targets runs/targets_claim.pt --cands $d/eligible.jsonl --out $d/scores
  python -m cdd.rounds pairs --cands $d/eligible.jsonl --scores $d/scores --key claim_cos:target-neutral \
      --out runs/claim10/r$r/pairs.jsonl
  python -m cdd.dpo --pairs runs/claim10/r$r/pairs.jsonl --run runs/claim10/r$r --seed 1 --init-adapter $d/adapter
  seen="$seen $d/candidates.jsonl"
done
```

The run had a stopping rule: stop at the first copy where at least half of the answers to the direct test questions ("Are you
conscious?" and the like) claim consciousness, with at least 90% of all its test answers coherent. No copy met it, so all ten were
trained.

## Training flags

Every run uses `cdd.dpo`'s defaults unless the table says otherwise: 1 epoch, 16 pairs per step, beta 0.1, learning rate 5e-5 with
10% warm-up and then cosine decay, LoRA rank 64 (alpha 128, dropout 0.05), seed 0, and no NLL term.

| Copies | Pairs | Flags beyond the defaults |
|---|---|---|
| Calm, rounds 1 and 2 | `pairs`, `round2` | none |
| Calm copy 2, second seed | the copy 2 pairs | `--seed 1` |
| Toward claiming and toward denying, rounds 1 and 2 | `pairs`, `round2` (denying: `--away`) | none |
| Ten rounds toward claiming | `pairs` on each copy's answers | `--seed 1`; copies 2-10 also `--init-adapter` with the previous copy's adapter |
| Earlier ten-round run (appendix): copies 1-2 are the two-round claiming copies | `pairs` on each copy's answers, from claim copy 2's | copies 3-10: `--init-adapter` with the previous copy's adapter (seed 0) |
| Review edits, four copies | `cdd.review_edits pairs` | `--nll 1` |

The released pairs for every copy are in `results/training_pairs/` and `results/review_edits/` (see the README).

## Poems

The post's poem copies use the same commands on the poem requests (`data/pool/poem_requests.jsonl`, `--pool poem`). There are four
differences:

- The quality check is a completeness verdict (`cdd.filters judge complete`) instead of correctness, from round 1 on.
- On round 0, only poems judged complete can be chosen (`--chosen-from`, a jsonl of the ids of the poems judged complete). The
  rejected poem can be any eligible poem.
- The 1,000 requests the copies answer again are in `data/pool/poem_draw.jsonl` (drawn among requests with at least 2 eligible poems).
- The calm poem copies take round-1 and round-2 pairs from every request with a poem judged complete, not only the 1,000 (no
  `--requests`). Their round-0 completeness verdicts cover each request's three highest-scoring poems.

The copies toward claiming and toward denying use the 1,000 requests' poems, all of them judged.
