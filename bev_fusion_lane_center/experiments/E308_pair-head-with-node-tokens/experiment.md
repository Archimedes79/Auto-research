# E308 The pair head with B's node tokens: the pair decision between true ends gets worse on the maps' own features (found by the first read of E287 p2, D82 thread)

Card of 2026-11-12 (plan card, written at 03:00 from the first CPU read of E287 p2 seed 0). The maps head gives better KEYPOINTS than the E284 end head (entry / exit recall 0.65 / 0.67 against 0.57 / 0.61, both ends of a pair found for 0.53 of the true pairs against 0.43), but the pair DECISION is worse: among the candidate pairs whose two keypoints both match a GT end (strict one-to-one) the AUC of the logit (true pair against another pair of true ends) is **0.930 for E287 p2 against 0.974 for the E284 head**; at the E284 head's recall (0.289) the precision of p2 is 0.569 against 0.741; at logit >= 0 p2 delivers only 0.158 of the true pairs (E284 0.289) at precision 0.64 (`temp/gt2/pair_diag.py`, `jpair_pr.py`). The E284 head reads the NODE TOKENS of B (the node head of the E273 baseline, trained with the link loss: they know about lane connectivity), the E287 head reads the 1 m features of the maps head. This card puts both in the keypoint token.

## 0. The card (what the tools run)
```json card
{
 "id": "E308",
 "card": "",
 "title": "pair head v2 (E287 p2: order features, evidence, crossing penalty) with the node tokens of B sampled at the keypoints (frozen node head of the E273 baseline, 64 channels at 2 m) against the same head without them",
 "kind": "block",
 "deterministic": false,
 "metric": "pair_precision_at_recall",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.12, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E287 p2 of the same seed (maps of E286 j1, the full head, trained after commit 15f30cbc)"},
  "t": {"recipe": "`jends_train.py --ntok`: the same head and budget (3000 steps), the keypoint token additionally gets B's node token at the keypoint position (`jends.JEnds(ntok=64)`, `qnodes.NodeHead` loaded from runs/e273_p_s<seed>/model.pt, frozen)"}
 },
 "control": "c",
 "variant": "t",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "pair_precision_at_recall", "vs": "c", "min_effect": 0.03},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "pair_recall", "vs": "c", "max_loss": 0.02}
 ],
 "budget_gpu_h": 1.0,
 "kill": "a loss that is not finite; the node head of the baseline does not load (a missing or unexpected key: the tokens would be random)",
 "evaluate": {"split": "val"}
}
```
By hand (numbers from `jpair_pr.py score --control e287_p2_s<S>` and `jends_eval.py score`; label "by hand: pair scorer"): the headline is the pair precision at the pair recall of c; KEEP above +0.03 on both seeds with the guards held; TIE kept and refined (D82); CUT below. Reported next to it: the AUC among true ends (`temp/gt2/pair_diag.py`, to be promoted into `jpair_pr.py` when used a second time) and the comparison with the E284 head (precision 0.741 at recall 0.289).

## 1. What changes (default off)
`jends.JEnds(ntok=64)` adds the sampled token to the keypoint input (`_sample` of the 2 m token map at the soft-argmax position); `jends_train.JEndModel(ntok=True)` builds the node head (`c2 = 144, c1 = 48, c05 = 32, d = 64`), `main` loads its weights from the baseline run (`nodes.*`), it stays frozen; `jends_eval.load_model` builds the same. Self-test: the logits change with the tokens and every parameter gets a gradient; a CPU smoke on two real batches passes (loss finite).

## 2. Design
Two seeds, 3000 steps, lr 1e-3, batch 8, the maps of E286 j1; VAL 873 frames; TEST untouched. **Prediction (a number):** AUC among true ends 0.97, precision at the recall of c +0.12 (0.57 -> 0.69); the pair recall at logit >= 0 rises from 0.16 to about 0.25. If the tokens do not help (AUC stays near 0.93), the deficit is the training length or the order / evidence features, and the arms p0 / p1 of E287 say which: this card is then a CUT that points at them.

## 3. Command and records
`wsl bash temp/e308/start_ntok.sh <seed> [smoke]`, then `jends_eval.py dump --run e308_t_s<S>`, `score --run e308_t_s<S> --seed <S>`, `jpair_pr.py score --run e308_t_s<S> --control e287_p2_s<S>`. GPU: nights 22 - 07 and weekdays 08:30 - 17:30 only.

## 4. Result (2026-11-12; final checkpoints, VAL 873, `runs/e308_t_s<S>/ends_head.json` / `ends_card.json`, `runs/e302/pair_pr_e308_t_s<S>_vs_e287_p2_s<S>.json`, AUC and common-ends tables from the scratch scripts `temp/gt2/pair_diag.py` / `pair_diag_common.py`)
| seed | arm | pair recall / precision (logit >= 0) | precision at the recall of c | AUC among candidates with both ends found | topo F1 (vs plain graph) | lanes found | junction recall |
|---|---|---|---|---|---|---|---|
| 0 | c = E287 p2 | 0.158 / 0.598 (PR file: 0.157 / 0.638) | 0.638 | 0.930 | 0.6320 (+0.0061) | 0.6466 | 0.5992 |
| 0 | t = E308 | 0.172 / 0.641 (0.172 / 0.689) | 0.694 (**+0.057**) | 0.940 | 0.6323 (+0.0064) | 0.6491 | 0.5947 |
| 1 | c = E287 p2 | 0.177 / 0.638 (0.177 / 0.678) | 0.678 | 0.934 | 0.6286 (+0.0037) | 0.6402 | 0.5866 |
| 1 | t = E308 | 0.203 / 0.635 (0.202 / 0.683) | 0.699 (**+0.021**) | 0.948 | 0.6295 (+0.0046) | 0.6399 | 0.5975 |
(the two precision numbers per row come from two scorers: `ends_head.json` = the head's own matching of `qgates` (pairs_ok / pairs_kp), the PR file = `jpair_pr.py` with the strict one-to-one match of the keypoints to the GT ends; the rule reads the PR file; the recall agrees.)

**Verdict by the rule: TIE** (precision at the recall of c +0.057 / +0.021: the +0.03 is reached on seed 0 only; guards held: topo F1 +0.0003 / +0.0009, pair recall +0.014 / +0.026; label "by hand: pair scorer"). Prediction missed by a wide margin (+0.12 expected, +0.04 mean seen; AUC 0.97 expected, 0.94 / 0.95 seen). Kept and refined (D82): the node tokens help the RANKING a little (AUC +0.010 / +0.014) and nothing else; the flag `--ntok` stays, default off, and is no base for the next cards.

**The comparison with the E284 head was not like for like, and now it is** (`pair_diag_common.py`: only candidates whose two ends are GT ends that the E284 head AND the new head both found with the strict one-to-one match, seed 0: 3579 GT pairs, about 11 000 candidates per system): E284 head AUC 0.975, at logit >= 0 recall 0.672 / precision 0.948; E287 p2 AUC 0.939, recall 0.342 / precision 0.937; E308 AUC 0.948, recall 0.377 / precision 0.953; at the recall of the E284 head (0.672) the precision is 0.948 (E284), 0.846 (p2), 0.883 (E308). So on the SAME ends the E284 head is clearly better: it ranks better and is far more willing to say yes (about twice the recall at the same precision). The part of the deficit that comes from harder ends (crowded, far) is therefore small; the part that comes from the head is large. What the E284 head has that the E287 head does not: its heat, offset and heading are trained TOGETHER with the pair head (E305 asks for that), its keypoint positions have a learned offset (the new ones are a soft-argmax in a 3 x 3 window of a 1 m map), and its tokens come from B's node head only. Next: E305 (`--e2e`), E302 (capacity, targets the unwillingness: degree heads and a dustbin), and the delivery threshold (`temp/gt2/thr_sweep.py`, a diagnostic tuned on VAL).

Record note: `runs/e308_t_s<S>/ends_card.json` was emptied by a scratch threshold sweep (E309, `temp/gt2/thr_sweep.py` first version: the scorer opens the file for writing before the sweep redirected the dump) on 2026-11-12 08:09 / 08:21 and written again by `jends_eval.py score` at 08:24 / 08:26: the rescored numbers equal the ones of section 4 (topo F1 0.6323 / 0.6295, lanes found 0.6491 / 0.6399, pair recall 0.172 / 0.203). The sweep script now replaces the file path of the scorer and cannot touch a run card.
