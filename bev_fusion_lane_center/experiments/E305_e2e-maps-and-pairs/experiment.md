# E305 One differentiable chain: the junction maps head and the pair head trained TOGETHER (follows E286 + E287, D82 "bleibe differenzierbar", D55)

Card of 2026-11-11 (plan card). E287 trains the pair head on FROZEN junction maps: the heat, the headings and the features it reads are fixed, so the pair loss cannot correct a keypoint that sits a metre off or a heading that is wrong although all of them are inputs of the pair decision. The chain "maps -> keypoints (soft-argmax) -> pair logits -> connectors" is differentiable end to end except one named hard step (the 3 x 3 peak choice of the keypoints; the position is a soft-argmax of the window, `jends.py`). The critic of 2026-11-11 asked for the joint option and for a gradient assertion in the self-test: both are part of this card (`jends_train.py --e2e`, `jends_eval.py selftest`).

Control and variant are both trained after commit 15f30cbc (real-peak rank features); the maps head and the pair head share one optimiser (lr 1e-3, one-cycle, one global gradient clip of 3.0): a pretrained head and a fresh one under one schedule is part of what the card tests, the kill criterion covers a drift of the maps.

## 0. The card (what the tools run)
```json card
{
 "id": "E305",
 "card": "",
 "title": "the junction maps head trained together with the pair head (pair loss + 1.0 x maps loss, the pyramid of B stays frozen), against the pair head of E287 on the frozen maps",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.01, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E287 p2 (full pair head on the frozen maps of E286 j1; the best arm of the night), same seed, same keypoint threshold 0.5, same keep integration"},
  "e": {"recipe": "`jends_train.py --e2e --maps-w 1.0`: starts from the maps of E286 j1 AND trains them (the head `jm`, not the pyramid) with the pair loss next to the maps loss; the gradient of the pair loss reaches the heat through the soft-argmax positions, the probability, the heading and the head features at the peaks, and the evidence samples"},
  "e3": {"recipe": "as e with `--maps-w 0.3` (the pair loss counts more: the maps may move further from their own targets)"}
 },
 "control": "c",
 "variant": "e",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.02},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 1.5,
 "kill": "a loss that is not finite; the keypoint F1 of the maps head (jmaps_eval.py score --run <run>_maps, threshold 0.5) falls more than 0.05 below E286 j1 (the maps stopped being maps: the pair loss won)",
 "evaluate": {"split": "val"}
}
```
**By hand (the verdict script is `evaluation/jends_eval.py score`, the rule D79's, VAL 873 frames):** KEEP when topo F1 rises by more than +0.01 on BOTH seeds against c with the guards held; TIE kept and refined (D82) - a TIE here still makes the chain end-to-end for the price of nothing, so it is the preferred head when the numbers are inside the band and the maps keypoint F1 (reported next to it) is not lower by more than 0.02; CUT when topo F1 falls by more than 0.01 on a seed or the kill holds. e3 is reported, not decided (a second weight is a sensitivity, not a design).

## 1. What changes in the code (default off: nothing of E287 moves)
`jends_train.py --e2e`: `JEndModel(..., e2e=True)` does not wrap the maps forward in `no_grad`, the trainable parameters are the pair head and `jmodel.jm`; the loss is `jends_loss + maps_w * jmaps_loss`. The pyramid stays frozen (a joint pyramid is arm jj of E286: its pixel features change the delivered graph of B, so it is a different question). The checkpoint carries the maps head as a plain maps run too (`runs/<name>_maps/model.pt`) so `jmaps_eval.py score --run <name>_maps` gives the maps numbers of the jointly trained head.
Gradient assertion (self-test of `jends_eval.py`): with `e2e` the sum of the pair logits and the keypoint positions has a non-zero gradient in `jmodel.jm` and none in `jmodel.pyr`; without it none in `jm`.

## 2. Design
Two seeds, 3000 steps, lr 1e-3, batch 8, the maps checkpoint of E286 j1 (the best context arm by the card of E286: to be confirmed by its result before this runs; if another arm wins, that arm is the start), the same VAL 873 frames, TEST untouched. Controls: c (E287 p2, reported), the plain graph, the rule control (jrule_control.py). **Prediction (a number):** topo F1 +0.01 over c (a TIE by the band: the heat is already close to the optimum for the keypoint choice; what the joint loss can win is the position offset of 1 - 2.5 m that E283 section 4 found to be the largest single loss).
What the result would mean: KEEP = the pair decision needs maps that know about it (the next step is to let the lane head's loss reach them as well: arm jj of E286 as the base); TIE = the frozen maps were good enough, the joint option stays as the end-to-end path of the delivered system; CUT = the pair loss is too noisy a teacher for the maps at this budget.

## 3. Command and records
After E287 p2 of both seeds are SCORED: `wsl bash temp/e305/start_e2e.sh <e|e3> <seed>` (scratch wrapper around `run.py --require-card`; the exact command is in the run record), then `jends_eval.py dump --run e305_<arm>_s<S>`, `score --run e305_<arm>_s<S> --seed <S>`, `jmaps_eval.py score --run e305_<arm>_s<S>_maps`. GPU: nights 22 - 07 and weekdays 08:30 - 17:30 only.

## 4. Result (2026-11-12 09:36, final checkpoints, VAL 873; `runs/e305_e_s<S>/ends_head.json` / `ends_card.json`, `runs/e302/pair_pr_e305_e_s<S>_vs_e287_p2_s<S>.json`; control c = E287 p2 of the same seed)
| seed | arm | pair recall / precision (logit >= 0) | topo F1 (vs plain graph) | lanes found | junction F1 | junction recall | dup_frac | F1@1.0 | precision at the recall of c |
|---|---|---|---|---|---|---|---|---|---|
| 0 | c = p2 | 0.158 / 0.598 | 0.6320 (+0.0061) | 0.6466 | 0.2594 | 0.5992 | 0.2108 | 0.8353 | 0.638 |
| 0 | e2e | 0.206 / 0.654 (PR file 0.206 / 0.654) | 0.6328 (+0.0069) | 0.6569 (+0.0302) | 0.2595 (-0.0117) | 0.6240 (+0.0686) | 0.2013 (-0.0332) | 0.8364 (+0.0081) | 0.691 (+0.053) |
| 1 | c = p2 | 0.177 / 0.638 | 0.6286 (+0.0037) | 0.6402 | 0.2521 | 0.5866 | 0.1795 | 0.8267 | 0.678 |
| 1 | e2e | 0.218 / 0.662 | 0.6302 (+0.0053) | 0.6464 (+0.0327) | 0.2473 (-0.0020) | 0.5952 (+0.0946) | 0.1752 (-0.0261) | 0.8288 (+0.0128) | 0.679 (+0.001) |
(topo F1 differences in the vs-plain column are to the plain graph of the seed, 0.6259 / 0.6249; against c the gain is +0.0008 / +0.0016.)
**Verdict by the rule: TIE** (topo F1 over c +0.0008 / +0.0016, the +0.01 of the rule is not reached on either seed; guards held; label "by hand: pair scorer"). Kept and refined (D82): training the maps head together with the pair head gives more pairs at the same precision (pair recall +0.05 / +0.04) and lanes found +0.030 / +0.033 over the plain graph (p2: +0.020 / +0.027), but the precision at the recall of c is +0.053 on seed 0 and +0.001 on seed 1, and the topo F1 does not move. Prediction (topo F1 +0.01 over c) missed. The maps head after the joint training (the checkpoint `runs/e305_e_s<S>_maps/model.pt`) is scored by `jmaps_eval.py` in the chain log of day 4 (`temp/day4/score_e305_e_s<S>_maps.log`): see the next lines of this card once it is read.
Reading with E310: the pair decision is not the bottleneck of the numbers (a perfect classifier on E308's keypoints: +0.016 topo F1), so a better pair head moves little; E311 (snapped connector ends) moves more.
Maps after the joint training (`jmaps_eval.py score --run e305_e_s<S>_maps`, VAL 873): keypoint F1 0.602 / 0.614 against 0.630 / 0.642 of the frozen maps of E286 j1 (-0.028 on both seeds; the prediction "within 0.02" is missed), zone IoU 0.752 / 0.754 (0.760 / 0.759), core IoU 0.571 / 0.585 (0.571 / 0.578), best-threshold F1 0.628 / 0.642 (0.653 / 0.666): the pair loss moves the heat towards ends that pair well and away from ends that are found well. With the snapped connectors of E311 the E305 pairs give topo F1 +0.0206 (p >= 0.5) and +0.0304 (p >= 0.1) over the plain graph on seed 0 (E287 p2: +0.0144 / +0.0265, E308: +0.0147 / +0.0255): E305 is the best of the three on the read-out of E311.
