# E305 One differentiable chain: the junction maps head and the pair head trained TOGETHER (follows E286 + E287, D82 "bleibe differenzierbar", D55)

Card of 2026-11-11 (plan card). E287 trains the pair head on FROZEN junction maps: the heat, the headings and the features it reads are fixed, so the pair loss cannot correct a keypoint that sits a metre off or a heading that is wrong although all of them are inputs of the pair decision. The chain "maps -> keypoints (soft-argmax) -> pair logits -> connectors" is differentiable end to end except one named hard step (the 3 x 3 peak choice of the keypoints; the position is a soft-argmax of the window, `jends.py`). The critic of 2026-11-11 asked for the joint option and for a gradient assertion in the self-test: both are part of this card (`jends_train.py --e2e`, `jends_eval.py selftest`).

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
