# E284 The junction end head: tolerant entry / exit keypoints and a pair decision on the frozen lane head (baseline B + the E275 hard step, D55 / D58 / D71 / D79)

Card of 2026-11-10, written by the lane chat on Mirko's word ("starte die diskutierten Experimente und arbeite weiter"; the metric was delegated to the chat by him the same day, D79). Origin: E283 sections 4 - 8 (the junction blur is the unknown pair of entry and exit; the delivered flags miss 25 - 33 % of the GT ends and about half of them sit at no GT end; with clean ends and the classifier's pairs the scorer's topo F1 rises by +0.04 to +0.05, with the measured spread alone by about +0.02, with the gaps of the delivered flags by +0.006 to +0.016 when the delivered connectors are kept).

**Disclosure.** The written reading of the CPU oracle `p22_keep_oracle.py` (E283 section 8) was: K3 fails -> the lever is closed for the GPU. This card runs the head anyway, for two reasons given before the run: the GPU is free in the night window (22 - 07) and the run is small (0.16 M trainable parameters on frozen stages, about 15 minutes per seed), and the oracle's gaps (25 % ends missing, half of the ends false) are an ASSUMPTION about what a tolerant head does; the run measures it. The prediction below is therefore a TIE, not a KEEP, and the rule is the one of D79 as written.

## 0. The card (what the tools run)
```json card
{
 "id": "E284",
 "card": "",
 "title": "the junction end head (phase 8 of qgraph_train.py: entry / exit keypoints with a Gaussian target of 2 m, offset and heading, plus a pair decision; 4000 steps on the frozen pixel / node / link stages of B) whose connectors are added to the delivered graph of B + the hard step of E275 (a delivered long link stays unless a head curve covers it), against that delivered graph",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.02,
 "prediction": {"value": 0.010, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = B + the hard step of E275 (runs e273_p_s0 / e273_p_s1, close:1.5@0.5), the delivered graph of the same seed, scored by the same scorer (`qends_eval.py score`, key `base`); no new training"},
  "h": {"recipe": "the same graph plus the connectors of the end head: `qgraph_train.py --phases 8 --resume-from runs/e273_p_s<S>/model.pt --steps 4000` (runs `e284_h2_s<S>`, `.pt` equals B's tensors outside `ends.*`, checked by `qends_eval.py score`), probability threshold 0.5 fixed beforehand, one Hermite curve (tangent = chord) per pair, `qends_eval.py dump` then `score`"}
 },
 "control": "c",
 "variant": "h",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.02},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_gain": 0.02},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 1.0,
 "kill": "a loss that is not finite; or the end gate (numbers only) unreadable; or the model.pt not equal to B's tensors outside ends.* (the delivered graph would not be B's)",
 "evaluate": {"split": "val"}
}
```
**By hand (the tools cannot read these numbers; the verdict script is `evaluation/qends_eval.py score`, the rule is D79's and is applied to the plain delivered graph of the same seed, VAL 873 frames):** KEEP when topo F1 rises by more than +0.02 on BOTH seeds with F1@1.0 not lower by more than 0.01, dup_frac not higher by more than 0.02, vertices per 100 m (Douglas-Peucker) not higher by more than 1.0, junction F1 not lower by more than 0.03; CUT when topo F1 falls by more than 0.02 on both seeds or a guard breaks on both; TIE otherwise (a mixed pair of seeds is INCONCLUSIVE: a third seed would be the next step, not an adoption). The guard 'assignment gain' of D79 is not applicable (B's assignment is untouched by construction). Reported, not judged: lanes found, Q, topo P / R, junction P / R, the head's own numbers (entry / exit recall and precision with a tolerance of 4 m / 30 degrees, pair recall and precision, `ends_head.json`), the curves added and wired at both ends.

## 1. The head (`training/grid/qendnet.py`, tested by `python qendnet.py`; data `training/grid/qends.py`)
Heat maps over the 2 m cells for entries and exits (penalty-reduced focal loss on a Gaussian target of 2 m, the cell that holds an end = 1), a position offset (+-5 m) and a heading per cell (trained within 3 m of an end), keypoints = 3 x 3 peaks with p >= 0.1 (32 per kind; in training the GT end cells are always candidates), 2 layers of self-attention over the keypoints of the frame, a pair logit for every geometrically possible (entry, exit) pair (`qends.allowed`: turn <= 130 degrees, chord >= 2 m, ahead of both), BCE (positive weight 3) + a dual softmax with a 'none' entry. Labels: a keypoint is the nearest GT end of its kind within 4 m and 30 degrees, one keypoint per GT end (the nearest; a duplicate is a negative); a pair is true when its two ends are the two ends of a GT connector run. The GT ends are the starts / ends of the connector runs of the lane GT l4 (`qends.end_table`); section 9 of E283 shows that the entry point of the GT coincides with the painted stop line where there is one (64 % within 2 m), so the target needs no new GT version (D78 stays: nothing of the l6 / v13 work is merged).
Why the head cannot damage B: `QGraph(ends=True)` adds only the module `ends`; phase 8 trains it under `no_grad` on B's pixel and node stages; the checkpoint equals B's tensors outside `ends.*` (the scorer asserts it); the delivered graph is B's full dump of the same seed plus the connectors.

## 2. Design
Two seeds (the seed of B and of the head are the same; the head sees the same TRAIN frames, 8414), 4000 steps, lr 1e-3, batch 4, GT end cells added to the candidates in training only, the threshold 0.5 and the keep integration fixed here, before any VAL number of the head exists. VAL 873 frames (the lane GT l4 store 0702a065c6); TEST untouched. Controls: the plain delivered graph of the same seed (c); the oracle numbers of E283 section 8 (clean ends V3, measured spread K1, gappy ends K3 / K4) are the reference for what the head's ends are worth: K3 / K4 +0.013 / +0.016 and +0.006 / +0.009 topo F1 (seed 0 / 1). Not built: the pair decision trained on the soft heat (no hard keypoint step) - the first version keeps the named index choices.
**Prediction (a number):** the head's end recall at 0.5 about 0.70, the pair recall about 0.35, topo F1 +0.010 over c (between K3 and K4), lanes found +0.07: a TIE. A KEEP needs the head to beat the delivered flags clearly (end recall above about 0.9 with few false ends).

## 3. Command and records
`wsl bash temp/e284/start_ends.sh <seed>` (scratch wrapper; the exact command is in the run record), then `wsl bash tools/env/_py14.sh qends_eval.py dump --run e284_h_s<S>` and `score --run e284_h_s<S> --seed <S>`. Runs E284.h2_s0, E284.h2_s1 through `run.py` with this card (hyp / pred / rule on the command line). Result files: `runs/e284_h2_s<S>/ends_head.json`, `ends_card.json`.

## 4. Revision before any result (2026-11-10)
The first attempts E284.h_s0 / h_s1 (K = 16, no one-to-one matching, plain scatter) were stopped at step 1600 of 4000 after the code critic (`ar-critic`, mode code) found three defects that change what the numbers mean: padded ends at (0, 0) overwrote a real end in the origin cell (about 4 % of the VAL frames) in the candidates and in the heat target; K = 16 is too small (21 % of the frames have more than 16 GT ends of a kind, up to 32); two keypoints near one GT end both counted as matches (duplicate connectors trained, precision inflated). Fixed (scatter_reduce amax, K = 32, the nearest keypoint per GT end is the only positive, `ends_forward` defaults to gt=False), selftest extended; no VAL number of the head had been read. The runs are E284.h2_s0 / h2_s1; prediction, rule, threshold and steps are unchanged. Nit noted, not changed: training runs the frozen stages under bf16 autocast, the dump in fp32. The node ids of added curves in `qends_eval.py` are unique per pair (the oracles shared them per end), so S and vertices per 100 m of the node-id count are not comparable with E283; topo F1, Q, F1@1.0, junction F1, dup_frac and vert100_dp (the rule's numbers) do not use the ids.
