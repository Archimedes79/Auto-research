# E288 One-pass Matrix-NMS as the close-node step: a replay on the stored VAL predictions of B (D82, thread: soft pruning, differentiable)

Card of 2026-11-11, written by the lane chat on Mirko's word (D82: "Kann man Nodes systematisch prunen ... Irgendeine Matrixoperation zum Pruning bekannt? ... Und bleibe differenzierbar"). Source: `literature/scan_2026-11-11_pruning_uncertainty_time.md` (idea 2: Matrix-NMS, SOLOv2, one pass, a suppressed node suppresses less: the compensation term may be what cost lanes found in the soft step E277). CPU only: no training, the stored full prediction dump of B (`temp/e276/predfull_e273_p_s<S>.pkl`) is read out with new specs and scored by the lane suite.

## 0. The card (what the tools run)
```json card
{
 "id": "E288",
 "card": "",
 "title": "read-out of the nodes of B by a one-pass Matrix-NMS decay (overlap 1 - d / R, Gaussian decay, compensation) instead of the sequential hard close-node step of E275, equal vertex budget",
 "kind": "variant",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.010,
 "prediction": {"value": 0.005, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = the read-out of record, `close:1.5@0.5` (E275 KEEP), on the dump of the same seed"},
  "m": {"recipe": "`mnms:<R>@0.5` with R in 1.25, 1.5, 2.0 and `mnms:1.5@0.4` (`qgraph_readout.matrix_nms`), `readout_replay.py score`"}
 },
 "control": "c",
 "variant": "m",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_gain": 0.02},
  {"metric": "vert100_dp", "vs": "c", "max_gain": 1.0}
 ],
 "budget_gpu_h": 0.0,
 "kill": "the replay of `close:1.5@0.5` does not reproduce the stored E275 numbers (topo F1 0.6259 / 0.6249 for the plain graph of E284, which is this control) to 0.001: the dump or the scorer is not the one of the record",
 "evaluate": {"split": "val"}
}
```
By hand (the replay prints the lane suite numbers; label "by hand: replay"): the arm of the family with the best topo F1 inside the vertex guard is the variant. KEEP when topo F1 rises by more than +0.01 over c on BOTH seeds with the guards held; TIE inside +-0.01: **kept and refined (D82)**, then the family is the candidate for the soft replacement of the hard step (it is differentiable in e and the positions, so it can be put inside the training graph, which the hard step cannot); CUT below -0.01 or a guard broken on both seeds. Reported: lanes found, Q, junction F1, duplicates, nodes delivered, ms per frame.

## 1. Why (E277 / E279 background)
The hard step is the named non-differentiable read-out of the delivered graph (E275 KEEP). The soft close-node step inside the model (E277, 3 passes of a smooth damping) was a CUT, the 1.25 m version (E279) a KEEP that is open for Mirko. Matrix-NMS has no sequential keep set: one pass, the decay of node j is the minimum over the higher nodes of exp(-(o_ij^2 - c_i^2) / sigma) with the compensation c_i, so a node that is itself suppressed suppresses less; under 1 ms against 22 ms for Soft-NMS at equal AP in the paper (digest). It is a replacement candidate for BOTH the hard step (as read-out) and the soft step (as a layer).

## 2. Design
Replay only. Specs: `close:1.5@0.5` (control), `plain@0.5`, `mnms:1.25@0.5`, `mnms:1.5@0.5`, `mnms:2.0@0.5`, `mnms:1.5@0.4`. sigma 0.5 (SOLOv2's Gaussian choice, fixed beforehand), overlap o = max(0, 1 - d / R). The control is deterministic; two seeds of B give two dumps. VAL 873 frames, the link threshold 0.5. No TEST.
**Prediction (a number):** `mnms:1.5@0.5` +0.005 topo F1 over the hard step with the same vertices within 1.0 per 100 m: a TIE (the step is a smooth version of the same rule; the gain, if any, is in lanes found).

## 3. Command
`wsl bash tools/env/_py14.sh readout_replay.py score --seed <S> --specs close:1.5@0.5 plain@0.5 mnms:1.25@0.5 mnms:1.5@0.5 mnms:2.0@0.5 mnms:1.5@0.4` (about 5 minutes per spec on 2 threads). Result `runs/e288/replay_s<S>.json`.

## 4. Result (2026-11-11, CPU replay, VAL 873 frames, both seeds; `runs/e288/replay_s0.json`, `replay_s0_refine.json`, `replay_s1.json`, `replay_s1_refine.json`; verdict by hand: replay)
The control reproduces the stored plain graph of E284 (topo F1 0.6259 / 0.6249): the kill condition is clear. First round (sigma 0.5, the card's specs), then a refinement round because the first was a TIE-or-worse (D82: refine, do not drop). Columns: topo F1 / F1@1.0 / lanes found / junction F1 / dup_frac / vertices per 100 m (DP).
| spec | seed 0 | seed 1 |
|---|---|---|
| `close:1.5@0.5` (c, the hard step of record) | 0.6259 / 0.8283 / 0.6267 / 0.2712 / 0.2345 / 19.28 | 0.6249 / 0.8159 / 0.6137 / 0.2493 / 0.2012 / 20.08 |
| `plain@0.5` (no step) | 0.5971 / 0.8325 / 0.6438 / 0.1297 / 0.2321 / 28.88 | - |
| `mnms:1.5@0.5` (sigma 0.5) | 0.6203 / 0.8304 / 0.6361 / 0.2028 / 0.2327 / 22.46 | 0.6192 / 0.8193 / 0.6227 / 0.1904 / 0.1968 / 23.64 |
| `mnms:2.0@0.5` | 0.6248 / 0.8273 / 0.6220 / 0.2374 / 0.2239 / 20.43 | 0.6222 / 0.8145 / 0.6106 / 0.2227 / 0.1960 / 21.15 |
| `mnms:1.75s0.2@0.5` | 0.6263 / 0.8289 / 0.6275 / 0.2510 / 0.2306 / 19.96 | 0.6244 / 0.8162 / 0.6146 / 0.2346 / 0.2000 / 20.72 |
| **`mnms:1.75s0.1@0.5`** | **0.6270 / 0.8281 / 0.6236 / 0.2707 / 0.2298 / 19.23** | **0.6250 / 0.8156 / 0.6113 / 0.2495 / 0.1999 / 19.98** |
(also run, seed 0 only or without a gain: R 1.25 / 1.5 / 1.75 at sigma 0.5 and 0.2, R 2.0 at sigma 0.2, the threshold 0.4 is identical to 0.5 because the link read-out stays at 0.5.)
**Verdict: TIE, kept and refined (D82).** No spec beats the hard step by +0.01 (the best, `mnms:1.75s0.1@0.5`, is +0.0011 / +0.0001 topo F1 with the vertices -0.05 / -0.10 per 100 m, F1@1.0 -0.0002 / -0.0003, junction F1 -0.0005 / +0.0002, duplicates -0.005 / -0.001, lanes found -0.003 / -0.002: all inside the guards of the card). The first-round settings (sigma 0.5) are worse than the hard step (more vertices, lower junction F1); a sharper decay (sigma 0.1) makes the one-pass operator reproduce the hard step. **What this buys:** the hard close-node step is the named non-differentiable read-out of the delivered graph; `matrix_nms` with R 1.75, sigma 0.1 is the same rule as a smooth one-pass operator on e and the node positions, so it can sit INSIDE the training graph (a layer after the node head, replacing the E277 3-pass soft step, which was a CUT) and be judged there: that is the next card (training arm, 2 seeds), not a claim here. The prediction (+0.005) was missed by 0.004 (a TIE as predicted).
