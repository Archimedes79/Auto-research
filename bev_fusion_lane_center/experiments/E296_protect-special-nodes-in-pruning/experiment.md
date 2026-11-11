# E296 Prune, but never the special types: stop / exit / split / merge nodes protected in the close-node step (brief: "Spezialtypen nicht prunen") (D82)

Card of 2026-11-11, CPU replay like E288 (the stored full prediction dump of B, no training). The hard close-node step of record (`close:1.5@0.5`) and its differentiable twin (`mnms:1.75s0.1@0.5`, E288) drop the weaker of two nodes closer than the radius; the entry and exit nodes of a junction are the ones the pair head needs (E284: the delivered flags miss 25 - 33 % of the GT ends). The spec suffix `x` protects every node with a stop / exit / split / merge flag probability >= 0.5 from being suppressed (it still suppresses the weaker nodes around it).

## 0. The card (what the tools run)
```json card
{
 "id": "E296",
 "card": "",
 "title": "close-node step and Matrix-NMS with the special-type nodes protected (spec suffix x), against the same steps without the protection",
 "kind": "variant",
 "deterministic": true,
 "metric": "junction_F1",
 "direction": "higher",
 "band": 0.02,
 "prediction": {"value": 0.01, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = `close:1.5@0.5` (the hard step of record) on the dump of the same seed"},
  "x": {"recipe": "`close:1.5x@0.5` and `mnms:1.75s0.1x@0.5` (`qgraph_readout`, `readout_replay.py score`)"}
 },
 "control": "c",
 "variant": "x",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "junction_F1", "vs": "c", "min_effect": 0.02},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.02},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0}
 ],
 "budget_gpu_h": 0.0,
 "kill": "the replay of `close:1.5@0.5` does not reproduce the stored plain graph of E284 (topo F1 0.6259 / 0.6249)",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): KEEP when junction F1 rises by more than 0.02 on both seeds with the guards held; TIE kept and refined (D82); CUT below -0.01 or a broken guard. Reported: lanes found, Q, topo P / R, the nodes delivered, the flag counts that were protected. Command: `readout_replay.py score --seed <S> --specs close:1.5@0.5 close:1.5x@0.5 mnms:1.75s0.1@0.5 mnms:1.75s0.1x@0.5`.

## 1. Result of the first round (2026-11-11, CPU replay, VAL 873, both seeds; `runs/e296/replay_s0.json`, `replay_s1.json`)
Columns: topo F1 / junction F1 (junction recall) / lanes found / duplicates / vertices per 100 m (DP).
| spec | seed 0 | seed 1 |
|---|---|---|
| `close:1.5@0.5` (c) | 0.6259 / 0.2712 (0.555) / 0.6267 / 0.2345 / 19.28 | 0.6249 / 0.2493 (0.501) / 0.6137 / 0.2012 / 20.08 |
| `close:1.5x@0.5` (protected, p >= 0.5) | 0.6227 / 0.2642 (0.585) / 0.6338 / 0.2530 / 19.72 | 0.6214 / 0.2464 (0.540) / 0.6202 / 0.2123 / 20.63 |
| `mnms:1.75s0.1@0.5` | 0.6270 / 0.2707 (0.551) / 0.6236 / 0.2298 / 19.23 | 0.6250 / 0.2495 (0.496) / 0.6113 / 0.1999 / 19.98 |
| `mnms:1.75s0.1x@0.5` (protected) | 0.6237 / 0.2631 (0.585) / 0.6328 / 0.2521 / 19.70 | 0.6219 / 0.2473 (0.540) / 0.6195 / 0.2122 / 20.56 |
**Verdict: TIE, kept and refined (D82).** Protecting the special types raises the junction RECALL by 0.03 - 0.04 and lanes found by 0.007 - 0.008 on both seeds and for both steps, but the junction F1 does not rise (-0.007 / -0.003: the precision falls as much as the recall rises), topo F1 falls by 0.003 and the duplicates rise by 0.011 - 0.019; the guards hold (duplicates +0.0185 on seed 0 is inside 0.02, vertices +0.4 / +0.5). The prediction (junction F1 +0.01) was missed: the extra protected nodes are mostly false or duplicate special-type nodes (the delivered flags are 48 - 49 % precise, E283). **Refinement round (running):** protect only the confident ones: `close:1.5x0.7@0.5` and `close:1.5x0.9@0.5` (the flag probability from which a node is protected); the next lever is the flag precision itself (E301 / E286 maps), not the pruning rule.

## 2. Result of the refinement round (2026-11-11, CPU replay, both seeds; `runs/e296/replay_s0_refine.json`, `replay_s1_refine.json`)
| `close:1.5x<p>@0.5` | seed 0: topo F1 / junction F1 (recall) / lanes found / dup / vertices | seed 1 |
|---|---|---|
| c: no protection | 0.6259 / 0.2712 (0.555) / 0.6267 / 0.2345 / 19.28 | 0.6249 / 0.2493 (0.501) / 0.6137 / 0.2012 / 20.08 |
| x0.5 | 0.6227 / 0.2642 (0.585) / 0.6338 / 0.2530 / 19.72 | 0.6214 / 0.2464 (0.540) / 0.6202 / 0.2123 / 20.63 |
| x0.7 | 0.6236 / 0.2678 (0.574) / 0.6320 / 0.2512 / 19.51 | 0.6232 / 0.2505 (0.529) / 0.6191 / 0.2104 / 20.35 |
| **x0.9** | **0.6250 / 0.2693 (0.561) / 0.6293 / 0.2455 / 19.4** | **0.6245 / 0.2524 (0.517) / 0.6165 / 0.2054 / 20.3** |
**Verdict: TIE, kept (D82): the protection threshold 0.9 costs nothing (topo F1 -0.0009 / -0.0004, junction F1 -0.002 / +0.003, duplicates +0.011 / +0.004, vertices +0.1 / +0.2) and gives a little recall (junction recall +0.006 / +0.016, lanes found +0.003 / +0.003).** No setting reaches the KEEP line (junction F1 +0.02): the protected nodes are mostly the right ones at p >= 0.9 and mostly wrong below; the delivered flags are the limit, not the rule. The rule `x0.9` is the default for later read-outs that protect the special types (it matches the brief: "Spezialtypen nicht prunen"); the real lever is the flag precision of the maps (E286 / E287). Not done: the same for `mnms` at x0.9 (the same logic, a replay of 8 minutes if it is wanted).
