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
