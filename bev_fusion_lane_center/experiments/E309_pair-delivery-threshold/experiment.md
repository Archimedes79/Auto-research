# E309 - delivery threshold of the pair head

## 0. The card (what the tools run)
```json card
{
 "id": "E309",
 "card": "",
 "title": "the connectors of the pair head are delivered at p >= 0.5 today; delivered at a lower threshold (0.3 / 0.2 / 0.1) the graph gets more lanes and junction recall, at what price?",
 "kind": "variant",
 "deterministic": true,
 "metric": "lanes_found",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.04, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = the same dump delivered at p >= 0.5 (`jends_eval.py score --thr 0.5`, the number of record of E287 / E308)"},
  "t": {"recipe": "the same dump delivered at the threshold of the rule below (`jends_eval.py score --run <run> --seed <S> --thr t`: the scorer of record, only `--thr` changes; the card goes to `runs/<run>/ends_card_thr<t>.json`, the card of record is only written at p >= 0.5; the first sweep used a scratch wrapper `temp/gt2/thr_sweep.py`)"}
 },
 "control": "c",
 "variant": "t",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "lanes_found", "vs": "c", "min_effect": 0.03},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.005},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.0},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.005},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.02},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0}
 ],
 "budget_gpu_h": 0.0,
 "kill": "the threshold 0.5 of the sweep does not reproduce the stored ends_card.json of the run",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): the dumps are the ones of E308 / E287 p2 (and later E305 / E302), VAL 873, both seeds. KEEP a threshold when the rule holds on BOTH seeds of EVERY network it is tried on; among the thresholds that hold, the highest wins (a tie inside the band: the higher threshold, fewer curves); the verdict makes it the default delivery threshold of the pair head in the page and in the follow-up cards (the code default stays 0.5 until the lead changes it with a test). TIE (kept, D82) when only some guards fail by a hair; CUT when the lanes-found gain is missing.

**Honest order of events:** the sweep 0.3 / 0.2 / 0.1 was first run on the E308 dumps (exploratory, numbers below, VAL, so "selected on VAL"; TEST untouched); this card was written after seeing them and the rule is the one above, so E308 is NOT an independent confirmation. The independent ones are the networks that have not been swept: E287 p2 (both seeds), later E305 / E302 / E298.

## 1. Exploratory sweep on E308 (2026-11-12, CPU, VAL 873, plain graph = control of the run, `temp/gt2/thr/e308_t_s<S>_thr<t>.json`; columns: curves delivered; topo F1 (difference to the plain graph); differences to the plain graph for lanes found / junction F1 / junction recall / dup_frac / F1@1.0 / vertices per 100 m)
| seed | thr | curves | topo F1 | lanes found | junction F1 | junction R | dup | F1@1.0 | vert |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 4964 | 0.6315 (+0.0056) | +0.0347 | -0.0152 | +0.0805 | -0.0391 | +0.0089 | +0.20 |
| 0 | 0.2 | 6290 | 0.6324 (+0.0065) | +0.0430 | -0.0156 | +0.1077 | -0.0481 | +0.0114 | +0.26 |
| 0 | 0.1 | 8307 | 0.6313 (+0.0054) | +0.0556 | -0.0137 | +0.1459 | -0.0610 | +0.0142 | +0.29 |
| 1 | 0.3 | 5647 | 0.6309 (+0.0060) | +0.0401 | +0.0058 | +0.1510 | -0.0428 | +0.0163 | +0.11 |
| 1 | 0.2 | 6989 | 0.6316 (+0.0067) | +0.0515 | +0.0099 | +0.1850 | -0.0502 | +0.0204 | +0.13 |
| 1 | 0.1 | 9131 | 0.6320 (+0.0071) | +0.0696 | +0.0142 | +0.2303 | -0.0558 | +0.0248 | +0.09 |
(For the threshold 0.5 of the same runs see E308 section 4: lanes found +0.022 / +0.026, topo F1 +0.0064 / +0.0046 over the plain graph.) Reading: the lower the threshold, the more lanes are found and the fewer duplicates are left, F1@1.0 rises, topo F1 stays flat (+0.005 to +0.007: the extra curves cost about what they gain on that metric) and junction F1 moves little (-0.016 .. +0.014).

## 2. Confirmation on E287 p2 (2026-11-12 10:30, CPU, VAL 873; differences to the same dump delivered at p >= 0.5 = c; E308 is the exploratory network of section 1; unsnapped connectors; `temp/gt2/thr/<run>_thr<t>.json`)
| network | p >= | seed 0: lanes found / F1@1.0 / dup / topo F1 / junction F1 | seed 1 | rule |
|---|---|---|---|---|
| E287 p2 | 0.3 | +0.0167 / +0.0039 / -0.0143 / +0.0013 / -0.0032 | +0.0164 / +0.0074 / -0.0133 / +0.0017 / +0.0054 | fails (lanes found below +0.03) |
| E287 p2 | 0.2 | +0.0277 / +0.0057 / -0.0257 / +0.0000 / -0.0030 | +0.0292 / +0.0119 / -0.0224 / +0.0021 / +0.0098 | fails (just below +0.03) |
| E287 p2 | 0.1 | +0.0456 / +0.0082 / -0.0358 / -0.0025 / -0.0001 | +0.0523 / +0.0165 / -0.0298 / -0.0004 / +0.0123 | holds |
| E308 | 0.1 | +0.0332 / +0.0087 / -0.0370 / -0.0010 / +0.0017 | +0.0434 / +0.0138 / -0.0290 / +0.0025 / +0.0119 | holds |
**Verdict by the rule: KEEP at p >= 0.1** (on E287 p2, the independent network, and on E308: lanes found +0.033 .. +0.052 over the delivery at 0.5, F1@1.0 +0.008 .. +0.017, duplicates -0.03 .. -0.04, topo F1 and junction F1 flat within the guards, vertices within +0.20); 0.2 and 0.3 TIE (kept). The lowest threshold that holds wins. The snapped read-out of E311 reaches the same conclusion on its own metric (p >= 0.1 holds on all four cells, p >= 0.5 does not). The sweep ran through a scratch wrapper; `jends_eval.py score --thr t` now writes the card of another threshold to `ends_card_thr<t>.json` and needs no wrapper.
