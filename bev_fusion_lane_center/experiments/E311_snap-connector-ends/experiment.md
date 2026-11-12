# E311 - snap the connector ends to the delivered lane pieces

Found 2026-11-12 while looking for why the numbers did not match the pictures (E310). The connector of a pair is a curve between the entry keypoint and the exit keypoint; the keypoints are 1.4 m off the GT ends on average, so the curve ends beside the delivered lane pieces and the topology scores see a gap. The keypoints decide WHICH entry joins which exit; the geometry of the connector should start where the delivered piece ends and end where the next piece starts. CPU only, a change of the read-out, no training.

## 0. The card (what the tools run)
```json card
{
 "id": "E311",
 "card": "",
 "title": "connectors whose ends are snapped to the end of the delivered piece wired to the entry and to the start of the piece wired to the exit (nearest within 4 m and 45 degrees), against the connectors drawn from the keypoints, on the pairs of E287 p2 / E308 / E305",
 "kind": "variant",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.02,
 "prediction": {"value": 0.015, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = the plain delivered graph of B + the hard close-node step of the same seed (no connectors); the unsnapped connectors of the same dump are the second reference"},
  "s": {"recipe": "`temp/jr/snapconn.py` `add_connectors_snap` (scored by `temp/gt2/snap_score.py`: the scorer of record `qends_eval.cmd_score` with only the connector function replaced), pairs at p >= t"}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.02},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.0},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 0.0,
 "kill": "the unsnapped scores at the same threshold do not reproduce the stored ends_card.json of the run",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): the rule is the D79 rule of E287 applied to the plain graph of the same seed: KEEP a threshold when it holds on BOTH seeds of EVERY network it is tried on (E287 p2, E305; E308 is the network on which it was found, so it counts as exploratory). Thresholds 0.5 / 0.2 / 0.1; among those that hold the lowest wins (more connectors, the maneuver coverage and lanes found rise; a tie inside the band: the higher threshold). TIE (kept, D82) when only some guard fails by a hair. The decision makes the snapped connectors, at the winning threshold, the delivered read-out of the pair head in the report and the follow-up cards; the code default (`qends_eval.add_connectors`) changes only with a test after a critic pass.

**Order of events (honest):** the snapped variant was first scored on the E308 dumps at p >= 0.5 and 0.1 (below); the card and its rule were written after seeing them, so E308 is exploratory; the confirmation runs are E287 p2 and E305, which were not scored with it before.

## 1. First scores on E308 (2026-11-12, CPU, VAL 873, `temp/gt2/snap/e308_t_s<S>_thr<t>.json`; differences to the plain graph of the same seed)
| seed | p >= | curves | topo F1 | vs plain | lanes found | junction F1 | junction recall | dup | F1@1.0 | topo P | topo R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 2977 | 0.6406 | +0.0147 | +0.0262 | +0.0092 | +0.0139 | -0.0195 | +0.0067 | +0.0074 | +0.0204 |
| 0 | 0.1 | 8307 | 0.6514 | +0.0255 | +0.0601 | +0.0316 | +0.0724 | -0.0501 | +0.0175 | -0.0012 | +0.0477 |
| 1 | 0.5 | 3546 | 0.6388 | +0.0139 | +0.0249 | +0.0233 | +0.0479 | -0.0198 | +0.0112 | -0.0054 | +0.0274 |
| 1 | 0.1 | 9131 | 0.6532 | +0.0283 | +0.0703 | +0.0574 | +0.1384 | -0.0433 | +0.0276 | -0.0192 | +0.0649 |
The same dumps drawn from the keypoints (E309): topo F1 +0.0064 / +0.0046 at 0.5 and +0.0054 / +0.0071 at 0.1. Snapping roughly doubles the topology gain and turns the junction F1 from -0.015 / +0.014 into +0.032 / +0.057 at p >= 0.1; the maneuver coverage (a pure geometry measure, E310) does not see it (53.6 % against 53.8 % at 0.5): the gain is in how the graph is connected, which the topology scores measure. The F1@1.0, duplicates and topo precision guards hold.
