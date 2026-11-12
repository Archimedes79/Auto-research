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
  "s": {"recipe": "`qends_eval.add_connectors(snap=True)` (promoted from the scratch `snapconn.py` after the critic pass; scored by `jends_eval.py score --snap --thr t`, the scorer of record with only the connector function changed), pairs at p >= t"}
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

## 2. Confirmation (2026-11-12 10:15, CPU, VAL 873, `runs/<run>/ends_card_snap_thr<t>.json` (the first confirmation scores were written by a scratch scorer into `temp/gt2/snap/` with the same code path and copied there), the scorer of record with only the connector function changed (`jends_eval.py score --snap` after the promotion); differences to the plain graph of the same seed; rule applied by `jdiag.py snaprule e287_p2 e305_e`)
| network | p >= | seed 0 topo F1 / F1@1.0 / dup / vert / junction F1 | seed 1 | rule |
|---|---|---|---|---|
| E287 p2 | 0.5 | +0.0144 / +0.0088 / -0.0198 / +0.09 / +0.0088 | +0.0132 / +0.0113 / -0.0163 / +0.07 / +0.0205 | fails (topo F1 below +0.02) |
| E287 p2 | 0.2 | +0.0241 / +0.0157 / -0.0400 / +0.20 / +0.0234 | +0.0229 / +0.0237 / -0.0328 / +0.09 / +0.0496 | holds |
| E287 p2 | 0.1 | +0.0265 / +0.0195 / -0.0478 / +0.20 / +0.0333 | +0.0253 / +0.0297 / -0.0382 / -0.02 / +0.0606 | holds |
| E305 | 0.5 | +0.0206 / +0.0106 / -0.0267 / +0.15 / +0.0157 | +0.0161 / +0.0122 / -0.0197 / -0.00 / +0.0228 | seed 1 fails |
| E305 | 0.2 | +0.0290 / +0.0172 / -0.0460 / +0.23 / +0.0301 | +0.0256 / +0.0219 / -0.0362 / +0.02 / +0.0428 | holds |
| E305 | 0.1 | +0.0304 / +0.0195 / -0.0518 / +0.25 / +0.0373 | +0.0285 / +0.0275 / -0.0384 / -0.05 / +0.0525 | holds |
| E308 (exploratory) | 0.5 / 0.1 | +0.0147 / +0.0255 | +0.0139 / +0.0283 | 0.5 fails, 0.1 holds |
**Verdict by the rule written before these runs: KEEP at p >= 0.2 and p >= 0.1** (all four confirmation cells hold: topo F1 +0.023 .. +0.030, F1@1.0 +0.016 .. +0.030, duplicates -0.033 .. -0.052, vertices per 100 m within +0.25, junction F1 +0.023 .. +0.061); **at p >= 0.5 TIE** (+0.013 .. +0.021; kept). The lowest threshold that holds wins: **p >= 0.1 with snapped ends is the delivered read-out of the pair head** in the report and in the follow-up cards (prediction +0.015 over c at p >= 0.5: hit at 0.5 on E308 and E305 seed 0, missed on E287 p2 and E305 seed 1; at 0.1 exceeded). Also at p >= 0.1: lanes found +0.060 .. +0.078, junction recall +0.07 .. +0.16, F1@1.0 +0.018 .. +0.030. Against the connectors drawn from the keypoints at the same threshold (E309: topo F1 +0.004 .. +0.007 over the plain graph on E287 p2 and E308) the snapping is worth about +0.02 topo F1. What is not settled: the thresholds below 0.1 (queued, `temp/gt2/wire_run.sh`), the wiring radius and angle (4 m / 45 degrees, queued) and the effect on the learned handle scales of E303.
The best network on this read-out: E305 (maps and pair head trained together), topo F1 0.6563 / 0.6534 against the plain graph 0.6259 / 0.6249 at p >= 0.1.

## 3. Critic pass and what the controls say (2026-11-12 10:35)
Independent review of the diff (`qends_eval.add_connectors(snap=...)`, `--snap` in the scorers): the default path is unchanged (660 calls on 110 real frames, 0 differences against HEAD), the snapped path equals the scratch version of the first scores (150 frames), no NaN or crash; findings fixed: the self-test now has decoy pieces (wrong direction, farther right-direction pieces first in the list, nearer wrong-direction pieces) and all nine mutations of the snap logic are caught; a scoring at another threshold than 0.5 no longer overwrites `ends_card.json` (only the unsnapped read-out at p >= 0.5 writes it; every other read-out has its own file). The review also named a CONFOUND: the snapped read-out wires only ONE piece per end (the unsnapped one wires every piece in reach, 40 - 50 % of the pairs have more than one in reach), so the gain mixes the geometry with fewer edges. Controls added as options (default off) and run on E305 seed 0 at p >= 0.1 (topo F1 over the plain graph; `temp/gt2/snapopt/`):
| read-out | geometry | wiring | topo F1 vs plain | junction F1 | junction recall | dup_frac |
|---|---|---|---|---|---|---|
| unsnapped (E287 / E308, p >= 0.1, other networks) | keypoints | every piece in reach | about +0.004 .. +0.007 | -0.015 .. +0.014 | +0.09 .. +0.23 | -0.06 |
| `--snap --snap-nogeom` | keypoints | nearest piece per end | +0.0179 | +0.0229 | +0.1201 | -0.0515 |
| `--snap --snap-all` | snapped | every piece in reach of the snapped end | +0.0141 | +0.0221 | +0.1376 | -0.0678 |
| **`--snap` (E311)** | snapped | nearest piece per end | **+0.0304** | +0.0373 | +0.0927 | -0.0518 |
| `--snap --snap-guard` (degenerate snapped curves keep their keypoints) | snapped | nearest | +0.0303 | +0.0355 | +0.0934 | -0.0521 |
Reading: both parts count and they add up: the geometry alone gives about +0.009, the one-piece wiring alone about +0.013, together +0.025 over the unsnapped read-out. The guard changes nothing (the 26 degenerate curves in 2049 pairs do not matter). The wiring radius 6 m instead of 4 m changes nothing either (+0.0308 / +0.0283 against +0.0304 / +0.0285). The verdict of section 2 is unchanged: it is the snapped read-out with one piece per end as a whole.
