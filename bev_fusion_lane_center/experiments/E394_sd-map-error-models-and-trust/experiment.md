# E394 - STEP 3 of the SD-map line: the ERROR MODELS of the map and a TRUST mechanism, so that a wrong map does not make the lanes per direction worse than no map

**PAUSED by D138 (2026-11-23), resume with step 0 (CPU, ~1 h): `bash tools/env/_py14.sh maperror_probe.py degrade --seed 0 --split holdout --modes all --every 2`, through run.py as `E394.step0_<part>_s0` once the card is a v1 (bars from step 0) and a fresh critic has passed it.** **VERSION 0 (DRAFT, NOT RUN: no step 0 was started, only a 22-frame smoke read of the probe: E391's F1 reads lanes_all 0.52 at a wrong scene against z 0.88, n = 22 frames, a hint, not a result).** Card of 2026-11-23 (system clock), written by the report chat on the lead's task of 08:20 (D137: the owner's go for both follow-ups after E391 'NO FUSION'; step 3 of D130: "Und man braucht Fehlermodelle, damit die Karte das Modell nicht kaputt macht", D135: a wrong map yields 'unassigned' or a low confidence, not a wrong overwrite). The E389 lesson: a WRONG map hurt a map head more than the no-map head (-0.043 / -0.053 against the clause -0.03). Base = the step-1 association head (E390 g7 weights) and E391's fusion head F1 (the learned association of NO FUSION: its lanes read-through gained +0.06 - 0.08 over perception alone); the perception stays untouched (D135, stop-gradient). Nothing of E390 / E391's code is edited: the new code lies in `training/data/lane_maperror_data.py`, `evaluation/maperror_probe.py` (and, after the critic, a gate module).

**Order (the lead's):** (1) the catalogue of map errors (section 1.1), (2) the trust candidates (section 1.2), (3) the clause pinned before any run (section 1.3), (4) CPU STEP 0 first (section 2: how each error degrades the heads, the nulls and the headroom; the bars are derived from it in version 1, never typed), (5) a fresh `ar-critic` on the card before any GPU run, then the runs. No TEST, no push.

## 0. The card (v0: the bars and the verdict numbers are written in v1 from step 0; the FORM of the rule is fixed here)
```json card
{
 "id": "E394",
 "card": "",
 "title": "E394 step 3 (v0): the error models of the SD map (a catalogue of calibrated nested draws: pose / shift / yaw, polyline simplification, lane-count errors (rate, size, unknown), missing / extra segments, missing / extra / swapped connections, an outdated map, a wrong route, a wrong scene, a gross shift) and a trust mechanism (the association's unassigned slot, a label-free map-perception consistency gate, a learned gate with a training mix of wrong maps, perception untouched) so that with a WRONG map the lanes per direction read through the map are not worse than the no-map head by more than a margin and with the right map the gain stays; scene bootstrap, two seeds, TRAIN-holdout read first, VAL once",
 "kind": "block",
 "deterministic": false,
 "metric": "lanes_all",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.35, "vs": "z", "note": "step 0 smoke (22 frames): E391's F1 at the wrong scene reads lanes_all 0.52 against z 0.88 (-0.35); a gate that sees the mismatch brings it to about 0.00 to -0.03; written precisely in v1"},
 "design_delta": [{"row": "R9", "change": "after a verdict: 'E394 step 3: <outcome> (the mechanism, the wrong-map harm against no-map, the gain kept at the right map)'; open to attack when -> the fusion of the map into the delivered graph (the perception is untouched by design)"}],
 "arms": {
  "step0": {"recipe": "CPU, no training: every mode of the catalogue at each level of its grid on an otherwise exact map, read with the zero-shot E390 head (p6zero), E391's F1 and the no-map head z (weights runs/e391_<arm>_s<S>, trained on the SIMPLE mix only), on the TRAIN holdout (every 10th TRAIN scene): the degradation curves, the nulls, the headroom; then the two deciding levels (a wrong map, the right map with realistic errors) with the scene bootstrap", "run_ids": ["E394.step0_a_s0", "E394.step0_b_s0"]},
  "a": {"recipe": "(a) the association's own 'unassigned' slot: E391's F1 as trained (simple mix), no gate; the reference of what the slot alone does under the new errors (read in step 0 and again at the deciding levels)", "run_ids": []},
  "u": {"recipe": "(c, part 1) F1 retrained on the CATALOGUE MIX (`lane_maperror_data.train_spec`: 35 % wrong maps), same data, steps, loss as E391 (3000 steps, bs 8, lr 1e-3, encoder frozen); no gate", "run_ids": []},
  "g": {"recipe": "(b) a label-free map-perception consistency gate on F1: frame-level features computed from the confident perceived tokens against the map (the share within 2 m / 4 m of a present segment, the heading agreement, the lane-count agreement of the no-map head's perceived lane class with the map's, the coverage of the map's segments by tokens), a logistic on them fitted on TRAIN draws of the mix with the frame label `is_wrong`; below the threshold the map read is not used (all tokens unassigned, lanes from z)", "run_ids": []},
  "c": {"recipe": "(c, part 2) the learned gate: a small MLP on the consistency features and the association's confidence statistics (max probability, the unassigned mass), trained on the frozen `u` with the mix (frame label `is_wrong` and a token-level 'the read is right' target), the same fall-back to z; an arm `ch` is trained without two modes (outdated, count_big) to read the gate on modes it never saw", "run_ids": []}
 },
 "control": "z",
 "variant": "g",
 "seeds": [0, 1],
 "rule": [{"metric": "lanes_all", "vs": "z", "min_effect": -0.03, "note": "by hand: section 1.3 clauses W, R, F (the FORM; the numbers from step 0 in v1); decide.py cannot read it"}],
 "budget_gpu_h": 0.6,
 "kill": "a loss is not finite; no learning (loss at step 3000 above 0.8 x the loss at step 100); the perception is touched (the gradient test of E391 section 1.4 fails for a new module); the step-1 encoder is not runs/e390_g7_s<S>; fewer labelled road tokens at a deciding level than the minimum counts of section 1.3; the catalogue's nesting test fails (a level changes the draws of another)",
 "evaluate": {"split": "val"}
}
```

## 1. The design
### 1.1 The catalogue of map errors (calibrated draws on the step-1 data layer)
Code: `training/data/lane_maperror_data.py` (`apply(G, rng, spec, other)`, `train_spec`, `other_plan`, `is_wrong`, `CATALOGUE`); E390's `perturb` is called with its own keys unchanged; the new modes are applied to the input graph after it, each with its own generator; draws are NESTED (a higher level only adds to a lower one) and the labels never change (`T['present']`: a missing segment's tokens are 'unassigned'; extra segments are appended to the INPUT graph only).
| mode | what | grid | 'realistic' and its source | new? |
|---|---|---|---|---|
| pose | map shifted by `pose` m AND turned by `pose` degrees about the ego (E390's coupled error) | 0.5, 1, 2, 4 | 1 - 2 m, yaw N(0, 1 deg): E374 / map_sim | E390 |
| shift | pure translation, random direction | 1, 2, 4, 6, 8 m | <= 2 m; 4 - 8 m is E389's gross | new |
| yaw | pure turn about the ego | 0.5 - 8 deg | N(0, 1 deg) | new |
| sigma | smooth deformation of the whole map | 1, 2, 4 m | E390 | E390 |
| eps | Douglas-Peucker simplification | 0.5, 1, 2, 4 m | 0.5 - 2 m: sd_lib.degrade_sd | E390 |
| count_rate / unknown_p | lane class off by one / unknown | 0.05 - 0.4 / 0.1 - 1.0 | 5 - 10 % (map_sim, degrade_sd); unknown 8 % (E389) | E390 |
| count_big | lane class off by two | 0.1, 0.3 | none: stress | new |
| drop / node_drop | successor links / whole segments missing | 0.1 - 0.4 / 0.05 - 0.4 | 5 - 10 % | E390 |
| hide | route flag removed | 0.3 - 1.0 | E390 | E390 |
| ghost | extra segments (a parallel or a crossing road that does not exist) | 0.1, 0.3, 0.6 per true road | none: stress | new |
| link_extra / link_swap | spurious / swapped connections | 0.1 - 0.6 / 0.1, 0.3 | none: stress | new |
| outdated | a stale map: intersections missing (approach roads linked straight to their exits), lane classes one higher | 0.3, 0.6, 1.0 | none: stress | new |
| wrong_route | the route flags on another path of the same map | 1 | E389 (another scene's route) | new |
| wrong_scene | the map of another scene (every true segment is missing from it) | 1 | E389 `wrong` | new |
**Honest limit:** only the first rows have a protocol (E374 / E077 / map_sim / E389); the others are STRESS levels without a real map to calibrate against (no real SD map is compared with the GT here): the result is the level at which a mechanism breaks, not a claim about a map provider's error rates.
**The TRAIN mix** (`train_spec`): exact 12 %, the realistic simple mix of E391 28 %, simple + one moderate new error 25 %, and 35 % WRONG maps (wrong scene 15 %, shift 4 - 8 m 10 %, route on another path 5 %, turn 4 - 8 deg 5 %): E389's lesson (5 % wrong maps were too few).

### 1.2 The trust candidates (what each can and cannot do)
- **(a) the association's confidence + 'unassigned'** (E391's F1 as it is). CAN: reject a token that matches no segment well (false tokens: 0.25 assigned). CANNOT: see that the MAP as a whole is wrong when some segment (a ghost, the next road of a shifted map) still fits a token; trained on simple errors only, its behaviour under the new modes is unmeasured: step 0.
- **(b) a label-free consistency gate** (frame level): the share of confident perceived tokens within 2 m / 4 m of a present segment, the heading agreement, the lane-count agreement of the no-map head's perceived lane class with the map's count, the coverage of the map's segments by tokens; a logistic on these few numbers. CAN: tell a map of another scene or a shifted / turned map from a right one (E076 of loop 6: a label-free geometric agreement score AUROC 0.996 where a learned scalar reached 0.82), no use of the association labels. CANNOT: see an error that leaves the geometry intact (a wrong route flag, a wrong lane count on one road: the lane-count agreement sees the second only where the no-map head knows its lanes).
- **(c) a learned gate with a training mix that contains wrong maps** (token level and frame level, on the frozen `u`). CAN: learn the thresholds and combine the features with the association confidence; CANNOT: be better than its mix (the held-out-mode arm `ch` measures that) and it can fall back to z only where z is available (the lanes, not the association itself).
- **(d) stop-gradient / separate heads** (D135): the perception (the delivered graph) cannot change by construction. CAN: guarantee that a wrong map does not touch the graph; CANNOT: protect the attributes read through the map (the lanes per direction): that needs (a) - (c). It is the base of every arm (the gradient test of E391).
**The fall-back** of a gated arm is the no-map head z's own read (lanes from perception alone) and 'unassigned' for every token: the planner receives the perception-only lanes and no map data.

### 1.3 The clause, pinned before any run (the FORM; the numbers are written in v1 from step 0)
- **The scored metric:** `lanes_all` = the share of delivered labelled road tokens whose lane class (lanes per direction) is read right, against the TRUE class of the token's segment whether or not that segment is in the map (so a missing segment or a wrong map costs); VAL, the scene bootstrap B 1000, the same draws for every arm. Reported beside it: assoc_acc, wrong_assign (the wrong-overwrite share of D135), assign_rate, route_f1, the gate's trust rate.
- **The deciding levels (one each):** WRONG = the map of another scene (`wrong_scene`) on top of the realistic errors; RIGHT = E391's `dec` (pose 1 m, counts 0.1, links 0.1, route hide 0.3, segments missing 0.1). Report-only: every catalogue level of section 1.1 (the 'failure map': where each arm falls below z).
- **Clause W (wrong map):** `lanes_all(S, WRONG) - lanes_all(z) >= -margin` on BOTH seeds; margin = max(0.03, the spread of z over the two seeds). 0.03 is E389's wrong-map clause; justification: it is about a third of the right-map gain over z that E391 measured on VAL (0.948 / 0.952 against 0.891 / 0.873: +0.057 / +0.079), the size at which a planner would notice the harm; the bar is a point estimate with the paired interval printed (as E389).
- **Clause R (the right map, the gain stays):** `lanes_all(S, RIGHT) >= lanes_all(a, RIGHT) - 0.01` (not below the ungated F1 by more than the band) AND `lanes_all(S, RIGHT) - lanes_all(z) >= +0.03`, both seeds; and **clause F:** the gate trusts at least 95 % of the frames of the RIGHT level (a gate that always falls back passes W and fails here).
- **Outcomes (in this order):** NOT RUN; INCONCLUSIVE (a clause holds on one seed only, or a minimum count is missed: 20000 labelled road tokens, 150 frames per mode of the failure map); **TRUST HOLDS** (W, R, F pass for the chosen arm on both seeds); **SAFE, NO GAIN** (W passes, R or F fails: the map is ignored, which is safe but loses the gain); **NOT TRUSTED** (W fails on a seed). The choice among arms that pass: the simplest (a < g < u < c, fewest parameters), unless a more complex one is better by more than the band on the mean of the gain at RIGHT and the harm at WRONG with the paired interval above 0 (RQ simplicity, D106).
- **The looks:** a TRAIN-holdout read first (every 10th TRAIN scene, left out of every training and every gate fit); the choice is made on the holdout; VAL is read ONCE for the verdict.

## 2. CPU step 0 (before any training; runs `E394.step0_<part>_s<S>`)
`maperror_probe.py degrade`: for every mode and level of the catalogue on an otherwise exact map (and the two deciding levels), the three heads on the same draws: assoc_acc, wrong_assign, assign_rate, lanes_all, the false-token and missing-segment parts. It gives (1) the failure map of the heads trained on simple errors (which new errors break them), (2) the nulls (z: perception alone; the oracle read), (3) the headroom: how much of the right-map gain a perfect gate could keep, (4) the numbers that the margin and the bars are derived from.

## 3. Read-out log (step 0, version 1 follows)
(pending)
