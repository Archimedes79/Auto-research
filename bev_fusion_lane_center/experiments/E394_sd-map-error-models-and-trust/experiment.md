# E394 - STEP 3 of the SD-map line: the ERROR MODELS of the map and a TRUST mechanism, so that a wrong map does not make the lanes per direction worse than no map

**CLOSED AT THE TRAIN DESIGN (v1.1, 2026-11-24; run `E394.gdesign_s0`): arm g, the label-free gate on E391's F1, cannot meet clauses W and F together at the card's deciding levels on 400 TRAIN frames outside its fit, and neither can any logistic on its four features (section 3.3): wherever W holds, F fails, and the reverse. Its holdout and VAL reads were NOT run (they would spend the looks to confirm a design that cannot pass); nothing adopted; the next step is v2 (section 1.4) on the head that E391b part B leaves.** v1.1 also takes in a fresh critic's REVISE on v1 (3 blocking, 7 should-fix; section 5). **VERSION 1 (2026-11-24, D145: the owner's order to run all experiments, which lifts the D138 pause and is the go for this card; written by the token-code chat, which owns the E390 / E391 / E394 cards).** Step 0 ran (CPU, `E394.step0_a_s0`, the lane chat, section 3); the bars below are derived from it. **v1 measures the simplest trust mechanism first: arm g, a LABEL-FREE consistency gate on E391's fusion head F1 (CPU only, no training, `evaluation/maptrust_gate.py`); the retrained arms u / c / ch of v0 are deferred to a v2 (section 1.4).** Two reasons: (1) D106 (the simplest arm first; the card's own choice rule a < g < u < c), (2) E391's F1 was fitted on a token population with 68.5 % false tokens against 23.5 - 33 % in the delivered set (the caveat at the head of the E391 card, found by the lane chat while writing E391b): E391b part B (F1d, the loss on the delivered tokens only) may change the base head, and a retrained arm built on F1 now would have to be built again. **Every number of v1 is for E391's F1 (`runs/e391_F1_s<S>`); if E391b part B adopts F1d, the gate is fitted and read again for F1d under the same rule (run ids `E394.*d_*`).** The runs wait for a fresh critic pass of this version. v0 (2026-11-23, the report chat on the lead's task, D137): the catalogue, the trust candidates and the FORM of the clause, kept below. The order of the lead stands: catalogue (1.1), candidates (1.2), clause before any run (1.3), CPU step 0 first (section 2), a fresh critic before the runs; no TEST, no push. (v0 header: E391's F1 read lanes_all 0.52 at a wrong scene against z 0.88 on a 22-frame smoke; the base = the step-1 association head (E390 g7 weights) and E391's F1; the perception untouched, D135, stop-gradient; the new code in `training/data/lane_maperror_data.py`, `evaluation/maperror_probe.py` and, v1, `evaluation/maptrust_gate.py`.)

**Order (the lead's):** (1) the catalogue of map errors (section 1.1), (2) the trust candidates (section 1.2), (3) the clause pinned before any run (section 1.3), (4) CPU STEP 0 first (section 2: how each error degrades the heads, the nulls and the headroom; the bars are derived from it in version 1, never typed), (5) a fresh `ar-critic` on the card before any GPU run, then the runs. No TEST, no push.

## 0. The card (v1: the numbers from step 0 and the TRAIN design, section 3)
```json card
{
 "id": "E394",
 "card": "",
 "title": "E394 step 3 (v1): the error models of the SD map (the catalogue of v0) and the simplest trust mechanism, a label-free map-perception consistency gate g on E391's fusion head F1 (a logistic on four per-frame features, fitted on TRAIN draws of the catalogue mix, threshold chosen on TRAIN; a distrusted frame falls back to the no-map head z and leaves every token unassigned), so that with a WRONG map the lanes per direction are not worse than z by more than 0.03 and with the RIGHT map the gain stays; CPU only, two seeds, the TRAIN holdout first, VAL once; the retrained arms u / c / ch deferred to v2",
 "kind": "block",
 "deterministic": true,
 "metric": "lanes_all",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": -0.047, "vs": "z", "note": "v1.1, from the TRAIN design read at the deciding levels (3.3) at the threshold 0.96: lanes_all(g, WRONG) - z -0.047 with trust at RIGHT 0.925, so W and F fail; no threshold meets both: the arm is closed there, not run"},
 "design_delta": [{"row": "R9", "change": "after a verdict: 'E394 step 3: <outcome> (the label-free gate on F1: the wrong-map harm against no-map, the gain kept at the right map, the trust rate)'; open to attack when -> the fusion of the map into the delivered graph (the perception is untouched by design)"}],
 "arms": {
  "step0": {"recipe": "DONE (section 3): CPU, no training: every mode of the catalogue at each level of its grid on an otherwise exact map plus the two deciding levels, read with the zero-shot E390 head (p6zero), E391's F1 and the no-map head z on the TRAIN holdout, every 2nd frame (434 frames), seed 0: the failure map, the nulls, the headroom; part b was not needed (the deciding levels are inside 'all')", "run_ids": ["E394.step0_a_s0"]},
  "a": {"recipe": "(a) E391's F1 as trained (simple mix), ungated: the association's own 'unassigned' slot; read by `maptrust_gate.py read` on the same draws as g (no run of its own)", "run_ids": []},
  "z": {"recipe": "the no-map head of E391 (`runs/e391_z_s<S>`): perception alone, the control of every clause; read by `maptrust_gate.py read` on the same draws (no run of its own)", "run_ids": []},
  "g": {"recipe": "(b) the label-free consistency gate on F1 (`evaluation/maptrust_gate.py`): per frame assigned_mass, max_prob (F1's association), p6_rate (the zero-shot E390 association's assign share: the geometric agreement), lane_agree (z's lane class against the map's on F1's assigned road tokens); a logistic fitted on 800 TRAIN frames outside the holdout, each with its own draw of the catalogue mix, label = the map of another scene; threshold = the 0.96 quantile of p(wrong) over the other draws (section 3.2); a distrusted frame: lanes from z, every token unassigned. v1.1: NOT RUN - closed on the TRAIN design read (section 3.3); its run ids stay reserved: per seed S `fit`, `read --split holdout`, `read --split val`. v1.1: the design read of 3.3 runs through run.py as `E394.gdesign_s0` (TRAIN only, the closure's record)", "run_ids": ["E394.gdesign_s0"]},
  "u": {"recipe": "DEFERRED to v2 (section 1.4): F1 retrained on the catalogue mix", "run_ids": []},
  "c": {"recipe": "DEFERRED to v2 (section 1.4): the learned gate on the frozen u, and its held-out-mode arm ch", "run_ids": []}
 },
 "control": "z",
 "variant": "g",
 "seeds": [0, 1],
 "rule": [{"metric": "lanes_all", "vs": "z", "max_loss": 0.03, "note": "by hand: section 1.3 (v1.1) clauses W, R, F at the deciding levels on VAL, both seeds (W: g at WRONG - z >= -margin, margin max(0.03, the z spread of the two seeds); R: g at RIGHT >= a at RIGHT - 0.01; F: trust at RIGHT >= 0.95; a - z at RIGHT reported as a property of the base head); decide.py cannot read it"}],
 "budget_gpu_h": 0.0,
 "kill": "the gate's fit refuses (no wrong or no right maps among the draws); the heads are not runs/e391_F1_s<S> / runs/e391_z_s<S> and the encoder not runs/e390_g7_s<S> (the fit and read json record all three, v1.1); a read json that exists is never overwritten (the tool refuses); a missed minimum count is not a kill but INCONCLUSIVE (1.3)",
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

### 1.3 The clause, pinned before any run (v0: the FORM; v1: the numbers, below)
- **The scored metric:** `lanes_all` = the share of delivered labelled road tokens whose lane class (lanes per direction) is read right, against the TRUE class of the token's segment whether or not that segment is in the map (so a missing segment or a wrong map costs); VAL, the scene bootstrap B 1000, the same draws for every arm. Reported beside it: assoc_acc, wrong_assign (the wrong-overwrite share of D135), assign_rate, route_f1, the gate's trust rate.
- **The deciding levels (one each):** WRONG = the map of another scene (`wrong_scene`) on top of the realistic errors; RIGHT = E391's `dec` (pose 1 m, counts 0.1, links 0.1, route hide 0.3, segments missing 0.1). Report-only: every catalogue level of section 1.1 (the 'failure map': where each arm falls below z).
- **Clause W (wrong map):** `lanes_all(S, WRONG) - lanes_all(z) >= -margin` on BOTH seeds; margin = max(0.03, the spread of z over the two seeds). 0.03 is E389's wrong-map clause; justification: it is about a third of the right-map gain over z that E391 measured on VAL (0.948 / 0.952 against 0.891 / 0.873: +0.057 / +0.079), the size at which a planner would notice the harm; the bar is a point estimate with the paired interval printed (as E389).
- **Clause R (the right map, the gain stays):** `lanes_all(S, RIGHT) >= lanes_all(a, RIGHT) - 0.01` (not below the ungated F1 by more than the band) AND `lanes_all(S, RIGHT) - lanes_all(z) >= +0.03`, both seeds; and **clause F:** the gate trusts at least 95 % of the frames of the RIGHT level (a gate that always falls back passes W and fails here).
- **Outcomes (in this order):** NOT RUN; INCONCLUSIVE (a clause holds on one seed only, or a minimum count is missed: 20000 labelled road tokens, 150 frames per mode of the failure map); **TRUST HOLDS** (W, R, F pass for the chosen arm on both seeds); **SAFE, NO GAIN** (W passes, R or F fails: the map is ignored, which is safe but loses the gain); **NOT TRUSTED** (W fails on a seed). The choice among arms that pass: the simplest (a < g < u < c, fewest parameters), unless a more complex one is better by more than the band on the mean of the gain at RIGHT and the harm at WRONG with the paired interval above 0 (RQ simplicity, D106).
- **The looks:** a TRAIN-holdout read first (every 10th TRAIN scene, left out of every training and every gate fit); the choice is made on the holdout; VAL is read ONCE for the verdict.

- **v1, the numbers (from step 0, section 3, and the TRAIN design, section 3.2; never typed by hand):** the deciding levels as v0: WRONG = `wrong_scene+dec`, RIGHT = `dec`. **Clause W:** lanes_all(g, WRONG) - lanes_all(z) >= -margin, margin = max(0.03, |z_s0 - z_s1| on VAL), on both seeds (step 0: the ungated F1 is at -0.3571 on the holdout, so W is the whole question; the paired interval is printed beside the point estimate, as E389). **Clause R (v1.1, critic blocking 3):** lanes_all(g, RIGHT) >= lanes_all(a, RIGHT) - 0.01 (the gate keeps what the base head gains). v0's second part, g - z >= +0.03, is dropped: E391's +0.057 / +0.079 are `lanes_acc` (the tokens of PRESENT segments), not `lanes_all`; in `lanes_all` the base head itself is only +0.029 above z at RIGHT on the holdout (step 0; `lanes_acc` there +0.059, matching E391) and +0.022 on held TRAIN frames (3.3), because F1 reads the tokens of missing segments worse than z. So a - z at RIGHT is reported as a property of the base head (its own label: 'the base head gains < 0.03 in lanes_all'), not a clause of the gate. **W's justification (v1.1):** in `lanes_all` the margin 0.03 is about the whole right-map gain of the base head, not a third of it: a gate that loses 0.03 at a wrong map gives back what the map brings at a right one. **Clause F:** the gate trusts >= 95 % of the frames at RIGHT. **The outcomes, in this order (v1.1, critic should-fix 5: one reading):** NOT RUN; INCONCLUSIVE = a minimum count missed on a seed, or W passes on one seed and fails on the other, or R or F differs between the seeds while W passes on both; NOT TRUSTED = W fails on both seeds; TRUST HOLDS = W, R and F pass on both seeds; SAFE, NO GAIN = W passes on both seeds and R or F fails on both. With one candidate arm (g) there is no choice among arms: the holdout read is the first look (reported, never changes the threshold or the features), VAL is read once for the verdict.
- **Report-only (the failure map of g):** `exact`, `wrong_scene=1`, `shift=4.0`, `shift=8.0`, `yaw=4.0`, `pose=4.0`, `outdated=0.3`, `outdated=1.0`, `count_big=0.3`, `ghost=0.6`, `wrong_route=1`, `node_drop=0.4` (`maptrust_gate.CARD_LEVELS`): where g falls below z, with the trust rate.

### 1.4 What v1 leaves out (named)
- **u, c, ch (the retrained arms of v0):** deferred to a v2, which is written only if g is NOT TRUSTED or SAFE, NO GAIN, and only after E391b part B has decided the base head (F1 or F1d): they would retrain the association, and their cost (about 0.6 GPU h) and the training-mix design belong on the head that stays.
- **A gate for the outdated map and gross count errors:** step 0 shows they break F1's lanes read (outdated 1.0: -0.3621 vs z; count_big 0.3: -0.1405) while the label of g is the map of another scene; g's read of those levels is reported, not judged (the catalogue marks them stress levels without a real map to calibrate against).

## 2. CPU step 0 (before any training; runs `E394.step0_<part>_s<S>`) - DONE
`maperror_probe.py degrade`: for every mode and level of the catalogue on an otherwise exact map (and the two deciding levels), the three heads on the same draws: assoc_acc, wrong_assign, assign_rate, lanes_all, the false-token and missing-segment parts. It gives (1) the failure map of the heads trained on simple errors (which new errors break them), (2) the nulls (z: perception alone; the oracle read), (3) the headroom: how much of the right-map gain a perfect gate could keep, (4) the numbers that the margin and the bars are derived from. Ran 2026-11-24 as `E394.step0_a_s0` (the lane chat; run.py `--no-card` diagnostic, 1949 s, CPU, code tag `lcr/E394.step0_a_s0` d4969ac0bc); part b was not needed.

## 3. Read-out log
### 3.1 Step 0 (TRAIN holdout, seed 0, every 2nd frame = 434 frames; `runs/e394_step0/degrade_holdout_s0_a.json`; no bootstrap, one seed: the failure map, not a verdict)
| level | z lanes_all | F1 lanes_all (F1 - z) | p6zero lanes_all | F1 assoc_acc | F1 wrong_assign |
|---|---|---|---|---|---|
| exact | 0.902 | 0.987 (+0.0856) | 0.933 | 0.849 | 0.070 |
| dec (RIGHT) | 0.902 | 0.931 (+0.0294) | 0.815 | 0.846 | 0.069 |
| wrong_scene=1 | 0.902 | 0.541 (-0.3601) | 0.406 | 0.665 | 0.335 |
| wrong_scene+dec (WRONG) | 0.902 | 0.544 (-0.3571) | 0.407 | 0.667 | 0.333 |
| shift=4.0 / 8.0 | 0.902 | 0.986 / 0.982 | 0.925 / 0.882 | 0.833 / 0.795 | 0.061 / 0.049 |
| yaw=4.0 | 0.902 | 0.986 | 0.933 | 0.840 | 0.064 |
| ghost=0.6 | 0.902 | 0.965 | 0.909 | 0.833 | 0.063 |
| wrong_route=1 | 0.902 | 0.985 | 0.933 | 0.842 | 0.064 |
| count_big=0.3 | 0.902 | 0.761 (-0.1405) | 0.658 | 0.841 | 0.073 |
| outdated=1.0 | 0.902 | 0.539 (-0.3621) | 0.009 | 0.831 | 0.082 |
**Read:** among the errors of the TRAIN mix only the map of another scene breaks F1's lanes read (-0.36 vs z); every geometric error up to 8 m / 4 degrees, ghosts, missing segments and a wrong route leave it above z; the stress levels outdated 1.0 and count_big 0.3 break it too (1.4). The headroom: a perfect gate keeps F1 at the right map and falls back to z at the wrong one (W = 0 exactly), so the question is how well a label-free gate tells the map of another scene from a right one.

### 3.2 The design of g on TRAIN, IN-SAMPLE (v1.1: the logistic and its threshold were fitted AND scored on the same 800 draws; before any read of the holdout or VAL; scripts `scripts/e394/e394_explore.py`, `scripts/e394/e394_q.py`, 800 TRAIN fit draws of seed 0, the fit frames of `maptrust_gate.py`)
- (v1.1: the AUROC values 0.79 / 0.915 / 0.908 and the 44 % below are the printed output of `e394_explore.py`, not stored; rerunning the script reproduces them.)
- **The label:** `is_wrong` of the catalogue (the map of another scene, a shift or turn of 4 - 8 m / deg, a route on another path) gave AUROC 0.79 and rejected 44 % of the wrong maps at 97.5 % trust: the shifted and turned maps look partly right AND do not harm the lanes read (3.1). The label became the map of another scene: AUROC 0.933.
- **The features:** the four of `maptrust_gate.FEATS`; adding the zero-shot association's mean mass and best probability changed nothing (0.933), a harm label (F1 worse than z on the frame) was worse (AUROC 0.915), p6 alone 0.908.
- **The threshold** (the quantile of p(wrong) over the draws that are not another scene's map; on the fit draws: the gated head's lanes_all on the wrong-scene draws minus z's, and the trust on the realistic draws - exact or the simple mix, n 321):

| quantile | trust, realistic draws | rejects another scene's map | g - z on those draws | g / a / z on the other draws |
|---|---|---|---|---|
| 0.90 | 0.925 | 0.824 | -0.025 | 0.962 / 0.958 / 0.911 |
| 0.95 | 0.956 | 0.794 | -0.029 | 0.959 / 0.958 / 0.911 |
| 0.96 | 0.963 | 0.779 | -0.029 | 0.959 / 0.958 / 0.911 |
| 0.975 | 0.978 | 0.735 | -0.043 | 0.959 / 0.958 / 0.911 |

**Chosen: 0.96** (`maptrust_gate.TRUST_Q`), the quantile with the most room on both W (-0.03) and F (0.95) on TRAIN: both clauses sit at their edge (v1: outcome odds TRUST HOLDS 35 %, NOT TRUSTED 40 %, INCONCLUSIVE 20 %, SAFE, NO GAIN 5 %; superseded by 3.3). The gated head keeps the gain on the other draws (0.959 against F1's 0.958: distrusting a few harmful moderate-error frames costs nothing).

### 3.3 The design read at the deciding levels (v1.1, the critic's blocking 2 and should-fix 4; TRAIN only; `scripts/e394/e394_design.py`; run through run.py as **`E394.gdesign_s0`** (2026-11-24, 173 s, CPU, code tag `lcr/E394.gdesign_s0` e11899b262; `runs/e394_gate/design_train_s0_rerun.json`, identical to the first read `design_train_s0.json` in every number: the read is deterministic))
The trust and the harm of g measured where the clauses are judged: 400 TRAIN frames outside the holdout AND outside the 800 fit draws (frame-disjoint, not scene-disjoint: they share scenes with the fit draws, which biases toward the fit and makes the closure conservative), each read at exactly `dec` (RIGHT) and `wrong_scene+dec` (WRONG), with the logistic fitted on the 800 mix draws (the wrong-scene maps are borrowed through `plan_for('holdout')`, so a borrowed map can be a holdout scene's: a map input, no label). Base on these frames: lanes_all at RIGHT a 0.929, z 0.907 (a - z +0.022); at WRONG a 0.477 (a - z -0.430).

| threshold quantile | trust at RIGHT (F: >= 0.95) | g - z at WRONG (W: >= -0.03) | rejects at WRONG | g - a at RIGHT (R: >= -0.01) |
|---|---|---|---|---|
| 0.90 | 0.840 | -0.0212 | 0.853 | +0.0089 |
| 0.925 | 0.870 | -0.0310 | 0.833 | +0.0091 |
| 0.95 | 0.910 | -0.0443 | 0.785 | +0.0085 |
| 0.96 (v1's choice) | 0.925 | -0.0467 | 0.772 | +0.0070 |
| 0.975 | 0.938 | -0.0562 | 0.748 | +0.0077 |
| 0.99 | 0.978 | -0.1267 | 0.597 | +0.0062 |
| 0.995 | 0.993 | -0.2129 | 0.432 | +0.0019 |
| 1.0 (never distrust = a) | 1.000 | -0.4282 | 0.002 | 0.0000 |

The fit itself, cross-validated (5 folds on the 800 mix draws) at 0.96: g - z on the wrong-scene draws -0.043, rejects 0.772 (in-sample -0.029 / 0.779): the 3.2 numbers were optimistic.
**The feature-set bound (the critic's second look, should-fix 2):** a logistic on the same four features fitted DIRECTLY on the 400 held frames at RIGHT (label 0) and WRONG (label 1), in-sample (the most a frame-level logistic on these features can do): AUROC 0.914; at a trust of 0.95 at RIGHT, g - z at WRONG -0.0535 (q 0.90: trust 0.900, -0.0189; q 0.99: trust 0.990, -0.1286). Even this bound misses W at F's trust, so the claim holds for the FEATURES, not only for the mix-fitted logistic.
**Read (by hand: the clauses are written, the arm is not run; run `E394.gdesign_s0`):** no threshold of this gate meets W and F together at the deciding levels - W needs a quantile of at most about 0.925, where the trust at RIGHT is 0.87; F needs about 0.99, where the harm at WRONG is -0.13. The reason is visible in the data: the realistic right map `dec` (five errors at once) looks to these four features much like the map of another scene (trust falls with the number of active errors, the critic's table: 0.99 - 1.00 with 0 - 2 errors, 0.83 with 4 or more). R would hold at every threshold (g - a +0.002 to +0.009: distrusting a noisy right map costs nothing in lanes_all, since F1 reads missing segments worse than z), which is why F, not R, is the binding clause: F guards the association and route that a distrusted frame withholds from the planner. **Outcome: arm g CLOSED on TRAIN (design infeasible), no holdout or VAL read, nothing adopted; D135's 'a wrong map must not overwrite' stays unmet by the label-free route on F1.** What it means for the line: a frame-level, label-free gate on these features is not enough; the next arms (v2, 1.4) must either see more than four frame statistics (the token-level gate c, trained) or make the head itself robust (u, retrained on the catalogue mix) - on the head that E391b part B leaves.

## 4. The launch (v1; NOT RUN - closed on the TRAIN design read, 3.3; kept for the record and for a refit on F1d / v2)
**The texts of every run:**

    H="E394 v1 (D145): a label-free map-perception consistency gate g on E391's fusion head F1 keeps the lanes per direction within 0.03 of the no-map head z under the map of another scene and keeps F1's gain on the right map; CPU"
    P="lanes_all(g, WRONG) - z about -0.03 (window -0.05 .. -0.01); trust at RIGHT about 0.96 (card 3.2)"
    R="by hand, card section 1.3 (v1 numbers), written before the run: W g at WRONG - z >= -max(0.03, z seed spread), R g at RIGHT >= a - 0.01 and g - z >= +0.03, F trust at RIGHT >= 0.95, both seeds, VAL; NOT RUN / INCONCLUSIVE / TRUST HOLDS / SAFE, NO GAIN / NOT TRUSTED"

Per seed S in 0, 1 (`runs/e394_gate/fit_s<S>.json`, `read_holdout_s<S>.json`, `read_val_s<S>.json`):

    python3 ../autoresearch/tools/run.py --id E394.gfit_s<S> --seed <S> --retries 0 --require-card --split val --timeout 60 --hyp "$H" --pred "$P" --rule "$R" -- bash -c 'env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh maptrust_gate.py fit --seed <S>'
    python3 ../autoresearch/tools/run.py --id E394.gread_holdout_s<S> --seed <S> --retries 0 --require-card --split val --timeout 120 --after E394.gfit_s<S> --hyp "$H" --pred "$P" --rule "$R" -- bash -c 'env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh maptrust_gate.py read --seed <S> --split holdout'

The VAL read only after both holdout reads are written into section 3 (the first look is reported, it changes nothing):

    python3 ../autoresearch/tools/run.py --id E394.gread_val_s<S> --seed <S> --retries 0 --require-card --split val --timeout 120 --hyp "$H" --pred "$P" --rule "$R" -- bash -c 'env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh maptrust_gate.py read --seed <S> --split val'

Cost: the fit about 1.5 min per seed (measured on the scratch fit of 3.2); a read: the code path was smoked on 9 holdout frames at the non-deciding level `exact` (6 s for the level, loading included; a code check, its numbers are not used), and step 0 took 34 s per level at 434 frames with the same three heads: about 1 min per level at the full holdout / VAL, 14 levels, about 15 min per read, about 1.2 h of CPU for all six runs; no GPU. The `run.py --split val` of the holdout reads follows the E391 holdout reads (the run's split field; the tool reads the TRAIN holdout).

## 5. Referee
**v1, a fresh ar-critic (cards + the code of `maptrust_gate.py`, 2026-11-24): REVISE**, 3 blocking + 7 should-fix; checked and fine: the features are label-free (`kept` from the predicted `pnone`, the input map only), the bootstrap paired, the fall-back and g's wrong_assign right, W / R / F computable from `read`, the 3.2 table reproduced by `e394_q.py`. Taken in (v1.1):
- B1 the card block: a and z as arms (no run of their own), W as `max_loss 0.03`.
- B2 F and the predicted trust came from easier draws than `dec`: the design read at the exact deciding levels on held TRAIN frames (3.3) - it closes the arm.
- B3 R's +0.03 was a `lanes_acc` number: R is now g >= a - 0.01, a - z reported as the base head's property; W's justification rewritten.
- Should-fix: (4) 3.2 labelled in-sample, the cross-validated numbers in 3.3; (5) one reading of the outcome order, a missed count INCONCLUSIVE not a kill; (6) the fit and read json record F1, z and the encoder (repository-relative); (7) `combine()` factored out with a test of the fall-back, and `lane_agree` tested against z (4/6 -> 5/6 and 1/6); (8) the step-0 differences from the json (+0.0856, -0.3601, -0.3571, -0.3621, -0.1405); (9) the dangling '1.5' reference removed, the unstored explore numbers named; (10) the borrowed holdout maps of the fit named (3.3); the `--split val` of the reserved runs is the run's split field as in E391.
**Second look (v1.1): REVISE**, 1 blocking + 3 should-fix: the closure judged honest and the call right; taken in (v1.2): (B) hard rule 3 - the design read now ran through run.py as `E394.gdesign_s0` (ledger row with the by-hand closure, code tag) and reproduces the first read exactly; (2) the feature-set bound added (3.3: even fitted on the deciding levels the four features miss W at 95 % trust, -0.0535); (3) the 400 frames named frame-disjoint, not scene-disjoint; (4) `provenance()` takes the encoder folder from `sdgraph_probe.runs_dir`.
