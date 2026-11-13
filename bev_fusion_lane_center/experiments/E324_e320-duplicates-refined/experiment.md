# E324 - E320 refined (D87 / D88): the duplicate excess of the attached-end recipe; arm s = the E320 recipe with the more selective pair stage of E316 (no new code)

Card of 2026-11-13 21:15, version 3 after the second critic pass (REVISE: arm s may be scheduled once the predictions are consistent and E316 is scored with the same read-out; arm n not ready): the connector-partner share is corrected (the first reading double-counted the connector-connector points: it is a minority, not a majority), the three-parts statement is limited to seed 0, arm s gets one derived prediction with a hit rule against the same-read-out reference (E316 snap-nogeom), arm n is reduced to a stub with a stop rule because the corrected diagnosis does not support it as the first fix. Follow-up of E320 ('E2E-required, kept and refined', D87); only differentiable solutions (D88). **Status: diagnosis done (CPU, `evaluation/jdup.py`); arm s needs no new code; nothing started.** E320's written token stays CUT (guard dup_frac +0.0184 / +0.0093, `experiments/E320_soft-attach-replaces-snap/experiment.md` section 3); E323 showed that it is not the number of curves.

## 0. The diagnosis (VAL 873 frames, p >= 0.1, B's delivered graph with the hard step; `jdup.py score --seed <S> --l-run e316_p_s<S>`, reproduces the dup_frac of the cards)
**dup_frac by read-out** (c = E312 snapped; u' = E312 snap-nogeom, the same curves as c; k = E320 own keypoints snapped; kn = E320 own keypoints snap-nogeom; a = E320 PRIMARY = attached ends + snap-nogeom; l = E316 snapped; l' = E316 snap-nogeom, scored 2026-11-13 21:14 as the same-read-out reference):

| | c | u' | k | kn | a | l (E316) | l' (E316 nogeom) |
|---|---|---|---|---|---|---|---|
| seed 0 dup_frac | 0.1747 | 0.1809 | 0.1834 | 0.1873 | 0.1931 | 0.1678 | 0.1744 |
| seed 1 dup_frac | 0.1557 | 0.1610 | 0.1589 | 0.1640 | 0.1651 | 0.1532 | 0.1585 |
| seed 0 / 1 topo F1 | 0.6825 / 0.6864 | 0.6706 / 0.6760 | | | 0.6803 / 0.6889 | 0.6898 / 0.6938 | 0.6782 / 0.6856 |
| a - c as steps, seed 0 | | read-out choice u' - c **+0.0062** (34 %) | | recipe kn - u' **+0.0064** (35 %) | attached ends a - kn **+0.0058** (32 %) | | |
| a - c as steps, seed 1 | | **+0.0053** (56 %) | | **+0.0030** (32 %) | **+0.0011** (12 %) | | |

The split of the guard excess into read-out choice, recipe and attached ends is three comparable parts on seed 0 and mostly the read-out choice on seed 1 (the split is path-dependent: via the own keypoints k the recipe is +0.0087 / +0.0032). The PRIMARY read-out of E320 is wired like u' (snap-nogeom), the guard of its rule compares with c (snapped with the snap's geometry): the read-out share exists for E312 itself (u' alone uses 0.0062 / 0.0053 of the 0.01 margin, as the E320 card said).
**Attribution of the duplicate points** (seed 0 c / a; seed 1 c / a): CC (connector point with no B partner) 23584 / 28007; 26584 / 25134; CB (with a B partner) 60510 / 76337; 58351 / 67570; BC (B point near a connector) 32180 / 43469; 31036 / 36970; BB 130126 / 130083; 95643 / 94518. **Connector duplicate points that have a connector partner** (CC + the CB points that also have one): c 36650 / 39865, a 46463 / 39356, i.e. **43.6 % / 46.9 % (c), 44.5 % / 42.5 % (a), 37.8 % / 38.8 % (E316) of the connector duplicate points CC + CB: a minority.** (The first version of this card divided (CC + this key) by (CC + CB), a double count, and called it a majority; the critic found it.) Of the excess a - c this key explains +9813 of +31496 duplicate points on seed 0 (31 %) and -509 on seed 1: on seed 1 the excess sits in connector-over-B-section overlaps (CB without a connector partner +8278, BC +5934). By arc length (CB, 4 - 8 / 8 - 16 / 16+ m) seed 0 c 32336 / 22253 / 5921, a 38499 / 29849 / 7989; partner kind section / link c 39123 / 21387, a 51281 / 25056.
**True and false curves** (both ends match the two ends of one GT pair, strict one-to-one, 4 m / 30 degrees; "false" mixes wrong pairs, unmatched or duplicate keypoints and unannotated lanes): seed 0 c 5965 true / 5413 false, a 5817 / 6201, **E316 6190 / 4697**; seed 1 c 5958 / 5891, a 5965 / 6243, **E316 6022 / 4672**. Duplicate points of FALSE connectors: c 47838 / 52630, a 67858 / 58795, E316 40610 / 37620; per false curve c 8.8 / 8.9, a 10.9 / 9.4, E316 8.6 / 8.1 (the count of false curves explains 35 % / 51 % of the false-connector excess, the overlap per false curve the rest, as E323 found). Curve length per curve is unchanged. Pair precision at E312's recall (`jpair_pr`): E320 -0.031 / +0.003, E316 +0.041 / +0.018.

**What it says.** (1) The guard excess of E320 is partly the read-out choice (not E320's), partly the recipe (its false curves: +788 / +352), partly the attached ends (seed 0). (2) E316 (the pair stage twice as long) moves the false-curve count and the duplicates the other way (-716 / -1219 false curves, dup -0.0065 / -0.0025 in the same nogeom read-out): selectivity is a lever for the false-curve share. (3) Connector-over-connector overlaps are a minority of the connector duplicate points and not the carrier of the excess on seed 1; connector-over-B-piece overlaps are. (4) Not tested: the share of 'false' that are legitimate unannotated lanes; connector-versus-B-piece overlaps of TRUE pairs.

## 1. The card (what the tools run)
```json card
{
 "id": "E324",
 "card": "",
 "title": "the E320 recipe (attached ends, geometry loss) with the pair stage at 6000 steps (the lever of E316), primary read-out `--attach --snap --snap-nogeom` at p >= 0.1, against E312 of the same seed and E316 read out the same way",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.0088, "vs": "c"},
 "better": {"dup_frac": "lower", "vert100_dp": "lower"},
 "arms": {
  "c": {"recipe": "control = E312 snapped: topo F1 0.6825 / 0.6864, dup_frac 0.1747 / 0.1557, junction F1 0.3194 / 0.3143, F1@1.0 0.8669 / 0.8685, lanes found 0.7376 / 0.7423, vertices 19.55 / 20.10"},
  "u": {"recipe": "floor u' = E312 with the snap's wiring and the keypoint geometry (`--snap --snap-nogeom`): topo F1 0.6706 / 0.6760, dup_frac 0.1809 / 0.1610"},
  "l": {"recipe": "same-read-out equal-budget reference of arm s (not a run of this card): E316 read out `--snap --snap-nogeom`: topo F1 0.6782 / 0.6856, dup_frac 0.1744 / 0.1585, junction F1 0.3053 / 0.3068, F1@1.0 0.8700 / 0.8723 (E316 snapped: 0.6898 / 0.6938, dup 0.1678 / 0.1532, false curves 4697 / 4672)"},
  "s": {"recipe": "the E320 recipe unchanged except `--steps 6000`: `jends_train.py --e2e --scales --maps-w 1.0 --steps 6000 --geom-w 0.5 --attach` on `runs/e307_n_s<S>/model.pt`, BFLC_SIG=1.0 (run e324_s_s<S>); no new code; primary read-out `jends_eval.py score --attach --snap --snap-nogeom --thr 0.1`", "run_ids": ["E324.s_s0", "E324.s_s1"]}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "u", "min_effect": 0.005},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 6.8,
 "kill": "the loss is not finite; or the snapped topo F1 of the PRIMARY read-out of the arm is below the floor u' on either seed (the attached ends make the keypoint geometry worse than the rule's: the idea fails). The maps keypoint F1 against E307 is REPORTED, not a kill (E320 -0.0457 / -0.0317, E316 costs about -0.010 / -0.004 on top: a 6000-step arm is predicted at about -0.056 on seed 0); a value below -0.08 on either seed is flagged in the summary",
 "evaluate": {"split": "val"}
}
```
**Verdict rule (the rule of E320 with its CUT branch replaced by D87; by hand: replay for the scores of stored dumps, the training runs go through run.py; `decide.py` reads only the arm s):** KEEP when, on BOTH seeds, topo F1 of the PRIMARY read-out is at least u' + 0.005 and every guard holds (the dup guard against c is the written one; it is also read against u' and against E316 nogeom). 'Ties the snap' means a - c >= -0.01. TIE, kept and refined (D87): a bound or a guard is missed but the kill did not fire. CUT only by the kill. **The rule cannot test 'the excess is removed'** (the topo clause is passed by E316's training length alone: s is predicted +0.012 / +0.017 above the bound), so the test of the mechanism is the pre-registered excess below.

## 2. Card text
- **Statement (falsifiable):** a more selective pair stage (the lever of E316) applied to the E320 recipe lowers the duplicate excess of the PRIMARY read-out beyond what the longer training does for the keypoint read-out, i.e. the excess of the recipe interacts with the training length instead of adding to it.
- **Pre-registered numbers (one hypothesis pair, a hit is within +-0.003 for topo F1 and dup_frac, +-0.01 for junction F1, per seed):** the additive guess (no interaction): a(s) = a + (l' - u') in every quantity, i.e. the effect of E316's training length in the same nogeom read-out is added to E320 as measured: **dup_frac 0.1866 / 0.1626** (s0 +0.0119 / s1 +0.0069 over c: the dup guard on seed 0 breaks marginally, a coin flip within retrain noise that is unknown for dup_frac), **topo F1 0.6879 / 0.6985** (+0.0054 / +0.0121 over c; the JSON value 0.0088 is the mean of the two), junction F1 0.2936 / 0.3109 (-0.0258 / -0.0034 against c: seed 0 is 0.004 from its guard), F1@1.0 0.8760 / 0.8780, vertices -0.3 / -0.4. The selectivity hypothesis (interaction): the excess of s over E316 nogeom, dup(s) - dup(l'), is smaller than the additive value a - u' = **+0.0122 / +0.0040**; **H_add if dup(s) - dup(l') is within +-0.003 of those values, H_sel if it is at most +0.006 / +0.002, inconclusive between.** 'Half of the excess falls' was a guess in the first version and is dropped; the false-curve count of s is read against E320's 6201 / 6243 and E316's 4697 / 4672 (no number is pre-registered: a value between the two and nearer E316 supports selectivity).
- **Why E316 nogeom is the reference:** E316 snapped is read out with the snap's geometry; the PRIMARY read-out of the arm is wired like u'. The same-read-out reference removes the read-out share (+0.0062 / +0.0053) from the comparison. The additive prediction is the null: it leaves the excess over the reference exactly where E320 at 3000 steps has it.
- **Controls and confounds:** c, u' and l' as stored; E320 as measured (a) is the reference of the defect; one training per arm and seed: the scene bootstrap is the interval, the retrain noise of dup_frac is unknown (nothing under 0.004 is claimed); the read-out choice is held fixed. The geometry loss in the recipe is E317 code that E320 already ran; E317 itself is not re-launched (the holds on E317 / E318 / E319 stay).
- **Arm n, a STUB, not built (the layer the lane lead proposed: a connector-level Matrix-NMS on the pair probabilities):** the corrected diagnosis does not support it as the first fix: connector-over-connector overlaps are 43 - 47 % of the connector duplicate points and explain 31 % of the excess on seed 0 and none on seed 1, where the excess sits in connector-over-B-section overlaps that a connector-level layer cannot reach (the delivered pieces of B do not exist inside the pair head's training loop). If it is ever built the spec must use the Hermite curve with scales (`_bezier_s`, `jends.py`), point-to-segment distances (`_dist_to_polyline`, not nearest sample points at 1.8 m spacing against 0.75 m), O = 0 with finite gradient for curves under 8 m, a decay capped at 1, real peaks only (no injected GT keypoints), pairs that share a keypoint index excluded, the share of frames with more than 48 candidates reported, `jends_eval.py dump` building the layer from the flag, and E277's lessons cited (the soft damping lost lanes on one seed: the loss fights the GT positives); a CUT / stop rule: dup n - a above -0.004 on a seed or topo F1 loss above 0.005; a flag-off retrain as the noise twin; a hit window that excludes effects under 0.004. Its prediction of the first version (-0.008 / -0.004) is withdrawn: it was not derived.
- **Price (reported next to it):** at inference nothing changes against E320 (9.6 k parameters, 3.9 MMACs of `Attach`); training 3.2 h per seed. **Budget:** 6.4 slot-hours for the two seeds (both slots, 3.2 h) plus scoring: 6.8.
- **Named hard steps (unchanged from E320):** the argmin of the wiring at the read-out, the candidate set of the attachment, the keypoint peak choice, p >= 0.1, the close-node step (E292).
- **Not claimed:** anything on TEST (D84 covers only E312); that the attached ends are the cause on seed 1.

## 3. Scheduling
`temp/e324/start_pairs.sh <seed>` (the name the chains call; the run-name prefix `e324_s_s` in the first line of `temp/e324/READY`; the marker `temp/e324/ACCEPTED_s` is written when this version has been read as accepted), the post chain inside `temp/day14/day14.sh` (`post_e324`: dump, PRIMARY / own-keypoint snapped / nogeom scores, `jpair_pr` against E312 and E316, `jdiag kperr`, the maps (reported), bootstraps against c and against E316, then `jdup.py score --a-run e324_s_s<S> --l-run e316_p_s<S>`). Both seeds need both slots for 3.2 h; starts 22:00 - 05:00 (a start after 03:50 ends after 07:00); the earliest clean start is tonight or a weekend night.

## 4. Referee
First pass 21:35 (REVISE, G2 no), second pass 21:50 (REVISE: arm s may be scheduled once the predictions are consistent and E316 nogeom is scored; arm n not ready; the 71 - 78 % was a double count). Version 3 answers both; the points on arm n are moved to the stub.
