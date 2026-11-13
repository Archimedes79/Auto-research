# E324 - E320 refined (D87): the duplicate defect of the attached-end recipe is a defect of FALSE pairs that run along delivered pieces; two fixes that stay end-to-end

Card of 2026-11-13 21:15, written by the lane chat after D87 (E320 is 'E2E-required, kept and refined'; this is its follow-up card, the first step of D87's order: CPU diagnosis first, then the fix). **Status: diagnosis done (CPU, `evaluation/jdup.py`, committed 7c15fc78), card only for the fix: no training code changed, nothing started.** E320's written token stays CUT (guard dup_frac +0.0184 / +0.0093, `experiments/E320_soft-attach-replaces-snap/experiment.md` section 3); E323 showed that it is not the number of curves; this card finds where the extra duplicates are and states two fixes that keep the system end-to-end differentiable at inference (both are training-only changes).

## 0. The diagnosis (VAL 873 frames, p >= 0.1, c = E312 snapped, a = E320 PRIMARY read-out; `jdup.py score --seed <S>`, reproduces the dup_frac of the cards: 0.1747 / 0.1931 and 0.1557 / 0.1651)
Duplicate points (dense points every 0.5 m that run parallel within 0.75 m of a piece that is neither neighbour nor sibling; `topo_metrics.dup_counts`), attributed (own piece type, partner type; C = connector, B = piece of B's delivered graph):

| | seed 0 c | seed 0 a | a - c | seed 1 c | seed 1 a | a - c |
|---|---|---|---|---|---|---|
| duplicate points / points | 246400 / 1410407 | 277896 / 1439491 | +31496 | 211614 / 1358948 | 224192 / 1358253 | +12578 |
| CC (connector near connector) | 23584 | 28007 | +4423 | 26584 | 25134 | -1450 |
| **CB (connector point near a B piece)** | 60510 | 76337 | **+15827** | 58351 | 67570 | **+9219** |
| **BC (B point near a connector)** | 32180 | 43469 | **+11289** | 31036 | 36970 | **+5934** |
| BB (B near B) | 130126 | 130083 | -43 | 95643 | 94518 | -1125 |
| curves delivered | 11378 | 12018 | +640 | 11849 | 12208 | +359 |
| curve points per curve | 42.6 | 42.6 | 0 | 43.2 | 42.5 | -0.7 |
| **true curves** (both ends match the two ends of one GT pair) | 5965 | 5817 | **-148** | 5958 | 5965 | +7 |
| **false curves** | 5413 | 6201 | **+788** | 5891 | 6243 | **+352** |
| duplicate points of TRUE curves | 36256 | 36486 | +230 | 32305 | 33909 | +1604 |
| duplicate points of FALSE curves | 47838 | 67858 | **+20020** | 52630 | 58795 | **+6165** |
| pair precision at E312's recall (`jpair_pr.py`) | | 0.712 vs 0.743 | -0.031 | | 0.727 vs 0.724 | +0.003 |
Share of the connector points that are duplicates, by pair probability (0.1 - 0.3 / 0.3 - 0.5 / 0.5 - 0.8 / 0.8+): seed 0 c 19.9 / 17.0 / 15.6 / 14.0 %, a 23.4 / 20.6 / 17.3 / 16.7 %; seed 1 c 18.9 / 18.1 / 15.2 / 11.7 %, a 21.3 / 18.3 / 15.6 / 12.0 %. The share of true pairs in the lowest bin is 34 % / 31 % (c) and 31 % / 29 % (a). CB by arc length from the connector's end (4 - 8 / 8 - 16 / 16+ m) seed 0 c 32336 / 22253 / 5921, a 38499 / 29849 / 7989; by the kind of the B piece (section / link) c 39123 / 21387, a 51281 / 25056.

**What it says.** (1) The extra duplicates are connector-versus-B-piece parallels (CB + BC: +27116 of the net +31496 on seed 0, +15153 of +12578 on seed 1), not connector-versus-connector (CC +4423 / -1450) and not B-versus-B. (2) The extra connectors are FALSE pairs: E320 delivers 788 / 352 more false curves at p >= 0.1 and no more true ones (-148 / +7); false curves carry +20020 / +6165 of the +20250 / +7769 extra duplicate points of the connectors (99 % / 79 %). Every curve has the same length, so a false curve that runs along a delivered piece adds about 25 duplicate points. (3) The pair head of E320 is less selective at the same threshold (pair precision at E312's recall -0.031 on seed 0, +0.003 on seed 1), consistent with E323: at an equal number of curves the excess stays. (4) The attached ends themselves are not collapsing neighbouring keypoints (E323: near-duplicate connectors under 1 %). (5) The share of duplicates per connector point is higher in every probability bin on seed 0 (+1.7 to +3.5 points), smaller on seed 1 (+0.2 to +2.4): a second, smaller effect of the attached ends that this card does not isolate. (6) E316 moved the same quantities the other way: pair precision at E312's recall +0.041 / +0.018 and dup_frac -0.0069 / -0.0025 (E316 card section 2), the mechanism is pair selectivity.

## 1. The card (what the tools run)
```json card
{
 "id": "E324",
 "card": "",
 "title": "the E320 recipe (attached ends, geometry loss) with (s) the pair stage at 6000 steps, and (d) a duplicate-aware weight on the pair negatives that run along a real lane, primary read-out `--attach --snap --snap-nogeom` at p >= 0.1, against E312 of the same seed",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.005, "vs": "c"},
 "better": {"dup_frac": "lower", "vert100_dp": "lower"},
 "arms": {
  "c": {"recipe": "control = E312 snapped (the rule step of the snap): topo F1 0.6825 / 0.6864, dup_frac 0.1747 / 0.1557, junction F1 0.3194 / 0.3143, F1@1.0 0.8669 / 0.8685, lanes found 0.7376 / 0.7423, vertices 19.55 / 20.10"},
  "u": {"recipe": "floor u' = E312 with the snap's wiring and the keypoint geometry (`--snap --snap-nogeom`): topo F1 0.6706 / 0.6760"},
  "s": {"recipe": "the E320 recipe unchanged except `--steps 6000`: `jends_train.py --e2e --scales --maps-w 1.0 --steps 6000 --geom-w 0.5 --attach` on `runs/e307_n_s<S>/model.pt`, BFLC_SIG=1.0 (run e324_s_s<S>); no new code; primary read-out `jends_eval.py score --attach --snap --snap-nogeom --thr 0.1`", "run_ids": ["E324.s_s0", "E324.s_s1"]},
  "d": {"recipe": "the E320 recipe at 3000 steps with `--dup-w 2.0` (new, default 0 = E320 bit for bit): the BCE term of every hard negative pair (a candidate pair that is not a true pair, logit above -2, at most 64 per frame) is multiplied by 1 + dup_w x ov, where ov in [0, 1] is the share of the points of its delivered curve (the Hermite curve of the attached ends and the learned scales, the same function as the geometry loss) that lie within 0.75 m of the GT lane centre points of the frame (`g05`) and run parallel to them (|cos| > 0.9); ov is a weight, not a gradient path (detached); training only, nothing changes at inference; run e324_d_s<S>; run only if s leaves dup_frac above c + 0.01 on a seed", "run_ids": ["E324.d_s0", "E324.d_s1"]}
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
 "budget_gpu_h": 10.0,
 "kill": "the loss is not finite; or the maps after the joint training are worse than E307 by more than 0.05 keypoint F1 on either seed (E320's kill, read afterwards); or for arm d: the weight does not reach the pair logits (the gradient report shows the same pair-logit gradient with dup_w 0 and 2 on hard negatives with ov > 0)",
 "evaluate": {"split": "val"}
}
```
**Verdict rule (the rule of E320, unchanged; by hand: replay for the scores of stored dumps, the training runs go through run.py):** KEEP when, on BOTH seeds, topo F1 of the PRIMARY read-out of the arm is at least u' + 0.005 and every guard holds (the arm is judged separately: s first, d only if s fails the dup guard). TIE, kept and refined (D87): a bound or a guard is missed but the topo F1 ties the snap; CUT only by the kill. The mirror and the duplicate attribution are read next to it: `jdup.py score` (false curves, duplicate points of false curves) must fall for the fix to be the cause; a fix that passes the guard without lowering the false-curve duplicates is reported as such.

## 2. Card text
- **Statement (falsifiable):** the duplicate excess of E320 is carried by false pairs that run along delivered pieces; a pair stage that is more selective (arm s, the lever of E316) or a loss that penalises exactly those pairs harder (arm d) brings dup_frac within +0.01 of E312 on both seeds while the attached ends keep topo F1 at the snap's level.
- **Why not a connector-level Matrix-NMS or a duplicate penalty between connectors that share their attached nodes (the lane lead's examples):** the diagnosis puts the excess in connector-versus-B-piece parallels (86 % of the net on seed 0, more than 100 % on seed 1); connector-versus-connector duplicates are 14 % / -12 % of it, and near-identical connectors are under 1 % (E323). A layer among connectors would act on the smallest part. A layer that sees B's pieces does not exist inside the pair head's training loop (the delivered pieces are a read-out of B); arm d therefore uses the GT lane centres in the loss, as the geometry loss uses the GT connectors.
- **Arm s (stack of recipes that already ran, no new code):** E320's recipe at 6000 pair steps: E316 doubled the pair steps of E312 and gained +0.0073 / +0.0074 snapped topo F1, +0.041 / +0.018 pair precision at E312's recall and -0.0069 / -0.0025 dup_frac; E320 has the geometry loss and the attachment. **Prediction (numbers, a guess that the effects add):** topo F1 over c: seed 0 -0.0022 + 0.0073 = +0.0051, seed 1 +0.0025 + 0.0074 = +0.0099; dup_frac over c: seed 0 +0.0184 - 0.0069 = +0.0115 (the guard of +0.01 would break by 0.0015: TIE of D87), seed 1 +0.0093 - 0.0025 = +0.0068 (holds); a - u' +0.0147 / +0.0203. A hit is within +-0.003 for topo F1 and +-0.004 for dup_frac per seed.
- **Arm d (new, training only):** a re-weighting of the BCE negatives by their overlap with real lanes (above). It has no parameter and no inference path; flag off (`--dup-w 0`) is E320 bit for bit; it needs `g05` in the training batch (the loader carries it with pixels) and the delivered curve of the hard negatives (the function of `geom_loss`, applied to at most 64 negatives per frame; cost about +5 % per step, not measured). **Prediction:** it removes half of the extra duplicates of false pairs: dup_frac over c +0.009 (seed 0), +0.005 (seed 1), topo F1 over c -0.002 / +0.002 (the attached ends and the geometry loss unchanged; the pair precision improves). Run only if s fails.
- **Controls and confounds:** c and u' as in E320; E320 as measured (a) is the reference of the defect; the 6000-step arm changes only the training length (E316's own pairing is the same recipe change); the arm d weight is a training-only change measured against a (the same recipe at 3000 steps). One training per arm and seed: the scene bootstrap is the interval. The share of the second effect (point 5 of the diagnosis) stays in the arms.
- **Cost / price (reported next to it):** at inference nothing changes against E320 (9.6 k parameters, 3.9 MMACs of `Attach`; the pair stage's step is 1.9 - 2.2 s under load); training: arm s 3.2 h per seed (6000 steps), arm d about 1.9 h per seed; budget 10 GPU hours if both run (two slots), 6.4 if only s.
- **Named hard steps (unchanged from E320):** the argmin of the wiring at the read-out, the candidate set of the attachment, the keypoint peak choice, p >= 0.1, the close-node step (E292).
- **Not claimed:** that the attached ends are the cause of the second, smaller effect; anything on TEST (D84 covers only E312); that E317 content is re-launched (the geometry loss is part of the E320 recipe that already ran; the holds on E317 / E318 / E319 stay).

## 3. What has to be built (after the critic pass)
Arm s: nothing (`temp/e324/start_pairs_s.sh` from the E320 script with `--steps 6000`, READY file, the post chain of day12c). Arm d: `--dup-w` in `jends_train.py` and the weight in `jends_loss` (`training/grid/jends.py`), the loader flag for `g05` in the jends batch, tests (flag off identical to E320 on a seeded synthetic batch, the weight value on a constructed scene: a negative along a lane gets 1 + dup_w, a negative far from lanes gets 1, a true pair 1; the gradient on the weighted negatives; mutations: the weight applied to true pairs, the weight attached to the gradient path). A fresh critic pass on the code before the READY file.

## 4. Referee
The fresh `ar-critic` pass (cards mode) is written below when it has run.
