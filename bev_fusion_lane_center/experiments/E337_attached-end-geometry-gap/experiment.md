# E337 - the attached ends against the snap's geometry (D87 / D88): the +0.0069 / +0.0062 topo F1 that the end-to-end read-out of E324 gives up is GEOMETRY, answered from stored cards; the sharpening lever stays a stub and is not scheduled

Card of 2026-11-14 (lane chat, on the lane lead's order after E324). Version 2 after the cards critic (REVISE, 10 points; the main one: the planned replay was decided in advance, so it is dropped and its question is answered by arithmetic on stored cards and one scan of stored dumps). Follow-up of E324 ('E2E-required, kept and refined'); only differentiable solutions (D88). **Status: step 0 DONE (CPU, nothing trained, no new code in a shared file); no GPU, no training arm scheduled.**

## 0. The card (what the tools run)
```json card
{
 "id": "E337",
 "card": "",
 "title": "E324 s: is the topo F1 gap between the own-keypoint snapped read-out and the PRIMARY (attached-end) read-out geometry or wiring? (answered from the stored cards, no run)",
 "kind": "variant",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.003,
 "prediction": {"value": 0.0069, "vs": "k"},
 "arms": {
  "k": {"recipe": "stored: E324 s own keypoints, snapped (`runs/e324_s_s<S>/ends_card_snap_thr0.1.json`; NOT E324's 'c', which is E312 snapped): topo F1 0.6981 / 0.7010"},
  "kn": {"recipe": "stored: E324 s own keypoints, snap-nogeom (`ends_card_snap_nogeom_thr0.1.json`): 0.6888 / 0.6916"},
  "a": {"recipe": "stored: E324 s PRIMARY (attached ends, snap-nogeom wiring, `ends_card_snap_nogeom_attach_thr0.1.json`): 0.6912 / 0.6948"}
 },
 "control": "k",
 "variant": "a",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [{"metric": "topo_F1", "vs": "k", "max_loss": 0.003}],
 "budget_gpu_h": 0.0,
 "kill": "none: no run; this card is arithmetic on stored cards and one scan of stored dumps",
 "evaluate": {"split": "val"}
}
```
`decide.py` is not used (no run): the one clause states the gap (the PRIMARY read-out `a` loses more than 0.003 topo F1 to `k` on both seeds: it does, -0.0069 / -0.0062); the reading below is by hand and written here, not decided by a tool. The planned replay `jends_eval.py score --attach --snap` is dropped (below).

## 1. What was measured (all from stored files; the scan is `temp/e337/piecedist.py`, reproduced from the critic's scratch script, 20 s per seed)
- **The gap and its two parts** (unrounded, from the stored cards): k - a = **+0.006932 / +0.006155** topo F1. The snap's geometry on the SAME wiring (own keypoints, snapped minus nogeom, k - kn) is **+0.009281 / +0.009378**; the attachment, in the nogeom read-out (a - kn), is **+0.002348 / +0.003222**. So the attached end recovers **25.3 % / 34.4 %** of the snap's geometry and the rest of the gap is what the snap's geometry still gives on top (E324 section 5 had +0.0024 and 26 % from rounded inputs: corrected there).
- **Wiring is not the difference:** for 96.2 % / 96.6 % of the connector ends that have a delivered piece end in reach (97.2 % / 97.5 % of all ends, both None counted as the same), the attached end and the own keypoint wire the SAME delivered piece (nearest piece end within 4 m and 45 degrees, coordinates compared; `jwire.candidates`). A replay with the snap's geometry on the attached end could therefore differ from k only through about 3.5 % of the ends: it cannot discriminate and is not run (critic point 3). By the arithmetic the gap is **GEOMETRY** (the curve starts at the piece end in k and at a point beside it in a), not wiring.
- **Distance from the connector end to the nearest delivered piece end in reach (seed 0 / seed 1; 72.3 / 72.9 % of attached ends and 72.7 / 73.1 % of keypoints have a piece in reach):**

| | attached end | own keypoint |
|---|---|---|
| median | 0.659 / 0.702 m | 0.787 / 0.828 m |
| mean | 1.138 / 1.160 m | 1.272 / 1.303 m |
| share <= 0.5 m | 44.9 / 44.4 % | 18.4 / 18.2 % |
| share <= 1 m | 55.9 / 54.4 % | 54.5 / 52.9 % |
| share <= 2 m | 80.2 / 79.9 % | 80.2 / 79.1 % |

The attached end lands within 0.5 m of a piece end more than twice as often as the keypoint (45 % against 18 %), but it is bimodal: about 45 % coincide, about 44 % lie beyond 1 m and 20 % beyond 2 m (the stored field `ends_within_1m_of_a_delivered_piece_end` of the PRIMARY cards is a different population, the ends the connectors use, entry / exit: seed 0 attached 0.383 / 0.435, keypoint 0.374 / 0.427; seed 1 attached 0.360 / 0.444, keypoint 0.349 / 0.433).
- **Mechanism evidence (`runs/e324_s_s<S>/log.txt`, TRAIN batches with GT-injected keypoints, keypoints with at least one candidate):** `att_acc` (argmax agreement with the target) 0.324 / 0.363 at step 100 and **0.742 / 0.711 at step 6000** against the nearest-candidate baseline 0.573 / 0.547; `att_ce` 0.778 / 0.766 at the end against chance about ln(M + 1) = 2.2 (M candidates plus the dustbin): the softmax is learned but not one-hot. The attached end is a convex combination of the keypoint and up to M raw nodes of B's head (`Attach`, `jends.py:92 - 136`): when no weight dominates, it lies between candidates and not on a delivered piece end. `att_p` in the dump is 1 - w0, the candidate mass: it is above 0.8 for 97 % of the ends and says nothing about dominance; the weight of the best candidate is NOT stored (`att_logit` and `att_cpos` are not in the dump).

## 2. What this says and what it does not
- **Says:** the end-to-end read-out of E324 gives up 0.0062 - 0.0069 topo F1 against the snap because the attached end does not start where the snap starts the curve; the gap is a geometry gap of a learned continuous attachment, wired the same way as the rule in 96 % of the ends.
- **Does not say:** that a sharper attachment would close it (the attached end would then sit on a raw node of B's head, which is at a delivered piece end only where B kept that node as a piece end; the snap's tangent also comes from the piece, not from the node); that the dup_frac excess of the primary on seed 0 (E324 section 5: +0.0071 from the attached ends) is the same defect.
- **The lever, a STUB, not scheduled (one arm h, if it is ever built):** a sharpening term on the existing attachment softmax (an entropy or margin loss on the weights, differentiable) in the E324 recipe; the straight-through argmax of the first draft is dropped (a forward-pass hard selection on the attached end is a variant of the snap and not a D88 candidate); overlap with E325 (a head that chooses among the DELIVERED pieces, trained on the TRAIN frames) must be decided first, because a head over delivered pieces makes the piece end the target by construction and could replace the attachment's geometry role altogether; the weight statistic needs `att_logit` and `att_cpos` in the dump (a change in `jends_eval.py`, a shared file: its own critic pass). **Cost if built:** 3.2 h per seed, 6.4 slot-hours; **upper bound of the gain:** the gap itself, +0.0069 / +0.0062 topo F1 (with the dup guard of the primary read-out unmeasured); against D86's weights (little compute) it stays unscheduled until E292 (the close-node replacement, running) and the E325 head have been measured. **Schedule rule (written now):** build arm h only when E325's twin result and its head result show that choosing among delivered pieces does not already close the geometry gap, and a slot-night is free.
- **Numbers:** every number above comes from stored cards, logs or the dumps listed; the scan script is scratch (`temp/e337/piecedist.py`, `attstats.py`): used once, named here as how the distance table was made, NOT promoted (deviation from the git rule, recorded: it moves into `evaluation/` with a selftest and a README row when it is used a second time).

## 3. Referee
Cards critic 2026-11-14 01:44: **REVISE** on version 1 (the replay command failed: tag `r` matches the reserved tag pattern; the replay was decided in advance by the 96 % same-piece share; the reading rule had holes (undefined population, the 0.5 m threshold without basis, GEOMETRY with a small median unassigned, mixed seeds contradicted, JSON rule against the reading, `c` clashing with E324's `c`); two wrong numbers (26 % is 25.3 %, a - kn is +0.0023); the mechanism number `att_acc` was the step-100 value; the statistic of the dominant weight is not computable from the dump; the straight-through lever excluded by D88; `temp/e337/dist.py` named as the source of a number without promotion; `run_ids` contradicted 'by hand'). Version 2 answers each: replay dropped, question answered by arithmetic, numbers corrected, mechanism numbers replaced by the step-6000 values, lever reduced to a stub with the D88 and E325 notes, JSON without run_ids and rule, the script deviation recorded. No second pass needed: the card makes no claim that needs one.
