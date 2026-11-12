# E317 - a geometry loss on the delivered connector (shape and ends supervised directly), on the E312 recipe

Card of 2026-11-12 22:50, written by the lane chat (Mirko's question of the evening: why does the right shape not appear by itself, and his three hypotheses: the shape is not a constraint, the position uncertainty is not large enough, forward and backward information exchange is missing). This card is the first hypothesis, E318 the second; the third (message passing along the lane) is not built yet.

Origin. The pair head of E312 is trained with the pair BCE, the dual softmax, the crossing penalty, the handle-scale L1 (E303) and, jointly, the maps loss; NOTHING compares the delivered curve with the true connector, and the position of a keypoint is only the soft-argmax over the 3 x 3 window of the 1 m heat (the keypoint ends are 1.44 m off the GT ends on VAL, E310). The delivered curve of a pair is a function of exactly four things, the two keypoint positions, the two headings and the two handle scales, and a loss on that curve reaches all of them (positions through the soft-argmax into the heat, so into the maps head with `--e2e`). Measured before this card (lead chat, `temp/e309/lines_stats.py`, VAL, seed 0): the joins between delivered lines have a heading kink of median 1.7 deg / p90 9.1 deg on E312 (2.5 / 11.6 deg on the plain graph B) where the GT has 0.1 / 1.6 deg: the shape is not a constraint of anything but the parametrisation. The read-out of E312 snaps the connector ends to the delivered pieces (E311); that step is a rule on the output (hard, not differentiable in the loss). The question of this card is also whether a geometry loss makes it unnecessary.

## 0. The card (what the tools run)
```json card
{
 "id": "E317",
 "card": "",
 "title": "geometry loss on the delivered connector (`jends_train.py --geom-w 0.5`: the Bezier curve of every true keypoint pair against the GT connector polyline + the keypoint ends against the GT ends) on top of the E312 recipe (`--e2e --scales --maps-w 1.0 --steps 3000`, E307 maps, BFLC_SIG=1.0), snapped read-out at p >= 0.1, against E312 of the same seed",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.005, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = E312 of the same seed (`runs/e312_p_s<S>`, same maps, same recipe, no geometry loss): snapped p >= 0.1 topo F1 over the plain graph +0.0566 / +0.0615 (0.6825 / 0.6864); unsnapped p >= 0.1 0.6438 (+0.0179) on seed 0, seed 1 scored 2026-11-12 evening (`ends_card_thr0.1.json`)"},
  "g": {"recipe": "`jends_train.py --e2e --scales --maps-w 1.0 --steps 3000 --geom-w 0.5` on `runs/e307_n_s<S>/model.pt`, BFLC_SIG=1.0 (run e317_g_s<S>); same read-outs (snapped p >= 0.1 = the rule, unsnapped p >= 0.1 = the differentiability question)"}
 },
 "control": "c",
 "variant": "g",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 4.0,
 "kill": "the loss is not finite, or the maps after the joint training are worse than E307 by more than 0.05 keypoint F1 on either seed",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): `jends_eval.py score --snap --thr 0.1` (the rule's read-out) and `jends_eval.py score --thr 0.1` (unsnapped), differences to the plain graph of the seed, then to E312. KEEP when snapped topo F1 over E312 rises by more than +0.01 on BOTH seeds with the guards held; TIE kept (the E312 recipe stays the cheaper one: the loss costs no inference time, but it is one more term); CUT below -0.01. **The differentiability reading (written before the run):** the gap of E312 between the snapped and the unsnapped read-out at p >= 0.1 is gap_c = +0.0387 topo F1 on seed 0 (0.6825 against 0.6438; seed 1 from the scored `ends_card_thr0.1.json`). The geometry loss "makes the snap unnecessary" when the UNSNAPPED p >= 0.1 topo F1 of arm g exceeds that of E312 by more than +0.02 on both seeds (more than half of gap_c) while the snapped one does not fall below -0.01: then the snap step becomes a candidate for removal and a follow-up card compares g unsnapped with E312 snapped. Reported next to it: the share of the gap closed ((unsnapped_g - unsnapped_c) / gap_c), keypoint end error (the `geom_end_d` of the training log and the head's own end recall / precision from `ends_head.json`), the heading kink at the joins of the delivered graph (`temp/e309/lines_stats.py`, scratch), pair recall / precision, the maneuver coverage (`jmaneuver.py`, p0.1s), the lenient topo F1 at 2 m (`temp/e309/sens.py`, scratch), maps keypoint F1 after the joint training (`jmaps_eval.py score --run e317_g_s<S>_maps`).

## 1. Design and prediction
Single factor: the geometry loss with weight 0.5 (`--geom-w`); everything else is the E312 recipe, the same maps (`runs/e307_n_s<S>`), the same seed, the same 3000 steps. Two seeds, VAL 873 frames, TEST untouched.

What the loss is (`training/grid/jends.py: geom_loss`, tests in its self-test, flag off = the E312 recipe bit for bit, checked against the original file on a synthetic batch and on real frames with the trained E312 model, `temp/e317/t_dev.py regress`, `t_real.py`):
1. **Shape.** For every true (entry, exit) pair whose two ends are keypoints (the pairs the BCE calls positive, `y = 1` and allowed) the DELIVERED curve is built in torch (`_bezier_s`: the cubic Bezier of `qends.curve_s` with the keypoint positions, the keypoint headings and the learned handle scales; 16 samples) and compared with the GT connector polyline of the pair (`batch['ppts']`: the first connector run of the pair, 16 points of equal arc length, `qjunction.pair_polylines`, a data field that exists only with `--geom-w`): symmetric point-to-polyline distance (curve -> GT and GT -> curve; the point-to-SEGMENT distance, so the sampling does not enter), smooth L1 with delta 1 m.
2. **Ends.** Every valid keypoint within 4 m and 30 degrees of a GT end of its kind (the matching constants of `qendnet`) against the NEAREST such end, smooth L1 with delta 1 m. Many to one on purpose: the BCE matching is one to one (a real peak next to the keypoint that sits on the injected GT cell counts as a false keypoint for the pair target), but every real peak is a connector end at read-out, so its position counts.
Gradient (self-test): into the entry / exit heat (the soft-argmax positions, so the maps head with `--e2e`), into the heading maps, into the scale head. Not into the pyramid (frozen). Soft-argmax moves a keypoint inside its 3 x 3 window at most, so the loss corrects the sub-cell position and shapes the heat around the end; a peak that is two cells off is not moved by it (the maps loss and the heat sigma of E307 / E313 work on that).

Caveat (measured before the run): in training the GT ends are keypoint candidates (injected), so the matched keypoint of an end is mostly the injected one; the end error of E312's own keypoints on 16 TRAIN frames is 0.65 m (VAL without injection: 1.44 m) and the shape distance of E312's curves to the GT connectors is 0.38 m (curve -> GT) / 0.37 m (GT -> curve), `temp/e317/t_real.py geom`. The loss has therefore a smaller start value than the VAL errors suggest and works mostly through the shared weights (heat around the true end cell, headings, scales).

**Prediction (a number):** topo F1 +0.005 over E312 on the snapped read-out (a TIE: the snap already repairs most of what the loss supervises); on the unsnapped read-out p >= 0.1 topo F1 +0.015 over E312 (a little more than a third of gap_c); the scale head's L1 against the fitted scales falls by 10 %. If the unsnapped number does not move, the shape is not what the delivered graph lacks (then the rule step of E311 stays and the next card is about where the error lives: the E316 / E315 training length, E299 time).

## 2. Cost
- Training: the geometry term costs 4.2 ms forward and 9.5 ms with the backward pass on a real batch of 8 (CPU, 2 threads, `t_real.py bench`; on the GPU it is a handful of small kernels); the step is about 1.9 s (the maps head and the pyramid dominate), so about +0.5 %. The data field `ppts` is a numpy resample of the pair runs per frame, negligible. 3000 steps = 1.6 h per seed, two seeds in the two GPU slots, then dump and scores about 15 minutes.
- Inference: 0 ms, 0 parameters (the loss is a training term; the read-out is unchanged).
- Files changed for it: `training/grid/jends.py` (`geom_loss`, `_bezier_s`, `_dist_to_polyline`, `jends_loss(geom_w)`), `training/grid/qjunction.py` (`pair_polylines`), `training/data/lane_gt_data.py` (`DEFAULT_PPTS`, collate), `training/train/jends_train.py` (`--geom-w`), `evaluation/jends_eval.py` (loads E318 runs).
- Start (not started; the day-chain 11 holds both GPU slots until about 04:30): `temp/e317/start_pairs.sh <seed>`.
