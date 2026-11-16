p = "experiments/E299_temporal-end-prior/experiment.md"
s = open(p, encoding="utf-8").read()
add = '''
## 4. Step 0 result (CPU, VAL 873 frames, 851 consecutive pairs of frames per seed, `evaluation/jtime.py`, run 2026-11-13 21:45; weight 0.5 PRE-REGISTERED; files `temp/jtime/e312_p_s<S>_ceiling.json` and `_replay_w0.5.json`)
| | seed 0 | seed 1 |
|---|---|---|
| true pairs delivered at t-1 (p >= 0.1) | 5822 | 5816 |
| of them delivered at t too | 4320 | 4259 |
| missed at t: logit-only flips (both ends have a counterpart keypoint, p < 0.1) | 408 | 413 |
| missed at t: end-miss flips (an end has no counterpart keypoint at t) | 1094 | 1144 |
| **logit-only share of the missed pairs** (the ceiling of a re-scoring layer; kill below 25 %) | **27.2 %** | **26.5 %** |
| flip rate (true pairs delivered in exactly one of two consecutive frames / in either) | 33.1 % | 34.1 % |
| replay w = 0.5: pair precision at the recall of the original, logit 0 (recall 0.300 / 0.328): original -> replay | 0.6738 -> 0.6786 (+0.0048) | 0.6649 -> 0.6675 (+0.0026) |
| replay w = 0.5, at the delivered operating point logit -2.2 (recall 0.538): original -> replay | 0.5243 -> 0.5357 (**+0.0113**) | 0.5029 -> 0.5139 (**+0.0110**) |
| replay w = 0.5: snapped topo F1 (original reproduces c: 0.6825 / 0.6864) | +0.0005 | +0.0001 |
| replay w = 0.5: dup_frac / lanes found | -0.0004 / -0.0009 | -0.0015 / not read |
**Verdict of step 0 by its written rule: PASSES, narrowly.** The logit-only flips are 27.2 % / 26.5 % of the missed pairs (kill below 25 %), the replay at the pre-registered weight raises the pair precision at the recall of the delivered operating point by +0.011 on both seeds (the bound was 0.01; at logit 0 it is +0.005 / +0.003, below it), and snapped topo F1 is not below c - 0.005 (it is +0.0005 / +0.0001). **Reading:** about three quarters of the pairs that are lost between two keyframes are lost because an END has no keypoint in the next frame, which no re-scoring of the current peaks can recover; the part a state can re-score is worth about +0.011 precision at fixed recall for a parameter-free average and nothing in topo F1. A learned gate can do more than the average, but the evidence for the training arm is thin: **expected gain: precision +0.01 to +0.03, topo F1 about +0.003 (the prediction of the card); cost 9 GPU hours.** Under the D86 weights (detection up, little compute) the training arms are LOW priority: they run only when the nights are free after the end-to-end replacements (E292, E325, E324); the flip rate of 33 % (higher than E290's 20 % because a flip is counted by predicted counterparts) says the instability is real. The weight sweep is not run (it would be an in-sample choice on VAL).
'''
marker = "\n## 3. Referee"
assert marker in s
s = s.replace(marker, add + marker, 1)
open(p, "w", encoding="utf-8").write(s); print("ok")
