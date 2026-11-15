import io
p = "experiments/E291_sd-road-level-lane-counts/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
a = "## 4. Referee"
assert s.count(a) == 1
add = '''**Step 0 result (2026-11-15 11:10, CPU, `temp/e291/step0.py` on the stored VAL dumps of E312 and E324 s, both seeds; 826 of the 873 VAL frames have junction GT; per arm and direction (6 058 entry sides, 5 977 exit sides); a stat of keypoints against the GT end count of the arm, `temp/e291/step0_thr<thr>_r<radius>.json`):**

| setting (real keypoints = peaks of the maps head with kp_p >= thr, within r of the arm centre, heading within 60 degrees) | deficit (found < n) | exact | surplus (found > n) | GT ends missed (share of GT ends) | surplus keypoints (share of found) |
|---|---|---|---|---|---|
| strict: thr 0.3, r 6 m (E312 s0 / s1; E324 s about the same) | 0.299 / 0.295 (in), 0.299 / 0.293 (out) | 0.62 / 0.61, 0.65 / 0.65 | 0.082 / 0.099, 0.051 / 0.059 | 0.253 / 0.249 (in), 0.251 / 0.243 (out) | 0.080 / 0.100, 0.050 / 0.058 |
| loose: thr 0.1, r 10 m (E312 s0 / s1) | 0.130 / 0.129 (in), 0.112 / 0.111 (out) | 0.635 / 0.610, 0.679 / 0.671 | 0.235 / 0.260, 0.209 / 0.218 | 0.108 / 0.105 (in), 0.093 / 0.089 (out) | 0.192 / 0.216, 0.179 / 0.192 |

**Reading (the ceiling).** (1) A count and the found keypoints disagree in 91 - 94 % of the junction frames (at least one arm), in 35 - 40 % of the arm sides (deficit + surplus). (2) **A count cannot give back a missing end** (10 % of the GT ends at the loose setting, 25 % at the strict one are not found as a keypoint: the pair head has nothing to pair), so the part of the deficit is out of reach of the prior by construction; the part a count CAN act on is the SURPLUS: 21 - 26 % of the arm sides carry more keypoints than ends (19 - 22 % of the found keypoints at the loose setting) and these are the duplicates and false pairs that `dup_frac` (0.15 - 0.18 on E312 / E324 s) counts. (3) The setting matters (the deficit shares move from 0.30 to 0.12 with the threshold and radius), so these numbers bound the potential, they do not predict the gain; the card's prediction of +0.008 stays a guess. (4) This is the gain's ceiling for an ORACLE count (the table is made of GT end clusters, section 1): a real map's count is no better.

'''
s = s.replace(a, add + a)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
