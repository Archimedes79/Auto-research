p = "experiments/E287_junction-pair-head-v2/experiment.md"
s = open(p, encoding="utf-8").read()
a = s.index("## 4. Controls measured before the runs")
b = s.index("The first temporal measurement is in E290 section 4")
b = s.index("\n", b)
new = '''## 4. Controls measured before the runs (2026-11-11, recomputed 2026-11-12 after the critic's findings; CPU, `evaluation/jrule_control.py`, the keypoints of the E284 head h2, VAL 873, 11083 true pairs)
The rule control (written in the file's docstring before the numbers; groups and ranks of the keypoints, a connected exit group per turn class, lane matching by normalised rank, extreme lanes for the turns) on the SAME keypoints as the E284 head. Matching of keypoints to GT ends is STRICT one-to-one (4 m, 30 degrees; a keypoint serves one GT end only) and the rule sees only the keypoints the head also sees (probability >= the threshold); the first version of this section (11 Nov: rule recall 0.37 / precision 0.13) let the rule use all keypoints and counted a GT pair as found when ANY keypoint near the end took part, which flattered the rule's recall and understated its precision.
| keypoint threshold | seed | ceiling: both ends of a pair found | rule recall / precision (pairs) | E284 head (logit >= 0) recall / precision (pairs) |
|---|---|---|---|---|
| 0.5 | 0 | 0.429 | 0.357 / 0.337 (11733) | 0.289 / 0.741 (4329) |
| 0.5 | 1 | 0.417 | 0.346 / 0.332 (11583) | 0.294 / 0.746 (4370) |
| 0.3 | 0 | 0.516 | 0.403 / 0.263 (16988) | 0.311 / 0.680 (5061) |
| 0.3 | 1 | 0.510 | 0.392 / 0.259 (16789) | 0.321 / 0.678 (5243) |
Reading: the rule delivers 2.7 - 3.4 times as many pairs and is wrong for two thirds of them (precision 0.33 against 0.74): as a decision it is far behind the head. But its recall (0.35 - 0.40) is higher than the head's (0.29 - 0.32) at the same keypoints, and the CEILING (both ends of the pair found) is 0.42 - 0.52: the head delivers 67 - 70 % of what its keypoints allow, the rule 83 - 78 %. **Two consequences for this card: (1) the keypoints (end recall, E286) are the first lever: the ceiling is 0.42 at threshold 0.5; (2) the pair decision still leaves 0.13 of the pairs on the table at logit >= 0, which is what the precision-recall curve (`jpair_pr.py`, E302) shows and what the rank features of p1 / p2 give the head the information for.** The head p0 / p1 / p2 is compared with the rule at equal keypoints (`jrule_control.py score --run e287_<arm>_s<S>`). The first temporal measurement is in E290 section 4 (pair flips 20 %, 52 - 54 % of the true pairs delivered in neither of two frames).'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
