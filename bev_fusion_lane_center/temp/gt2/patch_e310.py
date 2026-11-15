p = "experiments/E310_maneuver-coverage-and-connector-headroom/experiment.md"
s = open(p, encoding="utf-8").read()
i = s.index("## 2. Result")
j = s.index("## 4. What it means for the queue")
new = '''## 2. Result (E308 seed 0 keypoints, VAL 873; `evaluation/jmaneuver.py score --run e308_t_s0 --seed 0 --variants plain p0.5 p0.2 p0.1 oracle oracle+fit --rows` -> `runs/e308_t_s0/maneuver_val.json`)
The first version of the measure (scratch, 2026-11-12 08:50) limited the direction test to the 6 nearest delivered points; the critic pass of `evaluation/jmaneuver.py` found that a crowded graph hides the right line behind wrong-direction duplicates and penalises graphs with more lines (every 3rd VAL frame: plain 0.510 against 0.531 with all neighbours, p >= 0.5 0.544 against 0.573). The numbers below are the corrected ones (all neighbours within 1.5 m); the oracle now uses every `valid` keypoint like the p variants. Superseded first numbers: plain 0.504, p >= 0.5 0.538, p >= 0.1 0.573, oracle 0.591, oracle + fitted scales 0.615.
| delivered graph | maneuvers realised | share | gain over plain | junction precision |
|---|---|---|---|---|
| plain (B + hard close-node step) | 5773 / 10991 | 0.525 | | 0.953 |
| + connectors p >= 0.5 | 6223 | 0.566 | +0.041 | 0.948 |
| + connectors p >= 0.2 | 6572 | 0.598 | +0.073 | 0.939 |
| + connectors p >= 0.1 | 6770 | 0.616 | +0.091 | 0.934 |
| + oracle pair classifier (true pairs of the matched keypoints, chord tangent) | 7066 | 0.643 | +0.118 | 0.943 |
| + oracle pairs with handle scales fitted to the GT | 7375 | 0.671 | +0.146 | 0.951 |
**Reading.** The plain graph of B already draws half of the maneuvers. The connectors of E308 add 4.1 points at p >= 0.5 and 9.1 at p >= 0.1 (no training: the threshold of E309); a perfect pair classifier would add 11.8, perfect handle scales another 2.8: at p >= 0.1 the pair decision has 2.7 points left. The first version had also scored the connectors snapped to the delivered pieces (E311) with this geometric measure: no change (-0.2 .. +0.2 points); E311 is judged by the topology scores, where it moves much more.

## 3. Where the rest is lost
- Maneuvers by the ends found (loose match, 4 m / 30 degrees, any keypoint with p >= 0.5): both ends 69.5 %, one end 22.9 %, none 7.6 %. Realised in the plain graph: 0.569 / 0.436 / 0.394; with perfect pairs and scales: 0.774 / 0.448 / 0.400.
- Perfect pairs, both ends found: straight 80 %, left 64 % (78 % with fitted scales), right 60 % (70 %); chord up to 10 m 81 %, 20 - 35 m 68 % (77 %), 35 - 60 m 63 %; entry range 0 - 20 m 77 %, beyond 35 m 72 %. Left and right turns and the long connectors are where the fitted scales help most (E303); straight maneuvers are limited by the ends and by lanes that the delivered graph does not draw.
- Of the 3925 maneuvers not realised with perfect pairs, 2034 have both ends found, 1389 one end and 502 none.
- Curve from the TRUE ends against the GT connector (`temp/gt2/curve_vs_gt.py e308_t_s0`, 5837 pairs with both ends strictly matched; share of pairs with at least 90 % of the GT connector points within 1.5 m): chord tangent 0.871, fitted handle scales 1.000. From the KEYPOINTS (mean position error 1.44 m, 90 % within 2.5 m): chord 0.664, fitted 0.733. So the curve family is right, the scales are worth 13 points (E303), and the end localisation costs another 27 (the E284 head had 1.18 m).
- Not wired: only 3504 of 5798 oracle curves are attached to a delivered piece at both ends (60 %): B does not deliver the lane next to the end. The geometric measure does not need wiring; the topology scores (topo F1, junction F1) do, and that is what E311 exploits.
- The maneuver denominator (826 frames, 10991 maneuvers) is smaller than the 11083 GT pairs of the pair numbers: the 100 m box and the 3-point rule drop 0.8 %.

'''
s = s[:i] + new + s[j:]
s = s.replace("Command: `wsl bash tools/env/_py14.sh ../jr/maneuver.py <pair run> <seed> [<variants>]` (scratch, `temp/jr/maneuver.py` + `snapconn.py`; to be promoted into `evaluation/` because the day chain and the report use it a second time) and `temp/gt2/curve_vs_gt.py <run>`.",
              "Command: `wsl bash tools/env/_py14.sh jmaneuver.py score --run <pair run> --seed <S> --variants plain p0.5 p0.1 oracle oracle+fit --rows` (promoted into `evaluation/jmaneuver.py` with a self-test, README row and gate row after a critic pass, 2026-11-12) and `temp/gt2/curve_vs_gt.py <run>` (scratch).")
s = s.replace("a GT MANEUVER is one entry -> exit connection of a junction: the connector runs of the lane GT grouped by their entry / exit cluster (`qends.cluster`), the longest run per (entry, exit); 10991 maneuvers in 826 junction frames of VAL.", "a GT MANEUVER is one entry -> exit connection of a junction: the connector runs of the lane GT grouped by their entry / exit cluster (`qends.cluster`), the longest run per (entry, exit); 10991 maneuvers in 826 junction frames of VAL.")
s = s.replace("points of its GT run (every 1 m, inside the 100 m box) have a delivered polyline point within 1.5 m", "points of its GT run (the GT points, 0.5 m apart, inside the 100 m box) have a delivered polyline point within 1.5 m")
s = s.replace("E303 (handle scales: +13 points of the pairs that are found) and everything that moves the ends (E307 narrower end heat, E304 far range, a position offset for the keypoints) are worth more than the pair decision (E302, E298). Day chain 4 was re-ordered that way on 2026-11-12 09:00. Next: score each pair run with `maneuver.py` next to the D79 numbers; promote the measure into `evaluation/` with a self-test.",
              "E303 (handle scales: +13 points of the pairs that are found) and everything that moves the ends (E307 narrower end heat, E304 far range, a position offset for the keypoints) are worth more than the pair decision (E302, E298). Day chain 4 was re-ordered that way on 2026-11-12 09:00. The measure is promoted (`evaluation/jmaneuver.py`); the day chain scores each pair run with it next to the D79 numbers. What moved the topology scores most was not on this list: snapping the connector ends to the delivered pieces (E311).")
open(p, "w", encoding="utf-8").write(s)
print("ok")
