# E310 - maneuver coverage and where the connectors lose maneuvers

Mirko, 2026-11-12: "do the numbers express what we look for qualitatively?" The pair numbers (recall 0.18, precision 0.6) and topo F1 (+0.005) say little about what the pictures show (the delivered graph draws most junctions). This card defines the number that matches the pictures and measures the headroom of every link of the chain with it. A measurement card: no training, CPU, VAL 873 frames, TEST untouched.

## 0. The card (what the tools run)
```json card
{
 "id": "E310",
 "card": "",
 "title": "maneuver coverage of the delivered graph (plain, with connectors at p >= 0.5 / 0.2 / 0.1, with a perfect pair classifier, with perfect handle scales, with snapped connector ends) and the decomposition of what is lost",
 "kind": "variant",
 "deterministic": true,
 "metric": "maneuver_recall",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.03, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = the plain delivered graph (B + the hard close-node step), no connectors"},
  "t": {"recipe": "the same graph + the connectors of E308 (seed 0) at p >= 0.5; further arms are rows of the table (thresholds, oracle pair classifier, oracle + fitted handle scales, snapped ends)"}
 },
 "control": "c",
 "variant": "t",
 "pairing": "position",
 "seeds": [0],
 "rule": [
  {"metric": "maneuver_recall", "vs": "c", "min_effect": 0.03}
 ],
 "budget_gpu_h": 0.0,
 "kill": "the plain graph does not reproduce the stored graph of E284 (topo F1 0.6259)",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: replay"): a measurement, no verdict beyond the table; the number is new (written 2026-11-12), so it is NOT a rule set before a run of the arms it compares: every row is reported, none is selected. Command: `wsl bash tools/env/_py14.sh ../jr/maneuver.py <pair run> <seed> [<variants>]` (scratch, `temp/jr/maneuver.py` + `snapconn.py`; to be promoted into `evaluation/` because the day chain and the report use it a second time) and `temp/gt2/curve_vs_gt.py <run>`.

## 1. The measure
A GT MANEUVER is one entry -> exit connection of a junction: the connector runs of the lane GT grouped by their entry / exit cluster (`qends.cluster`), the longest run per (entry, exit); 10991 maneuvers in 826 junction frames of VAL. A maneuver is REALISED in a delivered graph when at least 90 % of the points of its GT run (every 1 m, inside the 100 m box) have a delivered polyline point within 1.5 m whose direction differs by at most 40 degrees (the opposite lane or a crossing lane does not count). No pairing and no tolerance at the ends: the question is "is the lane drawn there, in that direction". Next to it: the junction precision = delivered points within 8 m of a GT connector run that lie within 1.5 m of a GT lane point.

## 2. Result (E308 seed 0 keypoints, VAL 873; `temp/jr/media/errors/maneuver_e308_t_s0.json`)
| delivered graph | maneuvers realised | share | junction precision |
|---|---|---|---|
| plain (B + hard close-node step) | 5535 / 10991 | 0.504 | 0.953 |
| + connectors p >= 0.5 | 5913 | 0.538 | 0.948 |
| + connectors p >= 0.2 | 6178 | 0.562 | 0.939 |
| + connectors p >= 0.1 | 6297 | 0.573 | 0.934 |
| + oracle pair classifier (true pairs of the matched keypoints, chord tangent) | 6492 | 0.591 | 0.943 |
| + oracle pairs with handle scales fitted to the GT | 6757 | 0.615 | 0.951 |
| p >= 0.5, ends snapped to the delivered pieces | 5887 | 0.536 | 0.948 |
| p >= 0.1, snapped | 6273 | 0.571 | 0.935 |
| oracle pairs, snapped | 6425 | 0.585 | 0.945 |
| oracle pairs with fitted scales, snapped | 6786 | 0.617 | 0.955 |
**Reading.** The plain graph of B already draws half of the maneuvers. The connectors of E308 add 3.4 points at p >= 0.5 and 6.9 at p >= 0.1 (no training: the threshold of E309); a perfect pair classifier would add 8.7, perfect handle scales another 2.4: the pair decision has about 5 points left at p >= 0.5 and 1.8 at p >= 0.1. Snapping the connector ends to the delivered pieces changes nothing (-0.2 .. +0.2 points).

## 3. Where the rest is lost
- Maneuvers by the ends found (loose match, 4 m / 30 degrees, any keypoint): both ends 69.5 %, one end 22.9 %, none 7.6 %. Realised in the plain graph: 0.546 / 0.416 / 0.381.
- Oracle (perfect pairs, chord tangent) on the maneuvers with both ends found: 67 %; straight 75 %, left 55 %, right 52 %; chord up to 10 m 81 %, 20 - 35 m 59 %, 35 - 60 m 53 %.
- Curve from the TRUE ends against the GT connector (`curve_vs_gt.py e308_t_s0`, 5837 pairs with both ends strictly matched; share of pairs with at least 90 % of the GT connector points within 1.5 m): chord tangent 0.871, fitted handle scales 1.000. From the KEYPOINTS (mean position error 1.44 m, 90 % within 2.5 m): chord 0.664, fitted 0.733. So the curve family is right, the scales are worth 13 points (E303), and the end localisation costs another 27 (the E284 head had 1.18 m).
- Not wired: only 3504 of 5798 oracle curves are attached to a delivered piece at both ends (60 %): B does not deliver the lane next to the end. The maneuver measure does not need wiring; the topology scores (topo F1, junction F1) do.

## 4. What it means for the queue
E303 (handle scales: +13 points of the pairs that are found) and everything that moves the ends (E307 narrower end heat, E304 far range, a position offset for the keypoints) are worth more than the pair decision (E302, E298). Day chain 4 was re-ordered that way on 2026-11-12 09:00. Next: score each pair run with `maneuver.py` next to the D79 numbers; promote the measure into `evaluation/` with a self-test.
