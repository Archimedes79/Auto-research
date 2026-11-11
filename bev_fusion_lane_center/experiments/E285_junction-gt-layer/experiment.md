# E285 The junction ground truth of the lane head: zone, conflict core, arms, ranked entry / exit points, order-preserving pairs, painted evidence (D82)

Card of 2026-11-11, written by the lane chat on Mirko's word (D82: "Stelle erstmal eine sehr gute GT her, guck dir die beispielhaft an. Sie sollte alles enthalten."). CPU only, derived data of the lane project (D80): the lane GT of GtGenerator (store `0702a065c6` = l4) is read, nothing of GtGenerator is edited, no store signature changes; the GT chat's prepared l6 (`prep/v13-l6`) is not merged by this card.

## 0. The card (what the tools run)
```json card
{
 "id": "E285",
 "card": "",
 "title": "junction GT layer on the lane side (training/grid/jgt.py, qjunction.py, data/prep/junction_paint.py): the junction zone and the conflict core, arms with lane ranks, entry / exit points, the order-preserving pairs, and the painted stop lines / crossings of the map expansion",
 "kind": "block",
 "deterministic": true,
 "metric": "order_rule_share",
 "direction": "higher",
 "band": 0.02,
 "prediction": {"value": 0.97, "vs": "c"},
 "arms": {
  "c": {"recipe": "the monotone (non-crossing) share of the multi-pair maneuvers that the lane GT itself holds, counted by `jgt.junction_gt` on VAL: the rule of D82 is a property of the data or it is not"},
  "j": {"recipe": "the layer's invariants on VAL: every connector run lies in a zone, every end in an arm, the true pairs equal `qends.end_table`'s, the maps decode back to the ends within 1 m"}
 },
 "control": "c",
 "variant": "j",
 "pairing": "position",
 "seeds": [0],
 "rule": [{"metric": "order_rule_share", "vs": "c", "min_effect": 0.0}],
 "budget_gpu_h": 0.0,
 "kill": "the layer is not a pure function of the lane GT, a frame fails to load, or the invariants (section 2) fail on more than 1 % of the VAL junction frames",
 "evaluate": {"split": "val"}
}
```
By hand (a layer has no KEEP / CUT; the gates of section 2 are the verdict, labelled "by hand: gates"): the layer is accepted when the gates pass, the examples were looked at (section 3) and the defects found there are written down.

## 1. What the layer is (all by location from the lane GT, nothing predicted)
- **Zone** = the connector runs (the parts of the lane pieces that lie on `connector` map pieces, at least 3 points) grown by 1.8 m, closed by 2 m: the area the connectors really sweep (not the nuScenes road-segment polygons). **Conflict core** = where connector paths of DIFFERENT (entry, exit) pairs cross, merge or diverge (overlap of the connectors grown by 0.9 m, grown by 1 m).
- **Ends** as `qends` (entry = first point of a connector run, exit = last, clustered 1.5 m / 40 degrees). **Arms**: single-linkage groups of ends (heading within 25 degrees, at most 6 m along and across); an entry arm and an exit arm with opposite headings within 14 m are one arm (the two directions of a road). Per arm: centre, heading, number of entries and exits (n_in, n_out), spread of the ends along the heading. **Rank** of an end in its arm: 0 = leftmost looking along the travel direction.
- **Pairs** (entry, exit): arm in, arm out, ranks, number of entries / exits of that arm pair in use, turn angle, `driven` (tier 1: the ego drove it, non-causal label). **Order rule (D82):** within one (arm in, arm out) maneuver the pairs are monotone: entry rank i before i' implies exit rank j <= j'; where k entries meet k exits the solution is entry i to exit i (identity).
- **Painted evidence** (`data/prep/junction_paint.py`, nuScenes map expansion, lane frame, VAL / TRAIN only): stop lines with their type (turn stop, pedestrian crossing, traffic light, stop sign, yield) and pedestrian crossings.
- **Maps** (`qjunction.junction_targets`): zone + core (1 m), entry / exit heat (Gaussian sigma 1.5 m) with heading, stop line / crossing masks (0.5 m).

## 2. Gates (VAL, every third keyframe = 301 frames for the measures below; the full 873 for the invariants when the layer is used in training)
1. self-tests of `jgt.py`, `qjunction.py` (a plus junction, the order rule, the fields); 2. the order rule: monotone share of the multi-pair maneuvers >= 0.95 (prediction 0.97); 3. arms: spread of the entries along the arm heading below 3 m for >= 0.95 of the arms (else a cut line per arm is needed); 4. every junction frame has zone, ends, arms; 5. the maps decode back: peaks of the entry / exit heat within 1 m of the GT ends for >= 0.99.

## 3. Examples looked at (2026-11-11, `tools/viz/qgraph_view.py frame ... --pred oracle`, scenes 0221 / 0096 / 0017 / 0100)
The nuScenes intersection areas are large rectangles of the road segments; stops and exits sit at the lane ends of the map, grouped per arm; the connectors run through them as long curves. In all four frames the intersection mask covers the approach lanes' dividers too (green lines inside the grey area), and scene 0017 shows polygons without any connector (lower approach, upper left block).

## 4. Measures taken before the card was finished (VAL, 301 frames, every third keyframe; `temp/gt2/jstats*.py`, scratch)
| measure | value |
|---|---|
| junction frames | 284 of 301 |
| order rule: maneuvers with >= 2 pairs / monotone / exact identity | 659 / 659 / 435 (66 %: the others are monotone with k_in != k_out, lane add / drop) |
| spread of entries along the arm heading (p10 / p50 / p90) | 0.0 / 0.0 / 0.98 m: no cut line needed; 0 % of the arms above 3 m |
| lanes per arm (entry side, 1 / 2 / 3 / 4): arms | 1425 / 506 / 147 / 20 |
| zone area vs nuScenes polygon with connectors (p50) | 174 vs 159 m^2: the polygons are not inflated against the connectors; **the intersection covers a median 27 % (p90 41 %) of the road area of a frame** (154 frames) |
| nuScenes polygons without a connector (id None) | 3.3 % of the intersection area, in 11 % of the frames |
| conflict core / zone | p10 0.21, p50 0.38, p90 0.55: the real conflict region is about 40 % of the zone |
| natural piece ends without an event within 1 m | 0 of 5755 (the GT's flags are complete; the "25 - 33 % flag defect" of E283 is the delivered head's, not the GT's) |
| entries within 3 m / 6 m of a painted stop line | 43 % / 57 % (median distance 3.8 m); within 4 m of a crossing 21 %; exits within 4 m of a crossing 18 % |
| pairs the ego drove (tier 1) | 14 % |
| degree of the ends in the true pairs | entries (2958): out-degree 1 / 2 / 3 / 4 = 2153 / 741 / 50 / 14 (27 % above 1, mean 1.30); exits (2815): in-degree 1 / 2 / 3 / 5 = 1895 / 830 / 82 / 8 (33 % above 1, mean 1.36): a one-to-one transport (Sinkhorn with unit masses) is wrong for about a third of the ends |
Readings: (a) the order rule holds on every multi-pair maneuver of the sample (the prediction 0.97 is beaten); (b) the ends of an arm are already aligned, so no snapping is needed; (c) the GT zone is as large as the connectors sweep, but only about 40 % of it is a conflict region: the new core map gives the head the tighter definition; (d) painted evidence exists for about half of the entries only, so "visible against inferred" (D82 / E283) is a real split: an entry with a stop line within 3 m is `painted`.

## 5. Open
The full-split numbers (873 frames) and the decode gate (5) with the training loader; a per-end `painted` attribute in the targets; the SD road level with lane counts (E290). Pictures of the layer for the report: `tools/viz/junction_view.py` (to write).

## 6. Result of the gates (2026-11-11, VAL, all 873 frames, `temp/gt2/gates_e285.py`, scratch; verdict "by hand: gates")
**The layer is accepted** (all gates of section 2 hold): 826 of 873 frames have a junction, no frame broke the layer (0 errors); (2) order rule: **1899 of 1899** maneuvers with at least two pairs are monotone (prediction 0.97 beaten), 1260 (66 %) are the exact identity; (3) arms: 1 of 7848 arms has a spread of its entries above 3 m; (4) the true pairs of `jgt` equal those of `qends.end_table` on every frame (0 mismatches, 8535 entries, 8114 exits, 8962 maneuvers); (5) the maps decode back: the peaks of the entry / exit heat lie within 1 m of the GT end for 99.6 % / 99.7 % of the ends (8498 of 8535, 8093 of 8114). Open (not gates): the zones at the window border are fragments (a connector that starts outside the window); arms of one road split when the ends are more than 6 m apart along the heading (long connectors); the pictures in the report are made by `tools/viz/junction_view.py`; the per-end `painted` attribute (stop line within 3 m) is not in the targets yet.

## 7. The `painted` attribute (2026-11-11, `qjunction.painted_flags`, VAL all 873 frames, `temp/gt2/jpainted.py`)
An ENTRY is PAINTED when a stop line of a type other than the turn stop lies within 3 m, or a pedestrian crossing within 4 m (turn stops are waiting points inside the junction, not entries); an EXIT when a pedestrian crossing lies within 4 m (corrected 2026-11-12 after the critic's finding: the stop line of the arm's own entry side lies next to its exits and made them look painted; first version 28.3 % of the exits). Entries painted 38.5 % (3284 of 8535; stop line alone 34.9 %, crossing alone 21.2 %), exits painted 18.5 % (1503 of 8114, crossing only): **about 61 % of the entries and 81 % of the exits have no paint to see and must be inferred from context** (the first number of section 4, 43 % within 3 m of any stop line, counted the turn stops too). `jmaps_eval.py` reports the recall of the head and of B on painted and unpainted ends separately.
