# E300 Where does the pair head fail? Splits of the end and pair recall by driven / city / lanes / turn / painted / visible / range (D82: "what might be overlooked")

Card of 2026-11-11, CPU analysis on the dump of a head (`runs/<run>/ends_val.pkl`), `evaluation/jsplits.py`. One number per head hides where it fails; the splits show what the next experiment must attack and where the guards of the experiments should look.

## 0. The card (what the tools run)
```json card
{
 "id": "E300",
 "card": "",
 "title": "recall of the true pairs and of the GT ends of a pair head by the attributes of the ground truth: pair driven by the ego or not, city (Boston / Singapore), lanes of the entry arm, turn class, end painted or not, visible or hidden, distance from the ego",
 "kind": "variant",
 "deterministic": true,
 "metric": "pair_recall",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.0, "vs": "c"},
 "arms": {
  "c": {"recipe": "the head of record on the E284 dump (h2, seeds 0 and 1), all pairs"},
  "s": {"recipe": "the same recall inside each split; the largest differences between the splits are the findings (no verdict)"}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [{"metric": "pair_recall", "vs": "c", "min_effect": 0.0}],
 "budget_gpu_h": 0.0,
 "kill": "the splits do not add up to the total (the count of GT pairs differs from `jgt`'s 11083)",
 "evaluate": {"split": "val"}
}
```
By hand (an analysis, no KEEP / CUT; label "by hand: analysis"): the table of the result section is the finding; it is repeated for E287 when its dump exists and for the better head of the later experiments. Command: `jsplits.py score --run e284_h2_s0 e284_h2_s1`.

## 1. Result for the E284 head h2, seeds 0 / 1, STRICT matching (2026-11-12, `jsplits.py score --run e284_h2_s0 e284_h2_s1 --kp-thr 0.5`, VAL 873, 11083 true pairs, 8535 entries, 8114 exits; supersedes section 1b)
The critic's finding: the first version counted a GT end as found when ANY valid keypoint (p > 0.1) lay near it and let one keypoint serve several GT ends; the counts now use the head's own strict one-to-one matching (4 m, 30 degrees, keypoints with p >= 0.5, one keypoint per GT end) and a pair is delivered when the matched keypoint pair has logit >= 0.
| split | pairs n | pair recall seed 0 / 1 | | split | ends n | entry found s0 / s1 | exit found s0 / s1 |
|---|---|---|---|---|---|---|---|
| all | 11083 | 0.289 / 0.294 | | all | 8535 / 8114 | 0.573 / 0.586 | 0.607 / 0.582 |
| city Boston / Singapore | 5292 / 5791 | **0.348 / 0.236** (s1 0.356 / 0.237) | | city Boston / Singapore | 4031 / 4504 | 0.616 / 0.535 (s1 0.643 / 0.536) | |
| driven by the ego yes / no | 1553 / 9530 | **0.408 / 0.270** (s1 0.404 / 0.276) | | range < 20 / 20-35 / > 35 m | 1998 / 1987 / 4550 | **0.709 / 0.660 / 0.476** (s1 0.701 / 0.694 / 0.489) | exit 0.749 / 0.704 / 0.495 |
| range of the entry < 20 / 20-35 / > 35 m | 2687 / 2646 / 5750 | **0.381 / 0.337 / 0.225** (s1 0.370 / 0.323 / 0.245) | | painted yes / no (exits: crossing only) | 3284 / 5251 | 0.623 / 0.542 (s1 0.657 / 0.542) | exit 0.709 / 0.584 (s1 0.617 / 0.574) |
| entries in use in the maneuver 1 / 2 / 3+ | 7523 / 3031 / 529 | 0.287 / 0.291 / 0.308 (s1 0.292 / 0.293 / 0.333) | | visible yes / no | 5489 / 3046 | 0.568 / 0.582 (s1 0.589 / 0.582) | |
| turn: straight / left / right | 7611 / 1772 / 1667 | 0.288 / 0.275 / 0.316 (s1 0.293 / 0.305 / 0.291) | | | | | |
| driven < 20 m / other > 35 m | 695 / 5306 | 0.443 / 0.213 (s1 0.473 / 0.236) | | | | | |
Readings: (a) **the far range is the largest hole**: 52 % of the pairs have their entry beyond 35 m and only 0.225 - 0.245 of them are delivered against 0.37 - 0.38 within 20 m; the ends found fall from 0.70 to 0.48 over the same range; the pixel features are weaker there (E304 weights the far cells, E286 is read by range). (b) **Singapore is 11 points worse than Boston** in pair recall (0.236 against 0.348) and 8 points in entries found, on both seeds: a city-balanced check belongs to every guard (`jmaps_eval.py` has the city split). (c) the pairs the ego drives are found far more often than the others (0.41 against 0.27; driven within 20 m: 0.44 - 0.47): E298 starts from a head that is already better where it matters. (d) **paint now matters**: painted entries are found 8 - 12 points more often than unpainted ones (0.62 - 0.66 against 0.54), the exits with a crossing 3 - 9 points more (the first version's "2 - 3 points" came from the loose matching and from exits that counted a stop line of the other side); visibility still does not matter (0.57 - 0.59 against 0.58): the hidden ends are not harder, the far and the unpainted ones are. (e) **the first version's "more lanes in the arm make the pair easier" (0.32 / 0.38 / 0.40) was an artefact of the loose matching**: with one keypoint per GT end the recall is the same for 1, 2 and 3+ entries in the maneuver (0.29 - 0.33); the left turn is not the hardest class either (0.275 / 0.305 on the two seeds, noise). **What this changes in the plan:** range and city stay the two splits of every report; E304 (far cells) and the Singapore check come before more pair-head design; E302 (capacity) is not justified by a lane-count effect in the recall, only by the degree statistics of E285.

## 1b. (superseded, loose matching) Result for the E284 head h2, seed 0 (2026-11-11, `jsplits.py score --run e284_h2_s0`, VAL 873, 11083 true pairs, 8535 entries, 8114 exits; seed 1 and the E287 heads follow)
Pair recall = a GT pair is delivered when a keypoint of the dump matches its entry and one its exit and the head's logit is >= 0 (matching without the one-to-one rule: 0.341 in total, the strict count of the head itself is 0.29 - 0.31). End found = a valid keypoint (p > 0.1) matches the GT end (a recall of the extraction, not of the 0.5 decision).
| split | pairs n | pair recall | | split | ends n | entry found / exit found |
|---|---|---|---|---|---|---|
| all | 11083 | 0.341 | | all | 8535 / 8114 | 0.790 / 0.808 |
| city: Boston / Singapore | 5292 / 5791 | **0.400 / 0.288** | | city: Boston / Singapore | 4031 / 4504 | 0.812 / 0.770 (exit 0.860 / 0.764) |
| driven by the ego: yes / no | 1554 / 9529 | **0.449 / 0.324** | | distance from the ego < 20 / 20-35 / > 35 m | 1998 / 1987 / 4550 | **0.869 / 0.843 / 0.732** (exit 0.893 / 0.854 / 0.748) |
| lanes of the entry arm 1 / 2 / 3+ | 7523 / 3031 / 529 | 0.321 / 0.381 / 0.397 | | painted yes / no | 3284 / 5251 | 0.806 / 0.780 (exit 0.852 / 0.791) |
| turn: straight / left / right | 7611 / 1772 / 1667 | 0.355 / 0.300 / 0.331 | | visible yes / no | 5489 / 3046 | 0.797 / 0.777 (exit 0.821 / 0.782) |
Readings: (a) **the far range is the largest hole in the ends**: 53 % of the GT entries lie farther than 35 m from the ego (the window is 100 m) and only 73 % of them are found against 87 % within 20 m; the pixel features are weaker there (E286 has to be read by range); (b) **Singapore is 11 points worse than Boston** in pair recall (0.288 against 0.400; left-hand traffic and other layouts; Singapore has more scenes, 4504 entries against 4031): a city-balanced check belongs to every guard; (c) the pairs the ego drives are found far more often than the others (0.449 against 0.324): the head already favours what the camera sees ahead, and the driven-weight experiment (E298) starts from a head that is better where it matters; (d) paint and visibility matter little for the ENDS (0.806 / 0.780 painted, 0.797 / 0.777 visible: 2 - 3 points): the occlusion and the missing paint are not what limits the ends, the range is; (e) more lanes in the arm make the pair easier (0.32 / 0.38 / 0.40), the left turn is the hardest class (0.300). **What this changes in the plan:** the evaluation reports range and city for every head (the E286 scorer gets a range split, `jsplits.py` already has it), and the far range is a lever of its own (a larger context / the 4 m attention arm of E286, higher weight of the far cells, a 100 m pyramid level) beyond the pair head.
