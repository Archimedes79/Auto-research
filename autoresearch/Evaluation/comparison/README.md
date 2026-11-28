# comparison - where a prediction is compared with the ground truth

| file | what |
|---|---|
| `maps.py` | two pixel maps: confusion counts, iou / precision / recall / f1, the tolerant version for thin structures, `dilate` |
| `boxes.py` | two lists of boxes: matching, true-positive errors, average precision, velocity of moving and static objects |
| `metrics.py` | `MapMetric`, `BoxMetric`: ready-made metrics of the suite on the two files above; you write `extract(sample, pred)` |
| `mutation_check.py` | breaks one rule at a time in a copy of the package and demands that the tests fail |

Not here: parsing a dataset or a network output, thresholding, plotting, and what only one problem knows (range bins, classes, lane graphs).
The files of this folder are part of the BEV evaluator hash: an edit, also of a comment, makes stored BEV evaluations another evaluator's.

## Rules
1. Plain inputs: boolean (or 0 / 1) maps and dicts. Float maps and other integer values are refused: threshold first; "ignore" goes in `valid`.
2. Counts first: the counts of any set of frames add up, numbers are computed from the sum.
3. Nothing to judge is nan (a ratio with denominator 0), never 0 or 1; f1 of a model that predicts nothing is nan: look at recall.
4. Thresholds, tolerances and ranges are arguments. Wrong shapes are `ValueError`. Inputs are never changed.

## What a reviewer should know
- A box is `cx, cy, length, width, yaw` or `radius`, optionally `vx, vy`, a detection also `score`, and `group` (its class). Every function
  of `boxes.py` sees one class. The docstrings give each measure.
- Against the nuScenes devkit (all only at exact ties; without ties the average precision is identical, tested): a distance of exactly the
  threshold is a hit; of two free boxes equally near the later one is taken; equal scores are served in list order, and that order can decide
  the average precision. `ties="pessimistic"` (default of `detection_numbers`) makes it independent of the order, `"list"` is the BEV rule.
- `BoxMetric`'s mean TP errors are plain means over the true positives (not the devkit's curve over recall: not comparable with published
  numbers); a class with ground truth and no true positive counts 1.0. `ase` of rectangles ignores height. `biou` needs shapely.

## Checks
`python -m Evaluation.selftest comparison` (`../selftest_comparison.py`: hand cases, brute force in plain loops, the devkit when installed,
and `_REF`, a frozen copy of the BEV code from before this core). After a change of the core also
`python Evaluation/comparison/mutation_check.py` (slow; add a mutant with every new rule).
