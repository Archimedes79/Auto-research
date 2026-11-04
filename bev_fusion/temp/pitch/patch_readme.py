p = "evaluation/README.md"
s = open(p, encoding="utf-8").read()
s = s.replace('''## Use

```
''', '''## Use

**Quick start (the decision path, four commands):** `bev_suite.py RUN --curves --fusion eval --n 1000 --time` (every number of a run incl. PR / ROC and the fusion-area scoring), `bev_suite.py RUN generalisation` (train against val), `bev_suite.py compare RUN_A RUN_B --curves --fusion` (paired bootstrap, refused on another GT), `bev_suite.py RUN report --vs CONTROL` (the one page). Before you trust a number: `bev_response.py --selftest` and `bev_gtcheck.py --selftest` (the evaluation against known damage, CPU, no model).

```
''', 1)
s = s.replace('''python evaluation/bev_gtcheck.py [--n 24]''', '''python evaluation/bev_suite.py runs/t1_r4_soa --fusion eval --n 1000                    # also the fusion-area scoring `detf` / `detfp` and the coverage split `cov` of the complete GT set (D37 / D38, evaluation/bev_fusion_area.py); the old `det` numbers keep their label
python evaluation/bev_fusion_area.py --coverage [--n 300]                              # no model: how the complete GT set splits into care / partial / don't care and camera / camera + radar / radar only / neither (needs `bev_annotation_pts.py build` once for the radar classes)
python evaluation/bev_gtcheck.py [--n 24]''', 1)
s = s.replace('''| `bev_gtcheck.py` | GT against GT''', '''| `bev_fusion_area.py` | scoring on the FUSION AREA (D37 / D38 first cut, additive: the old numbers keep their label): fusion area = camera x radar fan x not hidden; footprint share f of a box in it; GT classes care (f >= 0.5) / partial (0.1 - 0.5) / don't care; detections with less than half their footprint inside, or on a neutral box, are ignored (two-sided don't care). `BevFusionDetMetric` (`detf` = care boxes, `detfp` = care + partial; the `observed` lidar flag plays no role), `BevCoverageMetric` (`cov`: n boxes, recall ceiling within 2 m and centre error per class of the COMPLETE GT set: all / care / partial / dont_care / camera / cam_radar / cam_only / radar_only / neither), switched on by `--fusion`; `--coverage` prints the split of the GT set | `--selftest` (footprint shares, classes, neutral detections, coverage classes; in `tests/run_gates.sh`), the asserted `detf` / `cov` rows of `bev_response.py`, 4 mutants of `bev_mutants.py` |
| `bev_annotation_pts.py` | token -> (lidar points, radar points) of every nuScenes annotation (1 166 187 rows, built once from `sample_annotation.json`, `data/cache/annotation_pts.npz`, not in git); the label for "a radar return inside the box" (`cam_radar` vs `cam_only`); `build`, `check` (every GT box token of 200 frames is found, `observed` = lidar + radar > 0) | `check` (5298 of 5298 tokens found; numpy's `S16` strips trailing zero bytes, the lookup compares the bytes as two uint64) |
| `bev_gtcheck.py` | GT against GT''', 1)
s = s.replace("(12 monkeypatches: AUC inverted,", "(16 monkeypatches: AUC inverted,")
s = s.replace("moving cells = all cells, camera mask = every cell) and counts the rows", "moving cells = all cells, camera mask = every cell; fusion area: care threshold 0.3, neutral radius 0, detections outside the area kept, hidden cells ignored) and counts the rows")
s = s.replace("(about 8 min CPU, on demand) | all 12 caught |", "(about 10 min CPU, on demand) | all 16 caught (the care threshold by the module selftest, the rest by rows) |")
s = s.replace("and the KNOWN GAP rows (detections on unobserved boxes still count as false, D38) | `--selftest`: 144 numbers incl. the role-swapped rows",
              "the fusion-area rows (`detf` / `detfp` / `cov`: asserted, expected values from counting the annotated boxes by class) and the KNOWN GAP of the old `det` (detections on unobserved boxes still count as false: printed, not asserted) | `--selftest`: 180 numbers incl. the role-swapped rows")
open(p, "w", encoding="utf-8").write(s)
