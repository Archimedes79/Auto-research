# evaluation/lane_task - the lane-centre-line Task of the evalsuite (picture and graph), common metrics, error causes

A ready-made `Task` of the evalsuite (`../../../autoresearch/evalsuite/`) for networks that output lane centre lines; `../lane_suite.py` is its adapter on the project's VAL data. No torch: numpy + scipy (matplotlib for the views). It lived in the evalsuite of autoresearch (package lanes) until 2026-10-31 and moved here with D24 (problem-specific code out of the problem-independent `autoresearch/`); `import lane_task` puts the evalsuite on the path itself. Design history: `../../../autoresearch/history/evalsuite_architecture_2026-10-30.md` section 8.

Use (the `img` metric scores the picture, `geo` and `topo` the graph; one output is scored at both levels in the same run):
```
from lane_task import LaneConfig, Grid, LaneTask
task = LaneTask(LaneConfig(Grid(u0=-50, v0=-50, res=0.5, nu=200, nv=200)))
```

| file | content |
|---|---|
| `geometry.py` | `LaneGraph` (polylines in metres, optional successor edges, direction = point order), `resample`, `simplify` (Douglas-Peucker), dense points, point graph, vertex count |
| `raster.py` | `Grid` (cell <-> metres), `rasterize` (graph -> picture), `edt` (distance in metres), `thin` (Zhang-Suen), `skeleton_graph` (picture -> graph: cut at junctions, follow chains, smooth, simplify) |
| `measures.py` | the common measures as ADDITIVE counts of one frame and the formulas for summed counts: picture (strict, tolerant, skeleton, soft Dice, AP), polylines (P / R / F1 at tau, chamfer, lanes found, 1:1 F1, duplicates, direction, vertices), topology (TOPO marbles, junctions) |
| `causes.py` | `error_budget`: every missing and false metre of a frame gets one cause (shift, offset, gap, truncated, missed near / far, duplicate, spur, overshoot, hallucinated, unobservable) with what to improve; `target_of` = the place to work (model / code / evaluation) that `LaneTask.causes()` hands to the probing |
| `synth.py` | the synthetic lane world (scenes with every error cause planted; the Subject emits a graph, a picture or both): test bed and porting template (was the evalsuite adapter lanes_synth.py) |
| `task.py` | `LaneConfig`, `LaneData` (picture and graph of both sides, cached per frame together with the frame it belongs to: a constant prediction object of a null baseline cannot carry one frame's numbers into the next), the Metrics `img` / `geo` / `topo`, `LaneTask` (checks, `probe()` tags and causes, guide) |
| `views.py` | `LaneOverviewView`: ground truth, prediction, errors on the lines, errors on the image, error budget, tags |

Adapter contract: `Sample.reference = {"lanes": LaneGraph | dict(pts, edges) | polylines, "mask": bool image (optional), "valid": evaluated cells (optional), "visible": what the input can observe (optional)}`, `Prediction.outputs = {"prob": image, "lanes": graph}` (one of them is enough). Tests: `../tests/test_lane_task.py` (10 checks, CPU: `wsl bash tools/env/_py14.sh test_lane_task.py [word ...]` from the problem folder), test bed `synth.py`.
