# evalsuite/lanes - lane centre lines (picture and graph), common metrics, error causes

A ready-made Task for networks that output lane centre lines. No project knowledge, no torch: numpy + scipy (matplotlib for the views). Overview and use: `../README.md`, design: `../ARCHITECTURE.md` section 8.

| file | content |
|---|---|
| `geometry.py` | `LaneGraph` (polylines in metres, optional successor edges, direction = point order), `resample`, `simplify` (Douglas-Peucker), dense points, point graph, vertex count |
| `raster.py` | `Grid` (cell <-> metres), `rasterize` (graph -> picture), `edt` (distance in metres), `thin` (Zhang-Suen), `skeleton_graph` (picture -> graph: cut at junctions, follow chains, smooth, simplify) |
| `measures.py` | the common measures as ADDITIVE counts of one frame and the formulas for summed counts: picture (strict, tolerant, skeleton, soft Dice, AP), polylines (P / R / F1 at tau, chamfer, lanes found, 1:1 F1, duplicates, direction, vertices), topology (TOPO marbles, junctions) |
| `causes.py` | `error_budget`: every missing and false metre of a frame gets one cause (shift, offset, gap, truncated, missed near / far, duplicate, spur, overshoot, hallucinated, unobservable) with what to improve |
| `task.py` | `LaneConfig`, `LaneData` (picture and graph of both sides, cached per frame), the Metrics `img` / `geo` / `topo`, `LaneTask` (checks, `probe()` tags and causes, guide) |
| `views.py` | `LaneOverviewView`: ground truth, prediction, errors on the lines, errors on the image, error budget, tags |

Adapter contract: `Sample.reference = {"lanes": LaneGraph | dict(pts, edges) | polylines, "mask": bool image (optional), "valid": evaluated cells (optional), "visible": what the input can observe (optional)}`, `Prediction.outputs = {"prob": image, "lanes": graph}` (one of them is enough). Tests: `../selftest_lanes.py` (9 checks, `python -m evalsuite selftest lane`), test bed `../adapters/lanes_synth.py`.
