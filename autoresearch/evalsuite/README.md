# evalsuite - qualitative and quantitative evaluation of any network

One core, two suites, three small boxes to plug in a network. Design and porting guide: `ARCHITECTURE.md`. Needs numpy and matplotlib (+ PIL, cv2 for videos, ffmpeg with libx264 optional); the core never imports torch or a dataset. Known-answer self-test of everything: `python -m evalsuite selftest [word]` (38 checks, about 2 minutes, run from `autoresearch/`; scipy needed for the lane part). Adapters in use: `adapters/toy.py` (template and test bed), `adapters/lanes_synth.py` (synthetic road world with every error cause planted: test bed of the lane task and the probing), `bev_fusion/evaluation/bev_adapter.py` (camera + radar BEV system, 17 contract checks pass on a real run) and `bev_fusion_lane_center/evaluation/lane_suite.py` (lane-centre-line data of VAL: dense heads and registry graphs).

## What you get

| you want | command (after the adapter, see "Run it") | result |
|---|---|---|
| look at ONE frame, results over ground truth, errors highlighted | `qual frame scene-0061/12` | `<out>/qualitative/frames/<scene>_<idx>_overview.png` (header: frame label + the frame's own scores) |
| the same with the inside of the network | `qual debug scene-0061/12 [--taps a b]` | + `_debug.png` (one picture per tapped stage, statistics, plausibility flags) and `_taps.npz` (raw arrays) |
| n random frames: a random video first, then a random frame in it | `qual random 6 --seed 1` | `<out>/qualitative/random_s1_n6/index.html` + pictures + `draws.json` |
| result videos of m random sequences | `qual videos 2 --seed 1` | `<out>/qualitative/videos/<scene>_<view>.mp4` (H.264) + per-frame score strip |
| numbers over all videos | `quant run [--n-scenes 20 \| --spread 500]` then `quant table RUN` | `<out>/quant/<run>/`: counts of every frame; dataset-level numbers |
| where do the numbers come from | `quant outliers RUN` / `quant report RUN` | scenes that drag the headline numbers (exact leave-one-out), frames furthest outside the others (robust z) with their own drag, plausibility flags; every row says which frame and the command that draws it; `report.html` embeds the worst frames |
| draw such a frame | `quant trace RUN scene-0061/012` | the exact replay of the run's state, overview + debug |
| is run B better than run A | `quant compare A B` | paired scene bootstrap of the headline numbers on the frames both have (needs no network) |
| how fast | `quant timing` | ms per frame (idle machine only; the load is recorded) |

| where do the errors come from | `probe scan RUN` / `probe budget RUN` / `probe situations RUN` / `probe sample RUN "dominated by fn.gap" -n 6` | `<out>/probe/<run>_s<seed>/index.html`: the error split by cause (what it is, whether to fix the model, the code or the evaluation), the situations that cost more than their size, k random examples of each (a video first, then a frame), each with the command that opens it |

A frame is named `scene/index` (`scene-0061/12`, the index is the position in the video) or by the dataset's own key. `--state carry` (default) runs a video in order with the network's memory carried from frame to frame, `--state reset` makes every frame independent, `--state warmup` starts every frame from empty and feeds only the history the network needs (any subset of frames); the difference of two such runs (`quant compare`) is the test whether a memory helps. `quant table RUN` prints the headline numbers of every metric, `--all` every number. Quick single-frame looks feed the history the network needs (`--history warmup`); `--history full` replays the whole video up to the frame (what `quant trace` does).

## Run it

```
cd autoresearch                              # the folder that contains evalsuite/
python -m evalsuite --adapter evalsuite.adapters.toy:build --out eval_out qual random 4      # the toy network, no setup
python -m evalsuite --adapter evalsuite.adapters.toy:build --out eval_out quant run --name t1
python -m evalsuite --adapter evalsuite.adapters.toy:build --out eval_out selftest-adapter    # the contract checker
```
`--arg key=value` goes to the adapter's `build(**kv)`. Problem entries: bev_fusion: `evaluation/bev_suite.py` (`bev_fusion/evaluation/README.md`).

## From Python

```python
from evalsuite import EvalSuite
suite = EvalSuite(source, subject, task, out="eval_out", state="carry")
suite.qual.frame("scene-0061/12", views=("overview", "debug"));  suite.qual.random(6, seed=1);  suite.qual.videos(2)
run = suite.quant.run(n_scenes=20)
suite.quant.table(run);  suite.quant.outliers(run);  suite.quant.report(run);  suite.quant.compare("runA", "runB")
suite.probe.scan(run, k=3, top=6);  suite.probe.budget(run);  suite.probe.sample(run, "dominated by fn.gap", 6)      # needs Task.probe()
```

## The three boxes (details in `contracts.py`)

- **Source**: `scenes()`, `frames(scene)`, `load(ref) -> Sample(inputs, reference, display)`.
- **Subject**: `predict(sample, taps) -> Prediction(outputs, taps)`, `reset()`, `taps() -> [TapSpec]`, `warmup`. Stateful along a scene. For torch networks `torch_taps.TapRecorder` records stages by module path.
- **Task**: `metrics()` (each `Metric`: `frame()` = additive counts, `reduce()` = numbers of any set of frames, `score()` = per-frame scalars), `views()` (`overview` required; a `View` returns a `Page` of `Panel`s, no plotting code), `checks()`.

## Lane centre lines, picture first, then sparse (`lanes/`)

`LaneTask` (in `evalsuite/lanes/`) is a ready-made Task for any network that outputs lane centre lines, as a PICTURE (a probability map of a dense head) or as a GRAPH (polylines with successor edges). The adapter delivers `Sample.reference = {"lanes": LaneGraph | polylines, "mask": bool image (optional)}` and `Prediction.outputs = {"prob": image and / or "lanes": ...}`; everything else is shared:

| metric | level | what it is |
|---|---|---|
| `img` | picture | strict IoU / Dice; tolerant (relaxed) precision, recall, F1 and quality at 0.5 / 1.0 / 1.5 m; the same on the thinned prediction (`skel`); soft Dice; threshold-free tolerant average precision, best F1 and its threshold |
| `geo` | polylines (a dense head's picture is thinned and traced into a graph first: `skeleton_graph`) | P / R / F1 at 0.5 / 1.0 / 1.5 m, chamfer (accuracy, completeness, clipped), lanes found, one-to-one F1 and duplicate share, direction accuracy, vertices per 100 m, length ratio |
| `topo` | connectivity | TOPO marbles (what can be reached along the graph within 30 m: precision, recall, F1) and junction F1 (forks / merges, 3 m) |

So one output is scored as a picture and then, with the same run, as a graph: `quant table RUN` shows both levels next to each other. The views draw ground truth, prediction (dots = vertices, arrows = direction), the errors on the lines (blue = missed, red = false), the errors on the image and the error budget of the frame.

**Probing** (`probing.py`, any Task that implements `probe()`; `LaneTask` does): per frame the lane task stores descriptors (`tag.*`: lanes, length, fork, curviness, share beyond 40 m, empty prediction ...) and the error assigned to causes (`cause.*`, metres of lane): missed lane near / far, hole in a lane (gap), prediction stops short (truncated), displaced beyond tolerance (offset: a PARALLEL neighbour within 3 tau; a lane that merely crosses does not count), the whole prediction shifted (a best-shift search over 2 m), duplicate, stray piece (spur), overshoot past the end of the truth, invented lane (hallucinated), outside the input's view (unobservable, with a mask). The pieces add up to the frame's error, so `probe budget` is an exact split of the error of the whole run, `probe situations` lists the groups of frames that cost more than their size, `probe sample` draws random frames of one of them, `probe scan` does all of it as one page. Each cause says what to improve: the model, the code (assembly: linking, pruning, de-duplication) or the evaluation (mask the unobservable, ground truth missing there?).

Reading the pictures: grey = true positive, RED = false positive, BLUE = false negative, cream = not evaluated, in every adapter. Each tap panel shows what it should look like under the picture.

## Files

`contracts.py` boxes and data | `suite.py` EvalSuite | `runner.py` order of calls, state | `sampling.py` video-then-frame draws | `store.py` run store | `stats.py` robust z, leave-one-out, bootstrap | `qualitative.py` | `quantitative.py` | `check.py` adapter contract checker | `torch_taps.py` | `cli.py` | `probing.py` budget / situations / sample / scan | `lanes/` lane-centre-line library (geometry, raster, measures, causes, task, views) | `render/` figure engine, taps, html, video | `adapters/toy.py` test bed and porting template, `adapters/lanes_synth.py` lane test bed | `selftest.py` (+ `selftest_lanes.py`).
