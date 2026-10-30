# evalsuite - qualitative and quantitative evaluation of any network

One core, two suites, three small boxes to plug in a network. Design and porting guide: `ARCHITECTURE.md`. Needs numpy and matplotlib (+ PIL, cv2 for videos, ffmpeg with libx264 optional); the core never imports torch or a dataset. Known-answer self-test of everything: `python -m evalsuite selftest` (29 checks, about 1 minute, run from `autoresearch/`). Adapters in use: `adapters/toy.py` (template and test bed) and `bev_fusion/evaluation/bev_adapter.py` (camera + radar BEV system, 17 contract checks pass on a real run).

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
```

## The three boxes (details in `contracts.py`)

- **Source**: `scenes()`, `frames(scene)`, `load(ref) -> Sample(inputs, reference, display)`.
- **Subject**: `predict(sample, taps) -> Prediction(outputs, taps)`, `reset()`, `taps() -> [TapSpec]`, `warmup`. Stateful along a scene. For torch networks `torch_taps.TapRecorder` records stages by module path.
- **Task**: `metrics()` (each `Metric`: `frame()` = additive counts, `reduce()` = numbers of any set of frames, `score()` = per-frame scalars), `views()` (`overview` required; a `View` returns a `Page` of `Panel`s, no plotting code), `checks()`.

Reading the pictures: grey = true positive, RED = false positive, BLUE = false negative, cream = not evaluated, in every adapter. Each tap panel shows what it should look like under the picture.

## Files

`contracts.py` boxes and data | `suite.py` EvalSuite | `runner.py` order of calls, state | `sampling.py` video-then-frame draws | `store.py` run store | `stats.py` robust z, leave-one-out, bootstrap | `qualitative.py` | `quantitative.py` | `check.py` adapter contract checker | `torch_taps.py` | `cli.py` | `render/` figure engine, taps, html, video | `adapters/toy.py` test bed and porting template | `selftest.py`.
