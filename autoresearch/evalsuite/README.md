# evalsuite - look at what a network does: pictures, numbers and cost

You write three small classes for your network. The suite then gives you **pictures** (a frame with ground truth, prediction and the errors; random frames; result videos), **numbers** (the metrics over all videos, which scenes and frames drag them down, whether run B is really better than run A) and **cost** (ms and operations per frame, and where each run sits on the quality-against-cost curve).

Needs numpy and matplotlib; videos are H.264 with an ffmpeg that has libx264 ($FFMPEG, the one-line file ~/.evalsuite_ffmpeg with this machine's path, the PATH or imageio-ffmpeg; an animated GIF without), the full selftest also needs scipy. The suite imports no torch and knows no dataset. Run it from `autoresearch/`.

## Try it
The whole adapter is `adapters/minimal.py` (about 70 lines): a moving disc in noise, a thresholding "network", one metric.
```
python -m evalsuite --adapter evalsuite.adapters.minimal:build --out ../eval_out qual random 3
python -m evalsuite --adapter evalsuite.adapters.minimal:build --out ../eval_out quant run --name a
python -m evalsuite --adapter evalsuite.adapters.minimal:build --out ../eval_out quant table a
python -m evalsuite selftest minimal          # checks the same against known answers
```

## The three boxes
| box | you say | what you write |
|---|---|---|
| `Source` | which scenes ("videos") and frames exist, how to load one | `scenes()`, `frames(scene)`, `load(ref)` -> `Sample(inputs, reference)` |
| `Subject` | how to run the network on one frame | `predict(sample)` -> `Prediction(outputs)` |
| `Task` | what to measure, how to draw | `metrics()`, `views()` with one `overview` |

One rule makes the rest work: a `Metric.frame()` returns **additive counts** (true / false positives, sums, ...) and `Metric.reduce()` turns the counts of any set of frames into numbers. A dataset number is therefore a sum over frames, never a mean of per-frame numbers, and it can be recomputed for one scene, without one scene or for a bootstrap resample without the network. A `View` returns a `Page` of `Panel`s (data, no plotting code). `build()` returns `(Source, Subject, Task)`.

## What you get
A frame is named `scene/index` (`demo-1/3`; the files and tables write `demo-1/003`). `--arg key=value` goes to `build()`.

| you want | command | you get |
|---|---|---|
| one frame | `qual frame demo-1/3` | png: ground truth, prediction, errors (green tint = false positive, blue tint = missed, grey = right), the frame's own scores in the header |
| random frames | `qual random 6 --seed 1` | a page: first a random video, then a random frame in it, n times |
| result videos | `qual videos 2` | mp4 per video + a strip of the per-frame scores |
| numbers over all videos | `quant run --name a`, `quant table a` | the counts of every frame stored; the dataset-level numbers; ms per frame of the run |
| where the numbers come from | `quant outliers a`, `quant report a` | the scenes and frames that drag the numbers down, each with the command that draws it |
| is B better than A | `quant compare a b` | paired scene bootstrap of the headline numbers, b - a = variant - control (refused when the runs differ in source or task) |
| how fast, how big | `quant timing --run a`, `Subject.ops()` | ms per frame (idle benchmark) and GMACs stored with the run |
| position on the Pareto curve | `quant pareto a b c` | quality against ms and GMACs, the runs that no other run beats on both |

From Python:
```python
from evalsuite import EvalSuite
suite = EvalSuite(source, subject, task, out="eval_out")
suite.qual.frame("demo-1/3");  suite.qual.random(3, seed=1);  suite.qual.videos(1)
run = suite.quant.run(name="a");  suite.quant.table(run);  suite.quant.outliers(run);  suite.quant.compare("a", "b")
suite.quant.timing(run="a");  suite.quant.pareto(["a", "b", "c"])
```

## The class interface: `Evaluation(project)`
For a project with an adapter module (`<project>/evaluation/evalsuite.json`, for BEV `evaluation/bev_eval.py`) the whole suite is one class, the pattern of `../INTERFACE.md`: plain data in and out, every method also a command.
```python
from evalsuite import Evaluation
ev = Evaluation()                                                   # the project is AR_PROJECT
ev.check("runs/t23_C_s1", "small.val")                              # the dry run: what would run, what is stored, the problems
r = ev.run("runs/t23_C_s1", "small.val", gt_key="v8")               # predict what is not stored yet (warm, D70), score it: a Result
r.quant();  r.by("scene");  r.worst("seg", n=5);  r.cost();  r.report();  r.card()
ev.compare(control_id, r["id"])                                     # b - a, paired bootstrap; one network per process: the control was run in another one, compare by ids
```
```
python -m evalsuite Evaluation.run runs/t23_C_s1 small.val --gt-key v8 --json        # from autoresearch/, AR_PROJECT names the project
python -m evalsuite methods                                                           # the methods and their arguments
```
Only `predict` / `run` (and `time`, `probe`, `qual_frame`) execute the network; what it predicted is stored once per identity (network, checkpoint, inputs) and every later question, a subset,
another ground truth, another metric, is scored from the store (`evalsuite/results.py`). The frames are a text (`val`, `spread:1000:large.val`, `scenes:a,b:val`, ...). Details and the adapter's
functions: the head of `evalsuite/api.py`; the methods: `../INTERFACE.md`; test: `python -m evalsuite.selftest_api`.

## Where the comparison happens
Every comparison of a prediction with the ground truth lives in `comparison/` (read `comparison/README.md`): `maps.py` compares two pixel maps, `boxes.py` two lists of boxes, both on plain numpy arrays and dicts, with counts that add up over frames. `comparison/metrics.py` turns them into ready-made `Metric`s (`MapMetric`, `BoxMetric`): you only write the function that hands over the two things to compare.

## Port it to your network
1. Copy `adapters/minimal.py` anywhere, e.g. `<problem>/evaluation/my_adapter.py`; it imports only `evalsuite`.
2. `Source`: list the scenes and their frames in time order; `load()` gives `inputs` (for the network) and `reference` (for metrics and pictures).
3. `Subject`: load the network; `predict()` returns its outputs.
4. `Task`: one `Metric` per family of numbers, one `overview` `View` (ground truth | prediction | errors).
5. `python -m evalsuite --adapter path/to/my_adapter.py:build selftest-adapter` checks the contracts and says what is wrong. Then `qual random 5` and `quant run`.

## More
`EXTENSIONS.md` covers what a bigger project needs: a network with memory, looking inside the network (taps, torch), outlier details, stored runs, cost and the Pareto front, probing the causes of errors, where the lane-centre-line task went (the lane project), the real adapters and a map of the files. The design notes of the first build are in `../history/evalsuite_architecture_2026-10-30.md`.
