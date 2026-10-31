# evalsuite - look at what a network does: pictures, numbers and cost

You write three small classes for your network. The suite then gives you **pictures** (a frame with ground truth, prediction and the errors; random frames; result videos), **numbers** (the metrics over all videos, which scenes and frames drag them down, whether run B is really better than run A) and **cost** (ms and operations per frame, and where each run sits on the quality-against-cost curve).

Needs numpy and matplotlib; videos are H.264 with an ffmpeg that has libx264 (an animated GIF without), the lane part and the full selftest also need scipy. The suite imports no torch and knows no dataset. Run it from `autoresearch/`.

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
| one frame | `qual frame demo-1/3` | png: ground truth, prediction, errors (red = false positive, blue = missed, grey = right), the frame's own scores in the header |
| random frames | `qual random 6 --seed 1` | a page: first a random video, then a random frame in it, n times |
| result videos | `qual videos 2` | mp4 per video + a strip of the per-frame scores |
| numbers over all videos | `quant run --name a`, `quant table a` | the counts of every frame stored; the dataset-level numbers; ms per frame of the run |
| where the numbers come from | `quant outliers a`, `quant report a` | the scenes and frames that drag the numbers down, each with the command that draws it |
| is B better than A | `quant compare a b` | paired scene bootstrap of the headline numbers |
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

## Port it to your network
1. Copy `adapters/minimal.py` anywhere, e.g. `<problem>/evaluation/my_adapter.py`; it imports only `evalsuite`.
2. `Source`: list the scenes and their frames in time order; `load()` gives `inputs` (for the network) and `reference` (for metrics and pictures).
3. `Subject`: load the network; `predict()` returns its outputs.
4. `Task`: one `Metric` per family of numbers, one `overview` `View` (ground truth | prediction | errors).
5. `python -m evalsuite --adapter path/to/my_adapter.py:build selftest-adapter` checks the contracts and says what is wrong. Then `qual random 5` and `quant run`.

## More
`EXTENSIONS.md` covers what a bigger project needs: a network with memory, looking inside the network (taps, torch), outlier details, stored runs, cost and the Pareto front, probing the causes of errors, the ready-made lane-centre-line task, the real adapters and a map of the files. The design notes of the first build are in `../history/evalsuite_architecture_2026-10-30.md`.
