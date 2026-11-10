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

## The class interface: `Evaluation` and `Runner`
For a project with an adapter module (`<project>/evaluation/evalsuite.json`, for BEV `evaluation/bev_eval.py`) the suite is two classes, the pattern of `../README.md`: plain data in and out, every method also a command.
The **Runner** (`runner_api.py`) is the only class that executes the network (D72: it reads no ground truth and computes no loss) and stores what it predicts; the **Evaluation** (`api.py`) calls it,
scores the stored predictions against the ground truth and reads the result (only `draw_frame`, which draws a frame, runs the network again).
```python
from evalsuite import Evaluation, Runner
ev = Evaluation()                                                   # the project is AR_PROJECT
ev.check("runs/t23_C_s1", "small.val")                              # the dry run: what would run, what is stored, the problems
r = ev.run("runs/t23_C_s1", "small.val", gt_key="v8")               # predict what is not stored yet (warm, D70), score it: a dict with the id
ev.show_numbers(r["id"]);  ev.split_numbers(r["id"], "scene");  ev.find_worst(r["id"], "seg", n=5);  ev.get_cost(r["id"]);  ev.report(r["id"]);  ev.make_card(r["id"])
ev.compare(control_id, r["id"])                                     # b - a, paired bootstrap; one network per process: the control was run in another one, compare by ids
p = Runner().run("runs/t23_C_s1", "small.val", taps=("fused",))    # predictions only (frames without labels work too); ev.score(p["results"], "small.val") scores them later
```
```
python -m evalsuite run runs/t23_C_s1 small.val --gt-key v8 --json        # from autoresearch/, AR_PROJECT names the project
python -m evalsuite methods                                               # the methods and their arguments
```
A network with a memory always runs warm (D70): whole scenes in time order from the network's own start; the frames you ask for decide which results are kept and scored. What was predicted is stored once
per identity (network, checkpoint, inputs; `evalsuite/results.py`), so a subset, another ground truth or another metric is scored from the store without the network. The numbers of an Evaluation have no
`loss.*` (a prediction carries no validation loss, D72; `bev_suite.py eval` keeps them). Ids: `<network>@r<hash>` the predictions, `<network>@e<hash>` a scored evaluation; the same question gives the
same id. One network per process (a project's code modules are imported once): run each network in its own process and compare the evaluations by their ids. Frames are a text: `val`, `small.val`,
`spread:1000:large.val`, `random:3[:SEED]:val`, `scenes:a,b:val`, `frames:scene-0061/12:val` (the set is last, and one the project defines). The head of `api.py` has the adapter's functions;
test: `python -m evalsuite.selftest_api` (a toy project, seconds) and, for BEV, `tests/test_evaluation_api.py [--real RUN]`.

### Methods
| method | command line | what it does |
|---|---|---|
| `Runner.check(network, frames, taps)` | `Runner.check NETWORK [FRAMES]` | the dry run of the network: scenes and frames, what is stored, how many frames would run, `problems` |
| `Runner.run(network, frames, taps, ...)` | `Runner.run NETWORK [FRAMES] --taps a,b` | run the network on what is not stored yet and store it (no ground truth), with the intermediate results `taps` when asked (a frame stored without a tap is predicted again to get it); warnings of the memory's health (D70: empty at the scene start, filling after) |
| `Runner.tap(results, frame, name)`, `Runner.list(network)` | `Runner.tap RESULTS FRAME [--name N]`, `Runner.list NETWORK` | which taps a stored frame has, or the file, shape and dtype of one; the stored predictions of a network |
| `Runner.time(network)`, `Runner.probe(network)` | `Runner.time NETWORK`, `Runner.probe NETWORK` | ms per module on an idle GPU, the module probe |
| `Evaluation.check(network, frames, metrics)` | `check NETWORK [FRAMES]` | the dry run of an evaluation: the Runner's plan plus the ground truth and the metrics, the id it would give, `problems` (exit 1 when there are) |
| `score(results, frames, gt_key, metrics)` | `score RESULTS [FRAMES]` | score stored predictions against the ground truth; no network |
| `run(network, frames, gt_key, metrics, ...)` | `run NETWORK [FRAMES] --gt-key v8` | `Runner.run` (the network, then released), then `score` (the labelled data is opened after it): a plain dict (the id, what was done, the warnings) |
| `show_numbers(id, match)`, `split_numbers(id, axis)`, `find_worst(id, match, n)` | `show-numbers ID`, `split-numbers ID`, `find-worst ID` | the numbers, the numbers per scene or frame, where they come from |
| `compare(a, b)` | `compare A B` | paired bootstrap, b - a (a = the control); refused on another GT or other frames |
| `get_cost(id)`, `report(id)`, `make_card(id)` | `get-cost ID`, `report ID`, `make-card ID` | ms per frame, the html page, the metric card in the project's contract (refused when the checkpoint or the evaluator code changed since: score again) |
| `draw_frame(id, frame, view)` | `draw-frame ID FRAME` | one picture (the network runs on that frame with its scene's history) |
| `list_sets()`, `list(network)` | `list-sets`, `Evaluation.list NETWORK` | the sets a project defines, what is stored for a network (predictions and evaluations) |

### Architecture
```
 command line   python -m evalsuite <Class.method> ...        (suitecli.py: every method = one command, --json, exit 0 / 1 / 2)
      |
      v
 Evaluation (api.py)   scores and reads
   run  score  check  show_numbers  split_numbers  find_worst  compare  get_cost  report  make_card  draw_frame  list  list_sets
      |  calls the Runner (injectable: Evaluation(runner=...))                        |  scores through
      v                                                                                v
 Runner (runner_api.py)   the only class that executes the network          engine: Quantitative.run(results=...)
   run  check  tap  list  time  probe          no ground truth, no loss,      -> RunStore  <net>/eval/suite/quant/<e-id>
   always warm (D70)                                                            (additive counts per frame: small)
      |  writes / reads
      v
 ResultSet (results.py)   <net>/eval/results/<r-id>/   frames/<scene>_<i>.npz   taps/<scene>_<i>/<tap>.npy   manifest.json
   one per identity (network, checkpoint, inputs); any question reuses its frames; derived data

 project adapter  <project>/evaluation/evalsuite.json -> bev_eval.py
   required: name out network sets identity source subject task
   optional: free_source + free_subject (together) reference card time probe resolve release inputs MEMORY_TAP COMPARE_IGNORE
   BEV: bev_adapter (frozen), bev_free (the network without labels), bev_suite, bev_sets (the sets = the training's splits/)
```
Data flow of one `Evaluation.run`: frames (a text) -> the Runner runs the scenes up to their last kept frame, whole and warm, stores the kept frames as a ResultSet (the temporary data: only what the
network predicted) and releases the network -> the Evaluation opens the labelled data, scores the frames against the ground truth (`reference`, no inputs) into a RunStore (the counts) ->
`show_numbers` / `compare` / `find_worst` read the counts. The ResultSet is only needed to score again (another ground truth, metric or subset) and to draw; it is derived data (deletable through
`autoresearch/tools/package.py`, D40). A wrong `gt_key` or an unreadable ground truth shows only after the network ran: the predictions are kept, and the message names them (`score` them again).

### Command line
| what | command (from `autoresearch/`, `AR_PROJECT=../bev_fusion`) |
|---|---|
| the dry run | `python -m evalsuite check runs/t23_C_s1 small.val` (the Evaluation's; `Runner.check` is the network's alone) |
| run and score | `python -m evalsuite run runs/t23_C_s1 small.val --gt-key v8 [--metrics maps,boxes,fusion] [--taps fused] [--device cpu]` |
| predict only, no labels | `python -m evalsuite Runner.run runs/t23_C_s1 small.val --taps fused` |
| read | `show-numbers ID [--match M]`, `split-numbers ID [--axis scene]`, `find-worst ID`, `compare A B`, `get-cost ID`, `report ID`, `make-card ID`, `draw-frame ID FRAME` |
| what exists | `list-sets`, `Evaluation.list NETWORK`, `Runner.tap RESULTS FRAME [--name N]`, `methods` |
| every command | a bare word is the Evaluation's method (`Evaluation.list`, `probe` and `selftest` need the class name; every word of the Runner needs `Runner.`); `--json` (nothing else on stdout), `--project`, `--adapter`; exit 0 ok, 1 `check` found problems, 2 any error (`EVALSUITE_TRACEBACK=1` shows the traceback) |

`run` here is not `autoresearch/tools/run.py`: it books no ledger row and takes no GPU slot; a number that feeds a verdict is launched with `run.py ... -- python -m evalsuite run ...`.

## Where the comparison happens
Every comparison of a prediction with the ground truth lives in `comparison/` (read `comparison/README.md`): `maps.py` compares two pixel maps, `boxes.py` two lists of boxes, both on plain numpy arrays and dicts, with counts that add up over frames. `comparison/metrics.py` turns them into ready-made `Metric`s (`MapMetric`, `BoxMetric`): you only write the function that hands over the two things to compare.

## Port it to your network
1. Copy `adapters/minimal.py` anywhere, e.g. `<problem>/evaluation/my_adapter.py`; it imports only `evalsuite`.
2. `Source`: list the scenes and their frames in time order; `load()` gives `inputs` (for the network) and `reference` (for metrics and pictures).
3. `Subject`: load the network; `predict()` returns its outputs.
4. `Task`: one `Metric` per family of numbers, one `overview` `View` (ground truth | prediction | errors).
5. `python -m evalsuite --adapter path/to/my_adapter.py:build selftest-adapter` checks the contracts and says what is wrong. Then `qual random 5` and `quant run`.

## More
`EXTENSIONS.md` covers what a bigger project needs: a network with memory, looking inside the network (taps, torch), outlier details, stored runs, cost and the Pareto front, probing the causes of errors, where the lane-centre-line task went (the lane project), the real adapters and a map of the files. The design notes of the first build are in git history (`autoresearch/history/evalsuite_architecture_2026-10-30.md`, removed 2026-11-10).
