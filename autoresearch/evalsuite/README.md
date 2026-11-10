# evalsuite - the Evaluation of a network: one class over the Runner, a command line on top

What a network does, as **numbers** (the metrics over all videos, which scenes and frames drag them down, whether run B is really better than run A), **pictures** (a frame with ground truth, prediction and the errors) and **cost** (ms per frame). The front door is the class `Evaluation`: it holds every function and reaches the network only through the `Runner` below it; the command line sits on top and is generated from the class. Underneath is the engine `EvalSuite` (Source, Subject, Task) that does the counting and drawing; a project connects its network through an adapter module.

Needs numpy and matplotlib; videos are H.264 with an ffmpeg that has libx264 ($FFMPEG, the one-line file ~/.evalsuite_ffmpeg with this machine's path, the PATH or imageio-ffmpeg; an animated GIF without), the full selftest also needs scipy. The core imports no torch and knows no dataset. Run it from `autoresearch/`.

## Layers
```
 command line      python -m evalsuite <method> ...       (suitecli.py: every method = one command, --json, exit 0 / 1 / 2)
        |
        v
 Evaluation (api.py)  -- the front door: scores and reads -------------------------------.
   run  score  check  show_numbers  split_numbers  find_worst  compare  get_cost  report   |  scores through
   make_card  draw_frame  list  list_sets  |  tap  time  probe (passed to the Runner)      v
        |  calls the layer below                                                    engine EvalSuite: Quantitative.run(results=...)
        v                                                                             -> RunStore  <net>/eval/suite/quant/<e-id>
 Runner (the project's class, BEV: runner/runner.py)  -- the only class that executes the network   (additive counts per frame: small)
   predict  plan  list_results  tap  time  probe     no ground truth, no loss, always warm (D70)
        |  writes / reads
        v
 ResultSet (results.py)   <net>/eval/results/<r-id>/   frames/<scene>_<i>.npz   taps/<scene>_<i>/<tap>.npy   manifest.json
   one per identity (network, checkpoint, inputs); any question reuses its frames; derived data

 project adapter  <project>/evaluation/evalsuite.json -> bev_eval.py     (both classes call it; the core imports no torch / nuScenes)
   required: name out network sets identity source subject task
   optional: free_source + free_subject (together) reference card time probe resolve release inputs evaluator_extra evaluator_now MEMORY_TAP COMPARE_IGNORE
   BEV: bev_adapter, bev_free (the network without labels), bev_suite, bev_sets (the sets = the training's splits/)
```
A caller (a person, a script, the command line) talks to the Evaluation only. The Runner is the layer below: the only class that runs a network (D72: it reads no ground truth and computes no loss) and stores what
it predicts. There is one Runner and the Evaluation always includes it: the class a project names in `<project>/evaluation/evalsuite.json` as `"runner": "module:Class"` (module relative to the
project root, like the adapter; built with `(project, adapter)`; BEV: `runner.runner:Runner`, whose command line is `python -m runner` from `bev_fusion/`, for frames without labels). evalsuite has no
Runner of its own and never imports a project's by name; a project that names none is an error, and `Evaluation(runner=...)` injects any object with `predict`, `plan`, `list_results`, `tap`, `time`,
`probe`. The toy selftest of this interface (`AR_PROJECT=../bev_fusion python -m evalsuite.selftest_api`) therefore runs the Evaluation together with the BEV Runner on a toy adapter.

Data flow of one `Evaluation.run`: frames (a text) -> the Runner runs the scenes up to their last kept frame, whole and warm, stores the kept frames as a ResultSet (the temporary data: only what the
network predicted) and releases the network -> the Evaluation opens the labelled data, scores the frames against the ground truth (`reference`, no inputs) into a RunStore (the counts) ->
`show_numbers` / `compare` / `find_worst` read the counts. The ResultSet is only needed to score again (another ground truth, metric or subset) and to draw; it is derived data (deletable through
`autoresearch/tools/package.py`, D40). A wrong `gt_key` or an unreadable ground truth shows only after the network ran: the predictions are kept, and the message names them (`score` them again).
A network with a memory always runs warm (D70): whole scenes in time order from the network's own start; the frames you ask for decide which results are kept and scored.

## Class interface
```python
from evalsuite import Evaluation
ev = Evaluation()                                                   # the project is AR_PROJECT; Evaluation(runner=...) injects another Runner
ev.check("runs/t23_C_s1", "small.val")                              # the dry run: what would run, what is stored, the problems
r = ev.run("runs/t23_C_s1", "small.val", gt_key="v8")               # the Runner predicts what is not stored yet (warm, D70), then the Evaluation scores it: a dict with the id
ev.show_numbers(r["id"]);  ev.split_numbers(r["id"], "scene");  ev.find_worst(r["id"], "seg", n=5);  ev.get_cost(r["id"]);  ev.report(r["id"]);  ev.make_card(r["id"])
ev.compare(control_id, r["id"])                                     # b - a, paired bootstrap; one network per process: the control was run in another one, compare by ids
ev.tap(r["results"], "scene-0061/12", "fused")                      # an intermediate result stored with the predictions (asked for with run(..., taps=("fused",)))
```
| method (Evaluation) | what it does |
|---|---|
| `check(network, frames, metrics, taps, workers, control)` | the dry run: the Runner's plan (scenes, frames, what is stored, how many frames would run) plus the ground truth and the metrics, the id it would give, `problems` (exit 1 when there are) |
| `run(network, frames, gt_key, metrics, taps, device, workers, control, progress)` | `Runner.predict` (the network, then released), then `score` on the labelled data: a plain dict (the id, what was done, the warnings of the memory's health) |
| `score(results, frames, gt_key, metrics)` | score stored predictions against the ground truth; no network |
| `show_numbers(id, match)`, `split_numbers(id, axis)`, `find_worst(id, match, n)` | the numbers (with `warnings`: devices, the memory's health, evaluator code changed since), the numbers per scene or frame, where they come from |
| `compare(a, b)` | paired bootstrap, b - a (a = the control); refused on another source, task or GT; `n_scenes` and `warnings`: frames in only one of the two, fewer than 30 scenes (the interval is too narrow then), other evaluator code |
| `get_cost(id)`, `report(id)`, `make_card(id, out)` | ms per frame, the html page, the metric card in the project's contract WRITTEN as a file (default `<net>/eval/card_<e-id>.json`, or `out`; -> `{path, card}`; a verdict reads it: name it in the experiment card's `arm.cards`, check it with `python autoresearch/tools/card.py check FILE`); refused when the checkpoint or the evaluator code changed since, or the predictions are gone: score again |
| `draw_frame(id, frame, view)` | one picture (the network runs on that frame with its scene's history; refused like `make_card`) |
| `list_sets()`, `list(network)` | the sets a project defines, what is stored for a network (predictions and evaluations) |
| `tap(results, frame, name)`, `time(network)`, `probe(network)` | passed to the Runner: which taps a stored frame has or the file, shape and dtype of one; ms per module on an idle GPU; the module probe |

| method (Runner, the layer below; Python) | what it does |
|---|---|
| `predict(network, frames, taps, device, workers, control, progress)` | run the network on what is not stored yet and store it (no ground truth), with the intermediate results `taps` when asked (a frame stored without a tap is predicted again to get it); the warnings of the memory's health (D70: empty at the scene start, filling after) |
| `plan(network, frames, taps, workers, control)`, `list_results(network)` | the dry run of `predict` (scenes, frames, stored, to run, `problems`); the stored predictions of a network |
| `tap(results, frame, name)`, `time(network)`, `probe(network)` | as above |

What an id stands for. `<network>@r<hash>` the predictions: the network (checkpoint file hash, config, code snapshot or live code, the adapter files), what it was fed (the set's data, the control) and the
memory mode; the same question gives the same id, a changed checkpoint another. `<network>@e<hash>` a scored evaluation: the predictions, the frames asked for (with their dataset keys), the set they were
asked through, the ground truth, the metrics, the evaluator code. The numbers of an Evaluation have no `loss.*` (a prediction carries no validation loss, D72; `bev_suite.py eval` keeps them). One network per
process (a project's code modules are imported once; the BEV adapter refuses a second code snapshot in a task, a timing, a probe or a source): run each network in its own process and compare the evaluations by
their ids. Frames are a text: `val`, `small.val`, `spread:1000:large.val`, `random:3[:SEED]:val`, `scenes:a,b:val`, `frames:scene-0061/12:val` (the set is last, and one the project defines). A repeated question
costs no network but still builds the data (about a minute for BEV). The head of `api.py` has the adapter's functions; test: `python -m evalsuite.selftest_api` (a toy project, seconds) and, for BEV,
`tests/test_evaluation_api.py [--real RUN]`.

## Command line
Every Evaluation method is a command (`python -m evalsuite <method>` or `Evaluation.<method>`; `_` is `-`); `list`, `probe` and `selftest` need the class name because older commands of `cli.py` use these words.
| what | command (from `autoresearch/`, `AR_PROJECT=../bev_fusion`) |
|---|---|
| the dry run | `python -m evalsuite check runs/t23_C_s1 small.val` |
| run and score | `python -m evalsuite run runs/t23_C_s1 small.val --gt-key v8 [--metrics maps,boxes,fusion] [--taps fused] [--device cpu]` |
| score stored predictions | `python -m evalsuite score t23_C_s1@r3f9a2c1d4b0 small.val --gt-key v8` |
| read | `show-numbers ID [--match M]`, `split-numbers ID [--axis scene]`, `find-worst ID`, `compare A B`, `get-cost ID`, `report ID`, `make-card ID [--out FILE]`, `draw-frame ID FRAME` |
| what exists | `list-sets`, `Evaluation.list NETWORK`, `tap RESULTS FRAME [--name N]`, `methods` |
| the network | `time NETWORK`, `Evaluation.probe NETWORK` (the Runner's, passed through) |
| every command | `--json` (nothing else on stdout), `--project`, `--adapter`; exit 0 ok, 1 `check` found problems, 2 any error (`EVALSUITE_TRACEBACK=1` shows the traceback) |

`run` here is not `autoresearch/tools/run.py`: it books no ledger row and takes no GPU slot; a number that feeds a verdict is launched with `run.py ... -- python -m evalsuite run ...`.

## Underneath: the engine `EvalSuite`
The Evaluation counts and draws with this engine and builds its workbench (`EvalSuite`: a Source, a Subject or none, a Task, with `.quant` and `.qual`) for each call; a person does not construct one. `EvalSuite`
is the older entry point of `bev_suite.py`, the lane project and the engine's selftests, and it works alone for a network that has no project adapter. You write three small classes for your network; the suite then gives you pictures, numbers and cost.

### Try it
The whole adapter is `adapters/minimal.py` (about 70 lines): a moving disc in noise, a thresholding "network", one metric.
```
python -m evalsuite --adapter evalsuite.adapters.minimal:build --out ../eval_out qual random 3
python -m evalsuite --adapter evalsuite.adapters.minimal:build --out ../eval_out quant run --name a
python -m evalsuite --adapter evalsuite.adapters.minimal:build --out ../eval_out quant table a
python -m evalsuite selftest minimal          # checks the same against known answers
```

### The three boxes
| box | you say | what you write |
|---|---|---|
| `Source` | which scenes ("videos") and frames exist, how to load one | `scenes()`, `frames(scene)`, `load(ref)` -> `Sample(inputs, reference)` |
| `Subject` | how to run the network on one frame | `predict(sample)` -> `Prediction(outputs)` |
| `Task` | what to measure, how to draw | `metrics()`, `views()` with one `overview` |

One rule makes the rest work: a `Metric.frame()` returns **additive counts** (true / false positives, sums, ...) and `Metric.reduce()` turns the counts of any set of frames into numbers. A dataset number is therefore a sum over frames, never a mean of per-frame numbers, and it can be recomputed for one scene, without one scene or for a bootstrap resample without the network. A `View` returns a `Page` of `Panel`s (data, no plotting code). `build()` returns `(Source, Subject, Task)`.

### What you get
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

### Where the comparison happens
Every comparison of a prediction with the ground truth lives in `comparison/` (read `comparison/README.md`): `maps.py` compares two pixel maps, `boxes.py` two lists of boxes, both on plain numpy arrays and dicts, with counts that add up over frames. `comparison/metrics.py` turns them into ready-made `Metric`s (`MapMetric`, `BoxMetric`): you only write the function that hands over the two things to compare.

### Port it to your network
1. Copy `adapters/minimal.py` anywhere, e.g. `<problem>/evaluation/my_adapter.py`; it imports only `evalsuite`.
2. `Source`: list the scenes and their frames in time order; `load()` gives `inputs` (for the network) and `reference` (for metrics and pictures).
3. `Subject`: load the network; `predict()` returns its outputs.
4. `Task`: one `Metric` per family of numbers, one `overview` `View` (ground truth | prediction | errors).
5. `python -m evalsuite --adapter path/to/my_adapter.py:build selftest-adapter` checks the contracts and says what is wrong. Then `qual random 5` and `quant run`.

## More
`EXTENSIONS.md` covers what a bigger project needs: a network with memory, looking inside the network (taps, torch), outlier details, stored runs, cost and the Pareto front, probing the causes of errors, where the lane-centre-line task went (the lane project), the real adapters and a map of the files. The design notes of the first build are in git history (`autoresearch/history/evalsuite_architecture_2026-10-30.md`, removed 2026-11-10).
