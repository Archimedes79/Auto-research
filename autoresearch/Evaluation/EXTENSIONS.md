# Evaluation - extensions

`README.md` is enough to use it. This page is for writing or growing a project's adapter.

## 1. The adapter contract
A module of the project, named in `<project>/evaluation/evaluation.json` as `{"adapter": "evaluation.adapter"}`. Worked examples:
`minimal.py`, `problems/toy/evaluation/adapter.py` (torch, taps), `problems/bev_fusion/evaluation/bev_eval.py` (a large chain),
`problems/lane_center/evaluation/adapter.py` (the network ran in a step of its own; the Subject reads its dump).

| required | |
|---|---|
| `sets()` | `{set name: number of scenes, or None}` |
| `source(net, of, workers, control)` | the `Source` of set `of`, with ground truth |
| `subject(net, source, device)` | the `Subject` |
| `task(net, metrics)` | the `Task`; `ValueError` for an unknown metric |

| optional | without it |
|---|---|
| `METRICS` | every call names the metrics |
| `name`, `out`, `network`, `resolve`, `checkpoint`, `identity` | run folders (`defaults.py`): `config.json`, the last `ckpt_*.pt`, results below `<run>/eval`, names looked up in `RESEARCH_RUNS` or `<project>/runs` |
| `free_source`, `free_subject` (both or neither) | the Runner reads `source` / `subject`; with them it sees no ground truth at all |
| `reference(source, ref)` | the ground truth of a frame is `source.load(ref).reference` |
| `release(source)` | what a source holds of the network is not given back early |
| `evaluator_now()` | `show-numbers` cannot warn that the evaluator code changed since the scoring |
| `COMPARE_IGNORE` | `compare` refuses any difference of the sources; `{"source": (keys,)}` names what two arms may differ in |
| `MEMORY_TAP` | no health check; with it the tap's mean must be empty at a scene start and fill (D70), else a warning |
| `time(net, n)`, `probe(net, frames, device)` | no `time` / `probe` |

`Source.describe()` must give `gt_signature` and `evaluator_hash` (the hash of every file that shapes a number); without either nothing is
scored. Its other keys say what the network is fed and are part of the identity of the predictions (not: class, set, ground truth, counts).

## 2. The boxes (`contracts.py`)
| box | methods |
|---|---|
| `Source` | `scenes()`; `frames(scene)`: `FrameRef(scene, index, key)` in time order; `load(ref)`: `Sample(ref, inputs, reference, display, meta)`; `prefetch(refs)` is a hint. `display` values may be callables (`sample.get`) |
| `Subject` | `predict(sample, taps)`: `Prediction(outputs, taps, info)`; `reset()` forgets the memory; `taps()`, `describe()`, `close()`. Outputs are stored without pickle: numpy arrays, numbers, texts, lists, tuples, dicts |
| `Task` | `metrics()`, `views()` with an `overview`; `checks(sample, pred)`: `Flag`s shown in the debug pictures |

## 3. A new metric
`frame(sample, pred)` returns counts: numpy arrays that add up over frames (or concatenate along axis 0, any length). `reduce(counts, keys)`
turns the counts of any set of frames into `{name: float}`, nan where there is nothing to judge, independent of the order; with `keys` it may
skip the rest (the bootstrap calls it thousands of times). `headline` are the numbers of comparisons, scenes and outliers, `better` marks the
`"low"` ones. `score(counts)` ranks frames (default: the headline of the frame alone); **override `support` for a ratio** (frames below it are
not ranked as outliers, the default is 1: a precision from three cells would count like any other). Ready-made: `MapMetric` and `BoxMetric`
(`comparison/metrics.py`). Since every number is a reduction of summed counts, per scene, leave-one-out and bootstrap need no network, and
`run` after a change of the metric scores the stored predictions again. Add the new file to the `evaluator_hash`.

## 4. A new picture
`View.render(sample, pred, ctx)` returns a `Page(title, panels, ncols, legend, footer)` of data, no plotting code. `Panel` kinds: `image`,
`raster`, `rgb`, `text`, `table`, `bars`; `Shape` overlays: polygon, line, points, circle, arrow, text. `ctx.scores` are the frame's scores.
One colour code for errors (`render.figure.error_rgb`): grey right, grey-green false positive, grey-blue false negative, cream not
evaluated; never red (red marks moving targets). `draw-frame ID FRAME --view NAME` draws it from the stored prediction, so does the `draw`
list of a research report; the frame's label and scores head the title. A view with `needs_taps = True` also gets the taps stored with the
frame, their `TapSpec`s and the subject's description (`ctx.taps`, `ctx.subject`: the subject is built, not run) and the plausibility flags
(`ctx.flags`: the Task's `checks` and the taps'); the built-in view `debug` shows each tap by its kind, their statistics and the flags.

## 5. Looking inside: taps
`Subject.taps()` lists `TapSpec(name, kind, stage, note, redflag, extent, flip_x, optional, alive_check)`. `run ... --taps a,b` stores those
arrays with the predictions (a frame stored without one is predicted again); `tap RESULTS FRAME --name a` gives file, shape and dtype. For
torch: `torch_taps.TapRecorder(model, {name: Hook(path, when, pick, select)})`, `arm(names)` before the forward pass, `read()` after it; a tap
that cannot be read warns and never breaks the pass. `torch_taps.count_ops(fn)` counts GMACs per module (convolutions, matrix products and
attention only: name the rest in your note).

## 6. The Runner and what is stored
`ev.runner` (`netrunner.py`): `predict`, `plan`, `list_results`, `tap`, `time`, `probe`. `predict` runs each scene in time order up to its
last frame asked for, keeps the frames asked for and resumes after a stop; the same identity finds the same frames again. Below `out(net)`:
```
results/r<hash>/       predictions: frames/<scene>_<index>.npz, taps/<scene>_<index>/<tap>.npy, manifest.json (identity, cost, health)
suite/quant/e<hash>/   an evaluation: manifest.json, frames.jsonl (scores, support), counts_<metric>_<chunk>.npz; report.html, frames.csv
suite/pictures/        drawn frames
```
Both are derived data; an evaluation stays readable without its predictions, drawing and scoring again need them. Code with statistics of
its own (the judge) reads `ev.stored_counts(id)` (`by_scene(metric)`) and `ev.metrics_of(id)`.

## 7. The engine without an adapter
`EvalSuite(source, subject, task, out=...)` gives `.quant.run(name=...)`, `table`, `per_scene`, `outliers`, `compare`, `report`: the same
counting on a Source and a Subject you built yourself (the tests do, `tests/test_engine.py`).
