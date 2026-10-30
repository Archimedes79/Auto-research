# evalsuite - architecture (plan written before the code, 2026-10-30)

Goal (Mirko): look into any network problem with two suites on one core.

- **qualitative**: pictures of results over ground truth with the errors highlighted, for a concrete frame (scene + frame, or token), for n random frames (first a random video, then a random frame in it), with a debug view into the network parts, and result videos of m random sequences;
- **quantitative**: numbers over all videos together, and every number traceable to the videos and frames it comes from (outliers), each openable in the qualitative suite;
- a documented box with very few interface boxes, the abstract machinery decoupled from the concrete network, so a new network is a small adapter.

## 1. The three boxes you implement for a network (everything else is shared)

```
                        +---------------------- evalsuite (core: numpy + matplotlib, no torch) -----------------------+
 your dataset           |                                                                                            |
 +--------+  scenes()   |  sampling   -> pick scenes / frames (video first, then frame; seeded; find by name / token)|
 | Source |  frames()   |  runner     -> Source.load -> Subject.predict -> Metric.frame / Task.checks  (state carry) |
 |        |  load()  ---+->  qualitative: Page -> figure engine -> png / html / mp4                                  |
 +--------+             |  quantitative: RunStore (counts per frame) -> reduce / per scene / outliers / compare       |
 your network           |                                                                                            |
 +---------+ reset()    |  render: figure (panels, error colours), taps (generic tap pictures + plausibility), report,|
 | Subject | predict()--+->        video (H.264, read back)                                                          |
 |         | taps()     |  stats: robust z, leave-one-out influence, paired scene bootstrap                          |
 +---------+            |                                                                                            |
 your meaning           |                                                                                            |
 +------+ metrics()     |                                                                                            |
 | Task | views()    ---+->                                                                                         |
 |      | checks()      |                                                                                            |
 +------+               +--------------------------------------------------------------------------------------------+
```

| box | you say | what the core never knows |
|---|---|---|
| `Source` | which scenes ("videos") exist, which frames each has (order = time), how to load one frame (inputs, reference, things to display) | nuScenes, files, caches, coordinate frames |
| `Subject` | how to run the network on one frame (stateful along a scene), which intermediate tensors can be tapped, how many history frames a random frame needs (`warmup`) | torch, models, checkpoints, devices |
| `Task` | which `Metric`s to compute (per-frame additive counts + the reduction to numbers), which `View`s draw a frame (a `Page` of panels, pure data), plausibility checks | what a lane, a box or a pixel class is |

`Metric` is the one place where the maths lives, in the form that makes tracing possible: `frame(sample, pred)` returns **additive counts** (numpy arrays, fixed shape or ragged), `reduce(list of counts)` turns any set of frames (all, one scene, all but one scene, a bootstrap resample) into numbers, `score(counts)` gives per-frame scalars to rank outliers. Dataset-level numbers are therefore sums over frames, never means of per-frame numbers, and a number can be recomputed for any subset without the network.

A `View` returns a `Page` (title, `Panel`s: image / raster / rgb / text / table, with `Shape` overlays, legend). The figure engine draws every page in one style; errors have one colour code everywhere (true positive grey, false positive red, false negative blue, not evaluated cream). A `View` holds no matplotlib code, so views are testable as data.

## 2. Data flow

```
qualitative.frame("scene-0061/12")      Runner.run_to: reset, feed Subject.warmup history frames, then the frame (taps on)
                                         -> View.render -> Page -> figure -> <out>/qualitative/frames/<slug>_<view>.png (+ taps .npz)
qualitative.random(n, seed)             Sampler: scene uniform, then frame uniform in the scene (n draws, no duplicates)
qualitative.videos(m, seed)             Sampler: m scenes; Runner.iter_scene (state carried) -> one Page per frame -> mp4 (+ per-frame score strip)
quantitative.run(...)                   Runner.iter_scene over the chosen scenes -> RunStore: per frame counts of every Metric, scores, flags, ms
quantitative.table / per_scene          Metric.reduce over all frames / the frames of each scene
quantitative.outliers                   per-frame robust z of Metric.score + leave-one-out influence of every scene (and frame) on the headline numbers
                                         + plausibility flags; each row carries the frame label and the command that draws it
quantitative.compare(a, b)              paired scene bootstrap of the headline numbers between two stored runs (no network needed)
quantitative.report                     html: tables, outlier list with thumbnails (rendered through the qualitative suite)
```

State: `state="carry"` runs a scene in order (the memory of a stateful network carried from frame to frame, what a deployed system does), `state="reset"` makes every frame independent (cold). Both are the same code with one flag; the paired difference of the two stored runs is the test whether a memory helps.

## 3. Directory and software layout

```
autoresearch/evalsuite/                 the core (this folder); python -m evalsuite ...
  README.md  ARCHITECTURE.md
  __init__.py       public API: EvalSuite, Source, Subject, Task, Metric, View, FrameRef, Sample, Prediction, Page, Panel, Shape, TapSpec
  contracts.py      the boxes and the data (about 250 lines, no logic besides identity parsing)
  suite.py          EvalSuite(source, subject, task, out): the one object the toolchain uses (.qual, .quant)
  runner.py         Runner: scene iteration with state carry / reset, random access with warmup, prefetch
  sampling.py       Sampler: video-then-frame draws, scene draws, evenly spread subsets, seeds
  store.py          RunStore: manifest + per-frame records + chunked counts (resumable, ragged arrays)
  stats.py          robust z, IQR fences, leave-one-out influence, paired scene bootstrap
  qualitative.py    frame / random / debug / videos
  quantitative.py   run / table / per_scene / outliers / compare / report / timing
  cli.py            python -m evalsuite --adapter pkg.module:build [--arg k=v] <qual|quant|selftest> ...
  torch_taps.py     optional: forward-hook TapRecorder for torch models (module path -> tap), imports torch lazily
  render/
    figure.py       Page -> png / ndarray (panel kinds, overlays, legend, error colours, helpers error_rgb / mask_rgb)
    taps.py         generic tap pictures by kind (energy, PCA, heat map), statistics and plausibility checks
    report.py       small html writer (index pages, tables, thumbnails, relative links)
    video.py        frames -> H.264 mp4, read back and checked (the former bev_fusion/evaluation/video_encode.py)
  adapters/
    toy.py          synthetic videos + a numpy "network" + metrics + views: the test bed and the porting template
  selftest.py       known-answer tests of every function on the toy adapter

<problem>/evaluation/                   one adapter per problem (bev_fusion: bev_adapter.py; entry bev_suite.py)
  <problem>_adapter.py   build(**kv) -> (Source, Subject, Task)
```

Rules: the core imports only numpy (+ matplotlib / PIL / cv2 lazily in `render/`); an adapter may import anything. One `Subject` per process when the network code is pinned per run (module cache). Outputs go below `--out` (default `<cwd>/eval_out`): `qualitative/`, `quant/<run id>/`; nothing is written into the core folder.

## 4. How to port the suite to another network (checklist)

1. Copy `adapters/toy.py` to `<problem>/evaluation/<name>_adapter.py`.
2. `Source`: list your scenes/frames, load one frame (`inputs` for the network, `reference` for the metrics, `display` for the pictures).
3. `Subject`: load the network, `reset()` its state, `predict(sample, taps)`; name the stages you want to look into as `TapSpec`s (with `note` = what it should look like, `redflag` = what a problem looks like); for torch use `TapRecorder` with module paths.
4. `Task`: one `Metric` per family of numbers (additive counts!), one `overview` `View` (GT | prediction | errors), optional more views and checks.
5. `build(**kv)` returns the three; run `python -m evalsuite --adapter yourpkg.module:build selftest-adapter` (the generic contract checker: loads a frame, predicts, metrics add up, pages render) and then `qual random 5`, `quant run`.

## 5. Proof plan (what makes "works flawlessly" checkable)

| layer | check |
|---|---|
| core on the toy adapter | every function against a known answer: sampling determinism and its two-stage law, frame lookup by label / tuple / token, store round trip incl. ragged arrays and resume, `reduce` of all = sum of parts, leave-one-out by hand, robust z vs numpy, planted outlier frames are found and rank first, bootstrap on a planted better / worse subject, error colours at exact pixels, pages render with the right size, video read back with the right frame count, html links resolve, carry vs reset differ for a stateful subject |
| BEV adapter, no network | the Metric wrappers give the same numbers as `SegMeter` / `DetMeter` fed directly (also on random data: flags-based AP = `DetMeter.result()`); views on synthetic GT |
| BEV adapter, real run | numbers of the suite on a fixed frame set = numbers of the old `evaluate_run.py` on the same frames (golden file written before the old code is removed); the worst frame of the outlier list is drawn; one random draw, one named frame, one debug view, one video of a real run |
| contract checker | `selftest-adapter` passes on the toy and the BEV adapter |

## 6. Migration of the old scripts (Mirko: replace and fix all callers)

`evaluate_run.py`, `observe_run.py`, `eval_memory.py`, `video_encode.py` of `bev_fusion/evaluation/` are replaced by the suite; their callers (about 25 scripts and shell chains of other sessions) are rewritten to the new API / command line. Order: golden numbers from the old code first, then the suite, then callers, then the old files go to `bev_fusion/_to_delete/` (listed there); a shell script that is executing right now is not edited (bash reads scripts by offset): it is rewritten when its lane has ended, and until then the old entry point stays as a short forwarding stub.
