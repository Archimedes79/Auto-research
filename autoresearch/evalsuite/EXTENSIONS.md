# evalsuite - extensions for bigger projects

`README.md` is complete for a simple project. Read only the chapter you need; each one stands alone.

| chapter | you need it when |
|---|---|
| 1 More on the three boxes, command line options | the network or the data has more than the minimum |
| 2 A network with memory | the network carries a state from frame to frame |
| 3 Looking inside the network | you want to see intermediate results |
| 4 Where the numbers come from | a number looks wrong, or you have to explain it |
| 5 Stored runs, cost and the Pareto front | long runs, resuming, ms and operations per frame, quality against cost |
| 6 Probing: where do the errors come from | you want an error budget by cause |
| 7 Lane centre lines | the network outputs lane centre lines |
| 8 Checking an adapter, the selftest | a new adapter, a change of the suite |
| 9 Real adapters, files, rules | porting a large project, changing the suite |

## 1. More on the three boxes, command line options
- **Source**: `FrameRef(scene, index, key)`: `key` is the dataset's own id (a token), so `qual frame <key>` works. `Sample.display` holds extra things for pictures (values may be callables, read them with `Sample.get`), `Sample.meta` free text. `describe()` goes into the run record, `prefetch(refs)` is a hint, the runner reads ahead in a thread.
- **Subject**: `reset()`, `warmup`, `taps()` (chapters 2 and 3), `describe()` (**put every parameter that changes numbers in it**: a resume of a stored run with another `describe()` is refused), `ops()` and `benchmark()` (chapter 5), `close()`.
- **Task**: several `Metric`s (unique names); more `View`s next to `overview` (`default_video_view` picks the one for videos, `--view a b` draws others); `checks()` returns `Flag`s (ok / warn / fail) per frame; `guide()` is the text on top of the `qual random` page; `probe()` / `causes()` are chapter 6.
- **Metric**: `score()` gives per-frame scalars (default: the headline numbers of the frame alone), `support()` how much evidence a score rests on (e.g. the ground-truth cell count; frames below `--min-support` are not ranked as outliers; the default is 1 for every frame, so **override it for ratio metrics**, a precision from a handful of cells would otherwise rank like any other), `better = {key: "high" | "low"}`. Counts may have any shape or be ragged along axis 0; the store never reads them. `reduce(counts, keys)` may skip expensive numbers when `keys` is given (the bootstrap calls it thousands of times) and must not depend on the frame order.
- **View**: panel kinds `image`, `raster`, `rgb`, `text`, `table`, `bars`; `Shape` overlays (polygon, line, points, circle, arrow, text). One colour code everywhere (`render.figure.error_rgb`): grey = true positive, red = false positive, blue = false negative, cream = not evaluated.
- **Options**: `list [SCENE]`; `qual frame|random ... --view a b --taps a b`, `qual videos M --fps F --max-frames N --scenes s1 s2`; `quant run --n-scenes N | --scenes s1 s2 | --spread N | --max-frames N | --seed S | --name NAME`; `quant compare A B --boot N`; `python -m evalsuite --help` lists all.

## 2. A network with memory
`--state` (or `EvalSuite(state=)`) says how the memory is treated:
- `carry` (default): a scene runs in time order, `reset()` once at its start. This is what a deployed system sees.
- `reset`: every frame independent. `quant compare` of a carry run and a reset run tests whether the memory helps.
- `warmup`: every frame starts empty and sees only the `Subject.warmup` frames before it; any subset of frames is allowed (from Python: `quant.run(frames=..., state="warmup")`).

`quant run --spread N` takes N frames evenly over all scenes and always runs them in state `reset`. `qual frame` feeds `Subject.warmup` history frames first (`--history warmup | full | none`; `full` = every frame from the start of the scene up to the frame). `quant trace` replays the stored run's state; the replay is exact for a carry run that started at the first frame of the scene. `carry` refuses a frame subset with gaps.

## 3. Looking inside the network
`Subject.taps()` lists `TapSpec(name, kind, stage, note, redflag, ...)`; `predict(sample, taps=[names])` returns those as numpy arrays (one sample, no batch axis). `kind` is `feature_map`, `map`, `image`, `logits`, `vector` or `scalar`; `note` says what the tap should look like (printed under the picture), `redflag` what a problem looks like, `optional` marks parts a configuration may lack, `alive_check=False` allows an all-zero tap.

`qual debug SPEC [--taps a b]` writes `_debug.png` (one picture per tap by kind: feature maps as energy and PCA colours, maps, images, logits as maximum probability, bars), a statistics table, plausibility flags (non-finite, dead or constant, no spatial structure, extreme magnitude) and `_taps.npz` with the raw arrays. The flags are heuristics: they point, a human judges.

For torch networks `torch_taps.TapRecorder(model, {name: Hook(path, when, pick, select)})` records by module path with forward hooks. `arm(names)` switches recording on for the next pass, `read()` returns the arrays. It never breaks a forward pass (a tap that cannot be read is skipped and counted). The hooks use only the tensors' own methods; torch is imported lazily in `count_ops` (chapter 5).

## 4. Where the numbers come from
`quant outliers RUN [--k --z --min-support]` returns three lists, every row with the command that draws it:
- **scenes**: per headline number the scenes whose removal improves it most (exact leave-one-out; `drag > 0` = the scene hurts); the row has `scene` and the command `qual videos 1 --scenes S`.
- **frames**: per score the frames furthest outside the others **in the bad direction** (robust z from median and MAD, among frames with enough support; only the bad tail is marked, `z <= -3.5` for a higher-is-better score, `z >= 3.5` for a lower-is-better one), with the frame's own drag on the headline numbers and its label.
- **flags**: frames with plausibility flags, worst first.

`quant trace RUN LABEL` draws such a frame (overview + debug), `quant report RUN` writes `report.html` (tables, outliers, pictures of the worst frames; `--render 0` for none) and `frames.csv`, `quant per-scene RUN` the headline numbers per scene, `quant table RUN --all` every number. `quant compare A B` is a paired scene bootstrap of `B - A` on the frames both runs have.

**Which commands need the network**: `quant table`, `per-scene`, `outliers`, `compare`, `pareto`, `probe budget` and `probe situations` work on the stored counts alone. `quant trace`, `quant report` (unless `--render 0`), `probe sample` and `probe scan` (unless `--no-render`) run the Subject again to draw frames.

## 5. Stored runs, cost and the Pareto front
**What is written where** (below `--out`, default `eval_out`; nothing goes into the suite's folder):
- `qualitative/frames/<scene>_<idx>_overview.png` (+ `_debug.png`, `_taps.npz`), `qualitative/random_s<S>_n<N>/index.html` + pictures + `draws.json`, `qualitative/videos/<scene>_<view>.mp4` + `_scores.png` / `_scores.json`.
- `quant/<run>/`: `manifest.json` (who ran what, commit, time, cost), `frames.jsonl` (one line per frame: scores, support, flags, info), `counts_<metric>_<chunk>.npz` (the counts, 64 frames per chunk), later `report.html`, `frames.csv`, `trace/`. `quant/pareto_<y>.png` and `.json`, `quant/timing.json`.
- `probe/<run>_s<seed>/index.html` (chapter 6).

**Resume**: an interrupted run leaves a consistent prefix; the same command with the same `--name` resumes it. A resume must be the same run (state, source, subject, task, selection; the device aside), otherwise it is refused (`RunStore.create(..., force=True)` overrides). `quant run(on_frame=...)` calls a function for every frame run. Frames are written to disk before the run ends, also when it is interrupted.

**Runtime.** `quant run` stores `cost.run_ms` in the manifest: median, p90 and mean of the time `Subject.predict()` took as the suite called it, over every stored frame (also those of earlier sessions of a resumed run), with the machine load when the run started (`load`) and ended (`load_end`). It times whatever `predict()` does besides the network (decoding, copies back to the CPU) with the read-ahead thread running beside it, on a machine that may be busy: a by-product, not a benchmark. For a number to compare, `quant timing --run RUN` stores `cost.bench`: warm-up, then the median of n calls on one frame (`Subject.benchmark()`, override it to time only the network and to synchronise a device), with the load average read **before** it started, the CPU count, the host and the GPU name. It is attached only if the Subject, Source and Task of the suite describe the run (`--force` overrides). Valid only on an idle machine; `pareto` warns when the load was above half the CPUs when the timing began (a load average cannot see contention that starts later) and when the runs were timed on different hosts or GPUs.

**Operations.** `Subject.ops(sample)` returns `{"gmacs", "method": "measured" | "estimated" | "mixed", "note", "parts"}` (a MAC is 2 FLOPs); `quant run` calls it once before the first frame, resets the memory afterwards and stores it as `cost.ops` (numpy numbers are turned into plain numbers); an `ops()` that fails or returns anything but a dict with a number under `gmacs` only prints a message and is tried again on a resume. Who should count?
- *The tool, for everything it can see.* `torch_taps.count_ops(lambda: model(x))` counts convolutions, matrix products (linear layers) and attention exactly for the input it was given (call it with the model in `eval()`), with GMACs per module down to `depth` levels below the root (`parts`). An agent or a person reading the code cannot beat that.
- *A reader of the code, for the rest.* The counter is blind to element-wise ops, norms, interpolation, `grid_sample`, scatter / splat, custom CUDA ops (deformable attention, voxelisation) and numpy / numba post-processing. Add such a part with its formula: `cost.with_estimate(ops, "splat", gmacs, "H*W*K = 200*200*12 MAC")`. The result is marked `mixed` (or `estimated`), the formula is in `note`, so a reviewer can check it.
- *Operations are not runtime.* Memory-bound and scatter-heavy parts cost time with few MACs; always look at both.

**Pareto front.** `quant pareto A B C [--y metric.key] [--cost ms gmacs]` takes the quality `y` (default: the first headline number) of every run on the frames **all** runs have, puts the runs on the plane quality x cost and marks the runs no other run beats on both (cost not higher and quality not worse, one of them strictly). It prints a table, writes `quant/pareto_<y>.png` (one panel per cost axis, the front dashed) and `.json`, and says in `notes` where the ms come from (benchmark or the run itself; mixed sources are flagged as not comparable), how many frames are common, which runs lack a cost or have no quality. A wrong quality axis (`--y`) is an error. The front is a point estimate in both directions, there is no noise band: a run 0.001 worse and 0.01 ms slower counts as beaten. For two neighbours, `quant compare` tells whether the quality difference is real. Compare only runs of the same task, timed on the same machine.

## 6. Probing: where do the errors come from
A `Task` that implements `probe(sample, pred)` returns per frame numbers `tag.<name>` (a property of the situation: a lane crossing, curvature, range, night ...) and `cause.<name>` (error assigned to a cause, in one unit that adds up, e.g. metres of lane); `causes()` says for each cause what it is and what to improve (model, code or evaluation). `probe budget RUN` splits the error of the whole run by cause, `probe situations RUN` lists all groups of frames with their excess (share of error minus share of frames; positive = costs more than its size), `probe sample RUN "dominated by fn.gap" -n 6` draws random examples of one (a video first, then a frame), `probe scan RUN` does all of it as one page (the situations with a positive excess, k examples each). From Python: `suite.probe.scan(run)`.

## 7. Lane centre lines (`lanes/`)
`LaneTask` is a ready-made `Task` for any network that outputs lane centre lines, as a picture (a probability map) or as a graph (polylines with successor edges). The adapter delivers `Sample.reference = {"lanes": LaneGraph | polylines, "mask": bool image (optional), "valid": evaluated cells (optional), "visible": what the input can observe (optional)}` and `Prediction.outputs = {"prob": image and / or "lanes": graph}`. One output is scored at both levels with the same run, `quant table RUN` shows them next to each other:

| metric | what it is |
|---|---|
| `img` | strict IoU / Dice; tolerant precision, recall, F1, quality at 0.5 / 1.0 / 1.5 m; the same on the thinned prediction; soft Dice; tolerant average precision |
| `geo` | on polylines (a dense head's picture is thinned and traced first): P / R / F1 at 0.5 / 1.0 / 1.5 m, chamfer, lanes found, one-to-one F1, duplicate share, direction accuracy, vertices per 100 m, length ratio |
| `topo` | TOPO marbles (what can be reached along the graph within 30 m), junction F1 (3 m) |

The probing causes: missed lane near / far, hole (gap), truncated, displaced (a parallel neighbour within 3 tau; a crossing lane does not count), the whole prediction shifted (best shift within 2 m), duplicate, stray piece (spur), overshoot, invented lane (hallucinated, or the ground truth is missing there), outside the input's view (unobservable). Every missing or false metre gets exactly one cause by a fixed priority, so the causes add up to the frame's error (tested against a brute-force count). Limits: a thinned picture merges parallel lanes closer than about 3 cells (duplicates are invisible at picture level); the skeleton graph is undirected, so direction accuracy and directed TOPO exist only for graph heads. Needs scipy. File map: `lanes/README.md`.

## 8. Checking an adapter, the selftest
- `selftest-adapter` (`python -m evalsuite --adapter pkg.module:build selftest-adapter`): scenes and frames in order, `find()` by label / tuple / key, `load()`, `predict()` and the declared taps (finite numpy arrays), `reset()` restores the start state, every metric (counts are arrays, `reduce()` works on one, several and no frame, does not depend on the order, produces the headline keys, `keys=` agrees with the full result), every view renders a page that draws.
- `python -m evalsuite selftest [word ...]` runs the known-answer tests of every function (44 checks, 2 to 3 minutes on a busy machine; `selftest minimal`, `selftest lane`, `selftest pareto`, or any word of a check name). A failing check prints its traceback. Needs scipy.

## 9. Real adapters, files, rules
**Adapters in use**: `adapters/minimal.py` (template), `adapters/toy.py` (test bed with memory, taps, planted outlier frames, two metrics, a hand-made `ops()`), `adapters/lanes_synth.py` (synthetic road world with every error cause planted), `bev_fusion/evaluation/bev_adapter.py` (camera + radar BEV system; its `ops()` is `count_ops` on the cached chain, its `benchmark()` times the same path), `bev_fusion_lane_center/evaluation/lane_suite.py` (lane centre lines of VAL: dense heads and registry graphs). These two bring their own entry scripts (they put the suite on the path).

**Files** (from `autoresearch/evalsuite/`):
- *core* (about 2,100 lines, numpy + matplotlib): `contracts.py` boxes and data, `suite.py`, `runner.py` order of calls and state, `sampling.py` video-then-frame draws, `store.py` run store, `stats.py` robust z, leave-one-out, bootstrap, `qualitative.py`, `quantitative.py`, `cli.py`, `render/figure.py` `report.py` `taps.py` (pictures of tapped tensors) `video.py` (H.264 with libx264, read back and checked; an animated GIF without ffmpeg).
- *small modules loaded when used*: `cost.py` (chapter 5), `probing.py` (chapter 6), `check.py` (contract checker), `torch_taps.py` (hooks and `count_ops`).
- *libraries the core never imports*: `lanes/` (chapter 7), `adapters/`.
- tests: `selftest.py`, `selftest_cost.py`, `selftest_lanes.py`.

**Rules**: the core imports only numpy (matplotlib and PIL are imported lazily where a picture is drawn, cv2 only to read a video back); an adapter may import anything. One `Subject` per process when the network code is pinned per run (module cache). Every number is a function of additive counts, so outliers, comparison and probing work on stored runs without the network.

**Design history** (the plan, the data flow, the proof plan, the migration of the old scripts): `../history/evalsuite_architecture_2026-10-30.md`.
