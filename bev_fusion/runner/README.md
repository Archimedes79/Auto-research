# runner - a trained network run on frames

The product code stays in `../training/` (precompute, interfaces, network slots, extraction) and is snapshotted with every run. This folder is only the shell that runs it: it takes no part in a training and holds no model logic. What needs the data loaders or the trainer (`network_inputs`: the dataset's frames through the precompute step) lives here, so that `model.py` and `losses.py` stay a closed bundle (`../tests/test_bundle.py`).

```
 python -m runner Runner.predict runs/t23_C_s1 small.val --taps fused       the command line (__main__.py): every method of the class is a command
          |
          v
 class Runner (runner.py)    predict  plan  list_results  tap  time  probe          one class, no queue, no launcher; the work is in its methods
```

Like `Training` (`autoresearch/trainsuite`) and `Evaluation` (`autoresearch/evalsuite`): one class, a command line above it, the same generator (`autoresearch/suitecli.py`). The `Runner` runs the network on the
frames that are not stored yet and stores what it predicts, with no ground truth and no validation loss (D72); `Evaluation.score` scores the stored predictions later. A network with a memory always runs warm
(D70): whole scenes in time order from the scene's own start, the frames asked for only decide which results are kept. A network with a memory is initialised by its own cold start at the start of every scene
and then carries its state from keyframe to keyframe; `bev_runner.py --cold` and frames spread over the split (no `--scene`) break that chain and are for the planner's smoke tests only, never for a number that is
compared. Checked on `t20_base_s0` (3 frames of scene-0003, CPU, `tests/test_runner_real.py`): the memory confidence is 0.00 at the first frame and 0.15, 0.29 at the next two.

Which networks run: a run whose code snapshot (or, without a snapshot, the live `training/`) has what the free subject calls before the model, `train.amp_context`, `batch_ego(gap_steps, …)`,
`supports_ego` and `to_device` (`NEEDS` in `evaluation/bev_free.py`), and whose dataset makes `gap_steps` from the poses: `t16_full_s0`, `t20_base_s0`, `t23_C_s1` and everything after them (the three of
them were run through `tests/test_runner_real.py`). A run from before that (`t8_*`, `t10_*`, `t11_*`, which are the three networks of `records/keep3.json`) cannot run without ground truth: its dataset
takes the ego steps from the ground truth and its `train.py` has none of these. `plan` says so, and `predict` refuses with `ValueError: … too old for the run without ground truth` before any frame;
such a run is evaluated by `evaluation/bev_suite.py` (the live path with the ground truth).

```
cd bev_fusion                                                                              # inside wsl; `python` is ~/miniconda3/envs/gcar/bin/python, the interpreter of the project
python -m runner Runner.plan    runs/t23_C_s1 small.val                                    # the dry run: what is stored, what would run, what is wrong (exit 1 when something is)
python -m runner Runner.predict runs/t23_C_s1 small.val --taps fused                       # run what is not stored, store it: the id <network>@r<hash>
python -m runner methods                                                                   # all commands with their arguments
```

```python
from runner import Runner                              # bev_fusion on sys.path: the working folder, or PYTHONPATH=bev_fusion; the interpreter of the project (gcar)
r = Runner()                                           # the project: bev_fusion
r.predict("runs/t23_C_s1", "small.val", taps=["fused"])      # -> {results, frames_predicted, predictions_reused, warnings, ...}
```

| method | what it does |
|---|---|
| `predict(network, frames, taps, device, workers, control, progress)` | the network on the frames that are not stored yet; the intermediate results `taps` are stored on request; the health of the memory is checked per scene (a warning, never an error; `memory_checked` says whether it could be judged); a tap the network does not produce is an error before anything is stored |
| `plan(network, frames, taps, workers, control)` | the dry run of `predict`: scenes, frames kept / stored / to run (the frames before a kept one run too: the memory), the problems |
| `list_results(network)` | the stored predictions of a network: id, kind, frames, megabytes, created |
| `tap(results, frame, name)` | which intermediate results a frame has, or the `.npy` path, shape and dtype of one |
| `time(network, n)` | milliseconds per module and per frame at batch 1 (idle GPU only: the adapter refuses a busy one) |
| `probe(network, frames, device)` | statistics and dead channels per module, the response when the camera or the radar is removed |

`predict` called directly is a diagnostic: it books no ledger row and takes no GPU slot. A run that feeds a verdict goes through `autoresearch/tools/run.py` with this command as its command. The frames
text (`small.val`, `scenes:a,b:val`, `random:3:7:val`, `spread:1000:large.val`, `frames:scene-0003/0,scene-0003/1:val`) and the stored format (a `ResultSet`) are those of `autoresearch/evalsuite`.

| file | what it is | self-test |
|---|---|---|
| `runner.py` | the class `Runner`: the six commands above, built on the pieces of `autoresearch/evalsuite` (frames, `ResultSet`, the frame iterator) and the project's adapter `../evaluation/bev_eval.py` with its ground-truth-free pair `../evaluation/bev_free.py` | `../tests/test_runner_class.py` (toy network, seconds), `../tests/test_runner_real.py` (a real run, minutes), `../tests/test_suites_conform.py` (the three classes against one convention) |
| `__init__.py` | makes `from runner import Runner` work (the class lives in `runner.py`) | the same |
| `__main__.py` | the command line: `python -m runner`, nothing but the generator pointed at the class | the same |
| `bev_runner.py` | the planner-facing runner: `check`: the same input built and checked, nothing written (and the radar fan compared with the ground truth's own radar layer); `precompute`: layer 1 alone (the network input of chosen frames, no ground truth, one `.pt` per frame + `index.json`); `network`: layers 2 - 5 alone (a frames folder through the run's EMA network in time order, the memory carried between consecutive keyframes of a scene); `full`: both, frame by frame. Writes `objects.json` (boxes at every dt in the ego frame, planner records) and `maps.npz` (probability maps of dt 0). Guide: `python runner/bev_runner.py` | `../tests/test_runner.py` (a real run against its stored prototype) |

```
python runner/bev_runner.py full runs/<name> --scene scene-0003 --first 3 --out temp/runner_try
python runner/bev_runner.py precompute runs/<name>/config.json --scene scene-0003 --first 3 --out temp/frames
python runner/bev_runner.py network runs/<name> temp/frames --out temp/runner_try
```

The code that runs is the run's snapshot (`runs/<name>/code`, so a run is executed by the code that trained it); `--code training` runs its checkpoint under the live code instead. A snapshot from before the five-layer build has no `precompute.py` and is refused with that hint. The ego step between frames and the true gap to +0.5 s come from the poses (`batch["gap_steps"]`), the way a deployed system supplies its own odometry. `--cold` restarts the memory at every frame (for the planner's smoke tests only: D70 allows no cold run for a number that is compared). The ground truth is not read in `full`; `precompute` reads it only to check the radar fan against the ground truth's own layer (the frames it writes carry none). For the evaluation of a run keep using `../evaluation/` (`bev_suite.py`, `bev_prototype.py`): it scores against the ground truth; this runner does not.
