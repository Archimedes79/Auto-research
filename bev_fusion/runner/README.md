# runner - the five-layer system run on frames, from a run's own code

The product code stays in `../training/` (precompute, interfaces, network slots, extraction) and is snapshotted with every run. This folder is only the shell that runs it: it takes no part in a training and holds no model logic. What needs the data loaders or the trainer (`network_inputs`: the dataset's frames through the precompute step) lives here, so that `model.py` and `losses.py` stay a closed bundle (`../tests/test_bundle.py`).

| file | what it is | self-test |
|---|---|---|
| `bev_runner.py` | `check`: the same input built and checked, nothing written (and the radar fan compared with the ground truth's own radar layer); `precompute`: layer 1 alone (the network input of chosen frames, no ground truth, one `.pt` per frame + `index.json`); `network`: layers 2 - 5 alone (a frames folder through the run's EMA network in time order, the memory carried between consecutive keyframes of a scene); `full`: both, frame by frame. Writes `objects.json` (boxes at every dt in the ego frame, planner records) and `maps.npz` (probability maps of dt 0). Guide: `python runner/bev_runner.py` | `../tests/test_runner.py` (a real run against its stored prototype) |

```
python runner/bev_runner.py full runs/<name> --scene scene-0003 --first 3 --out temp/runner_try
python runner/bev_runner.py precompute runs/<name>/config.json --scene scene-0003 --first 3 --out temp/frames
python runner/bev_runner.py network runs/<name> temp/frames --out temp/runner_try
```

The code that runs is the run's snapshot (`runs/<name>/code`, so a run is executed by the code that trained it); `--code training` runs its checkpoint under the live code instead. A snapshot from before the five-layer build has no `precompute.py` and is refused with that hint. The ego step between frames and the true gap to +0.5 s come from the poses (`batch["gap_steps"]`), the way a deployed system supplies its own odometry. `--cold` restarts the memory at every frame (the suite's cold mode). The ground truth is not read in `full`; `precompute` reads it only to check the radar fan against the ground truth's own layer (the frames it writes carry none). For the evaluation of a run keep using `../evaluation/` (`bev_suite.py`, `bev_prototype.py`): it scores against the ground truth; this runner does not.
