# jobs/evaluation - how the evaluation is called (suite: `autoresearch/evalsuite`, class `Evaluation`; BEV adapter `evaluation/bev_eval.py`)

The evaluation takes arguments, not job files. What it does (the docstring of `autoresearch/evalsuite/api.py` is the reference; `python -m evalsuite methods` lists every method):

```
cd autoresearch                                                  # from the workspace root, inside wsl
AR_PROJECT=../bev_fusion python -m evalsuite Evaluation.run     runs/t23_C_s1 small.val --gt-key v8     # run the network, store the predictions, score them
AR_PROJECT=../bev_fusion python -m evalsuite Evaluation.predict runs/t23_C_s1 spread:1000:large.val     # only run and store (the Runner of Mirko's sketch)
AR_PROJECT=../bev_fusion python -m evalsuite Evaluation.score   <results id> small.val --gt-key v8      # only score what was stored (another GT or metric needs no network)
```

| Mirko's sketch | here |
|---|---|
| `resultSet = runner.run(network, dataset_json, settype)` | `Evaluation.predict(network, frames)` -> the stored predictions (a `ResultSet`, `runs/<name>/eval/results/<id>`) |
| `Evaluation.run(runner, network, dataset_json, setId / seq1, seq2 / n, gt_key)` | `Evaluation.run(network, frames, gt_key)` with `frames` = a set (`small.val`), `scenes:a,b:val`, `random:3:7:val` (3 random scenes, seed 7), `spread:1000:large.val`, `frames:scene-0061/12:val` |
| `getQuantEvalFromLastRun`, `getQualEvalFromLastRun`, `getCalcTimeFromLastRun` | `quant(id)`, `qual_frame(id, frame)`, `cost(id)` / `time(network)`; ids instead of "the last run" |

A network with a memory always runs warm (D70): whole scenes in time order from the network's own start; `frames` only decides which results are kept and scored. There is no cold mode.

Intermediate results (taps) and the memory check are built (Evaluation chat). The run without ground truth is decided (D72: a separate `Runner` class that the Evaluation class calls, always warm, the
same stored format): the BEV side is built (`evaluation/bev_free.py`, `FreeSource` / `FreeSubject`, outputs bit-equal to the live subject on real frames, 2026-11-10), the class `evalsuite.Runner` and the delegation
of `predict` / `run` to it are being built by the Evaluation chat. What the user loses: a run without the ground truth has no validation loss, so `loss.*` / `val_loss/*` are not in the numbers of `Evaluation`
(`bev_suite.py eval` keeps them). Until the class is in, `runner/bev_runner.py` is the ground-truth-free runner (planner data only).

**To check when the `Runner` is finished** (measured numbers of the reference run for comparison: `runs/t20_base_s0/eval/timing_modules.json`, idle GPU, dts -0.5 / 0 / +0.5: inference 102 ms per frame,
forward with loss 153 ms; the whole `numbers` step of the old live path took 962 s for 1000 frames = 0.96 s per frame, i.e. the network is about a seventh of it):
1. equivalence: the predictions of the `Runner` equal those of `Evaluation.predict` on the same frames (every dt: maps, distances, detections, velocity fields); scoring them gives the same numbers;
2. speed on an idle box, through `run.py --no-card` as a diagnostic: ms per frame split into data, network, post-processing, storing; megabytes per frame with and without a tap; the same scene stored twice (the second time must cost no network);
3. the memory check on the real network: the confidence of the first frame of a scene is 0.00 and grows (measured 0.00 / 0.15 / 0.29 on `t20_base_s0`);
4. the run alone, without the Evaluation class, on frames without labels; and as an `Evaluation` helper (`predict` and `run` delegate to it);
5. the interface: one entry path in three calls (`Runner.run`, `Evaluation.score`, `quant`), no cold option, errors as `EvaluationError` with exit code 2, one shared command-line module.
