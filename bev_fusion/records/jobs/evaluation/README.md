# jobs/evaluation - how the evaluation is called (class `Evaluation`, `autoresearch/Evaluation`; BEV adapter `evaluation/bev_eval.py`)

The evaluation takes arguments, not job files. It scores what the network predicted; the network itself runs in the project's `Runner` (`bev_fusion/runner`, see `../running/README.md`), which the
`Evaluation` always includes (`evaluation/evaluation.json`: `"runner": "runner.runner:Runner"`). The docstring of `autoresearch/Evaluation/api.py` is the reference; `python -m Evaluation methods` lists every command.

```
cd autoresearch                                                  # from the workspace root, inside wsl; `python` = ~/miniconda3/envs/gcar/bin/python
AR_PROJECT=../bev_fusion python -m Evaluation Evaluation.check runs/t23_C_s1 small.val                   # the dry run: what would run, what is wrong with the ground truth and the metrics
AR_PROJECT=../bev_fusion python -m Evaluation Evaluation.run   runs/t23_C_s1 small.val --gt-key v8     # run the network (what is not stored yet), store the predictions, score them
AR_PROJECT=../bev_fusion python -m Evaluation Evaluation.score <results id> small.val --gt-key v8      # only score what was stored (another GT or metric needs no network)
AR_PROJECT=../bev_fusion python -m Evaluation Evaluation.show-numbers <evaluation id>                    # the numbers of a stored evaluation
```

| Mirko's sketch | here |
|---|---|
| `resultSet = runner.run(network, dataset_json, settype)` | `Runner.predict(network, frames)` -> the stored predictions (a `ResultSet`, `runs/<name>/eval/results/<id>`) |
| `Evaluation.run(runner, network, dataset_json, setId / seq1, seq2 / n, gt_key)` | `Evaluation.run(network, frames, gt_key)` with `frames` = a set (`small.val`), `scenes:a,b:val`, `random:3:7:val` (3 random scenes, seed 7), `spread:1000:large.val`, `frames:scene-0061/12:val` |
| `getQuantEvalFromLastRun`, `getQualEvalFromLastRun`, `getCalcTimeFromLastRun` | `show_numbers(id)`, `draw_frame(id, frame)`, `get_cost(id)` / `time(network)`; ids instead of "the last run" |

A network with a memory always runs warm (D70): whole scenes in time order from the network's own start; `frames` only decides which results are kept and scored. There is no cold mode. The run has no
ground truth and no validation loss (D72), so `loss.*` / `val_loss/*` are not in the numbers of `Evaluation` (`bev_suite.py eval` keeps them); the BEV side of that run is `evaluation/bev_free.py`
(`FreeSource` / `FreeSubject`, outputs bit-equal to the live subject on real frames: `tests/test_bev_free.py`). `runner/bev_runner.py` stays the planner's runner (boxes and dt-0 maps).

**Checked** (reference numbers: `runs/t20_base_s0/eval/timing_modules.json`, idle GPU, dts -0.5 / 0 / +0.5: inference 102 ms per frame, forward with loss 153 ms; the whole `numbers` step of the old live
path took 962 s for 1000 frames = 0.96 s per frame, i.e. the network is about a seventh of it):
1. equivalence: the free subject's outputs equal the live subject's bit for bit (`tests/test_bev_free.py`, real frames);
2. the memory check on the real network: the confidence of the first frame of a scene is 0.00 and grows (0.00 / 0.15 / 0.29 on `t20_base_s0`, `tests/test_runner_real.py`);
3. the run alone, without the Evaluation class, on frames without labels (`python -m runner Runner.predict`), and as the `Evaluation`'s helper (`tests/test_evaluation_api.py`);
4. the interface: three calls for the whole path (`Runner.predict`, `Evaluation.score`, `Evaluation.show_numbers`), no cold option, errors as one error type with exit code 2, one command-line generator (`tests/test_suites_conform.py`).

**Not measured yet:** the speed of the whole path on an idle box (ms per frame split into data, network, post-processing, storing; megabytes per frame with and without a tap; the same scene stored twice
must cost no network). It needs the GPU alone, through `run.py --no-card` as a diagnostic.
