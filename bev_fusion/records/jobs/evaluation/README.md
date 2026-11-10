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

Two things Mirko asked for on 2026-11-10 that are not in the interface yet (sent to the Evaluation chat): intermediate results (taps) stored with the predictions on request, and a way to run
the network without reading the ground truth (the live subject computes a validation loss against it). Until then `runner/bev_runner.py` is the ground-truth-free runner.
