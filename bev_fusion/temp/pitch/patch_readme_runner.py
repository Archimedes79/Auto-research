def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/README.md", [
    ('''`Evaluation(project)` (`evalsuite/api.py`; the command line `evalsuite/apicli.py`, the project side `<project>/evaluation/evalsuite.json` naming an adapter module: for BEV `evaluation/bev_eval.py`).
One base class: only `predict` (hence `run`), `time`, `probe` and `qual_frame` execute the network, everything else reads what was stored. A network with a memory always runs warm (D70);
what was predicted is stored once per identity (network, checkpoint, inputs) and every question about it, a subset, another ground truth, another metric, reuses it. `run` is `predict` + `score`.''',
     '''Two classes (`evalsuite/api.py`, `evalsuite/runner_api.py`; the command line `evalsuite/apicli.py`; the project side `<project>/evaluation/evalsuite.json` naming an adapter module: for BEV
`evaluation/bev_eval.py`). The **Runner** executes the network (D72: it reads no ground truth, computes no loss) and stores what it predicts; the **Evaluation** calls it, scores the stored
predictions against the ground truth and reads the result (only `qual_frame`, which draws a frame, runs the network again). A network with a memory always runs warm (D70); what was predicted
is stored once per identity (network, checkpoint, inputs) and every question about it, a subset, another ground truth, another metric, reuses it. `Evaluation.run` is `Runner.run` + `score`.'''),
    ('''| method | command line | what it does |
|---|---|---|
| `check(network, frames, metrics)` | `check NETWORK [FRAMES]` | the dry run: scenes and frames, what is stored, whether the network would run, the id it would give, `problems` (exit 1 when there are) |
| `predict(network, frames, taps, ...)` | `predict NETWORK [FRAMES] --taps a,b` | run the network on what is not stored yet and store it, with the intermediate results `taps` when asked (a frame stored without a tap is predicted again to get it); warnings of the memory's health (D70: empty at the scene start, filling after) |
| `tap(results, frame, name)` | `tap RESULTS FRAME [--name N]` | which taps a stored frame has, or the file, shape and dtype of one |
| `score(results, frames, gt_key, metrics)` | `score RESULTS [FRAMES]` | score stored predictions against the ground truth; no network |
| `run(network, frames, gt_key, metrics, ...)` | `run NETWORK [FRAMES] --gt-key v8` | `predict` then `score`: a Result (a dict with the id, whose methods read the evaluation) |''',
     '''| method | command line | what it does |
|---|---|---|
| `Runner.check(network, frames, taps)` | `Runner.check NETWORK [FRAMES]` | the dry run of the network: scenes and frames, what is stored, how many frames would run, `problems` |
| `Runner.run(network, frames, taps, ...)` | `Runner.run NETWORK [FRAMES] --taps a,b` | run the network on what is not stored yet and store it (no ground truth), with the intermediate results `taps` when asked (a frame stored without a tap is predicted again to get it); warnings of the memory's health (D70: empty at the scene start, filling after) |
| `Runner.tap(results, frame, name)`, `Runner.list(network)` | `tap RESULTS FRAME [--name N]`, `Runner.list NETWORK` | which taps a stored frame has, or the file, shape and dtype of one; the stored predictions of a network |
| `Runner.time(network)`, `Runner.probe(network)` | `time NETWORK`, `Runner.probe NETWORK` | ms per module on an idle GPU, the module probe |
| `Evaluation.check(network, frames, metrics)` | `check NETWORK [FRAMES]` | the dry run of an evaluation: the Runner's plan plus the ground truth and the metrics, the id it would give, `problems` (exit 1 when there are) |
| `score(results, frames, gt_key, metrics)` | `score RESULTS [FRAMES]` | score stored predictions against the ground truth; no network |
| `run(network, frames, gt_key, metrics, ...)` | `run NETWORK [FRAMES] --gt-key v8` | `Runner.run` then `score`: a Result (a dict with the id, whose methods read the evaluation) |'''),
    ('''| `time(network)`, `probe(network)` | `time NETWORK`, `probe NETWORK` | ms per module on an idle GPU, the module probe |
| `sets()`, `list(network)` | `sets`, `list NETWORK` | the sets a project defines, what is stored for a network |''', '''| `sets()`, `list(network)` | `sets`, `Evaluation.list NETWORK` | the sets a project defines, what is stored for a network (predictions and evaluations) |'''),
    ("`run` here is not `autoresearch/tools/run.py`: it books no ledger row and takes no GPU slot; a number that feeds a verdict is launched with `run.py ... -- python -m evalsuite Evaluation.run ...`.",
     "`run` here is not `autoresearch/tools/run.py`: it books no ledger row and takes no GPU slot; a number that feeds a verdict is launched with `run.py ... -- python -m evalsuite Evaluation.run ...`. A bare word on the command line is the Evaluation's method when it has one, else the Runner's (`Runner.run` needs the class name)."),
])
patch("../bev_fusion/evaluation/README.md", [
    ("(`predict` and `score` are the two halves of `run`)", "(`Runner.run` and `Evaluation.score` are the two halves of `Evaluation.run`)"),
])
print("ok")
