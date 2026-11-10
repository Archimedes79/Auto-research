import sys

mode, path = sys.argv[1], sys.argv[2]
s = open(path, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


if mode == "eval_readme":
    rep("## Use\n\n**Quick start", '''## The class interface: `Evaluation` (D70: the memory is carried, the predictions are stored)

The same evaluation as one class with the pattern of the other suites (`../../autoresearch/INTERFACE.md`; the code in `../../autoresearch/evalsuite/api.py`, this project's side is `bev_eval.py`, named in `evalsuite.json`).
From `autoresearch/` with `AR_PROJECT=../bev_fusion`:
```
python -m evalsuite Evaluation.check runs/t23_C_s1 small.val                  # the dry run: scenes, frames, what is stored, the id, the problems (exit 1 when there are)
python -m evalsuite Evaluation.run runs/t23_C_s1 small.val --gt-key v8        # predict what is not stored yet, then score; --json for a machine
python -m evalsuite Evaluation.quant t23_C_s1@e3f9a2c1d4b0 --match drivable  # the ids are printed by run and by `Evaluation.list runs/t23_C_s1`
python -m evalsuite Evaluation.compare CONTROL_ID VARIANT_ID                  # variant - control, refused on another GT or other frames
```
```python
from evalsuite import Evaluation
ev = Evaluation()                              # AR_PROJECT=../bev_fusion
r = ev.run("runs/t23_C_s1", "spread:200:large.val", gt_key="v8", metrics=("maps", "boxes", "fusion"))
r.quant(), r.by("scene"), r.worst("seg"), r.card()
```
What differs from `bev_suite.py eval`: a network with a memory runs warm, whole scenes in time order from its own start (the frames you ask for decide which results are kept and scored; `eval` without
`--warm` starts every frame cold, so the numbers of the two are not comparable); what the network predicted is stored once per network and checkpoint under `RUN/eval/results/<id>/` (about 1 - 2 MB per
frame) and a subset, a new GT or the metrics `curves` / `fusion` / `velocity` added are scored from it without the network (`predict` and `score` are the two halves of `run`); the ids are
`<run>@r<hash>` (predictions) and `<run>@e<hash>` (a scored evaluation, under `RUN/eval/suite/quant/`, readable by the older `quant` commands). The val loss is computed by the network while it
predicts, against the ground truth of that moment: a rescoring under another GT keeps it and says so (`warnings`; the card leaves `val_loss/*` out). `run` here books no ledger row and takes no GPU
slot: a number that feeds a verdict is launched with `run.py ... -- python -m evalsuite Evaluation.run ...`. Checks: `../tests/test_evaluation_api.py` (without option: the interface on a toy project,
the adapter's identity, TEST refused; `--real`: on `t16_full_s0`, CPU, the interface equals the engine's own run score for score). `evaluate.py` and `bev_suite.py` stay the front doors of the
frozen frame; `bev_eval.py` adds no logic to them (it calls `bev_adapter.build`, `bev_suite._task_only`, `bev_suite.contract_card`).

## Use

**Quick start''')
    rep("| `bev_suite.py` | the command line of this folder:", "| `bev_eval.py` + `evalsuite.json` | the BEV system as a project of the class interface `Evaluation` (see \"The class interface\"): `identity` (checkpoint fingerprint, config, code, the adapter), `source` / `subject` (via `bev_adapter.build`), `task`, `reference` (the GT without the inputs), `card`, `time`, `probe`, `resolve` (a run name or path) | `../tests/test_evaluation_api.py` |\n| `bev_suite.py` | the command line of this folder:")
elif mode == "tests_readme":
    rep("| `test_evaluate_door.py` |", "| `test_evaluation_api.py` | the class interface `Evaluation` on the BEV system (`autoresearch/evalsuite/api.py`, adapter `evaluation/bev_eval.py`): without option (CPU, seconds) the interface's own checks on a toy project (`python -m evalsuite.selftest_api`: frames, result set and writer lock, run = the engine's run on the kept frames, reuse of stored predictions also for a subset, rescoring under another GT without the network and the stale-loss warning, a run killed half-way resumes with the right memory, predict + score = run, `check` finds problems first, the command line: clean json, exit codes, no collision with the older words), the adapter's identity (a retrained checkpoint of the same name is another one), the run resolution, TEST and unknown metrics refused; `--real [RUN]` (CPU, 4 frames, about 5 minutes): the interface on the real run equals the engine's own run score for score, the stored predictions are reused, the fusion metrics added later are scored from them, the card | all checks pass | passes (2026-11-10) |\n| `test_evaluate_door.py` |")
elif mode == "gates":
    rep('''run "evaluate door"     "$PY" tests/test_evaluate_door.py ''', '''run "evaluation api"    "$PY" tests/test_evaluation_api.py          # the class interface (toy project, the adapter's identity, TEST refused); no data, seconds
run "evaluate door"     "$PY" tests/test_evaluate_door.py ''')
    rep('''    run "evaluate door, sets"   "$PY" tests/test_evaluate_door.py --sets''', '''    run "evaluation api, real" "$PY" tests/test_evaluation_api.py --real   # the class interface equals the engine on a real frame (CPU, 4 frames, about 5 min)
    run "evaluate door, sets"   "$PY" tests/test_evaluate_door.py --sets''')
open(path, "w", encoding="utf-8").write(s)
print("ok")
