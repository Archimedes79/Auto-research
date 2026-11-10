def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:90])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/README.md", [
    ("## Where the comparison happens\n", '''## The class interface: `Evaluation(project)`
For a project with an adapter module (`<project>/evaluation/evalsuite.json`, for BEV `evaluation/bev_eval.py`) the whole suite is one class, the pattern of `../INTERFACE.md`: plain data in and out, every method also a command.
```python
from evalsuite import Evaluation
ev = Evaluation()                                                   # the project is AR_PROJECT
ev.check("runs/t23_C_s1", "small.val")                              # the dry run: what would run, what is stored, the problems
r = ev.run("runs/t23_C_s1", "small.val", gt_key="v8")               # predict what is not stored yet (warm, D70), score it: a Result
r.quant();  r.by("scene");  r.worst("seg", n=5);  r.cost();  r.report();  r.card()
r.compare(ev.run("runs/t21_ctl_s0", "small.val", gt_key="v8"))      # variant - control, paired bootstrap
```
```
python -m evalsuite Evaluation.run runs/t23_C_s1 small.val --gt-key v8 --json        # from autoresearch/, AR_PROJECT names the project
python -m evalsuite methods                                                           # the methods and their arguments
```
Only `predict` / `run` (and `time`, `probe`, `qual_frame`) execute the network; what it predicted is stored once per identity (network, checkpoint, inputs) and every later question, a subset,
another ground truth, another metric, is scored from the store (`evalsuite/results.py`). The frames are a text (`val`, `spread:1000:large.val`, `scenes:a,b:val`, ...). Details and the adapter's
functions: the head of `evalsuite/api.py`; the methods: `../INTERFACE.md`; test: `python -m evalsuite.selftest_api`.

## Where the comparison happens
'''),
])
patch("evalsuite/EXTENSIONS.md", [
    ("- *small modules loaded when used*: `cost.py` (chapter 5),", "- *the class interface* (`README.md`, `../INTERFACE.md`): `api.py` (`Frames`, `Evaluation`, `Result`), `results.py` (the stored predictions, the writer lock), `apicli.py` (every method a command), `selftest_api.py`.\n- *small modules loaded when used*: `cost.py` (chapter 5),"),
    ("- tests: `selftest.py`, `selftest_cost.py`, `selftest_comparison.py`.", "- tests: `selftest.py`, `selftest_cost.py`, `selftest_comparison.py`, `selftest_api.py`."),
])
print("ok")
