import sys

path = sys.argv[1]
s = open(path, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("| `evalsuite` | pictures, numbers and cost of what a network does | `python -m evalsuite` (adapters) |",
    "| `evalsuite` | what a network predicts, and pictures, numbers and cost of it | `from evalsuite import Evaluation` / `python -m evalsuite` (the class; the adapter commands `qual`, `quant` stay) |")

new_section = '''## Evaluation (evalsuite): the methods
`Evaluation(project)` (`evalsuite/api.py`; the command line `evalsuite/apicli.py`, the project side `<project>/evaluation/evalsuite.json` naming an adapter module: for BEV `evaluation/bev_eval.py`).
One base class: only `predict` (hence `run`), `time`, `probe` and `qual_frame` execute the network, everything else reads what was stored. A network with a memory always runs warm (D70);
what was predicted is stored once per identity (network, checkpoint, inputs) and every question about it, a subset, another ground truth, another metric, reuses it. `run` is `predict` + `score`.
Ids: `<network>@r<hash>` the predictions, `<network>@e<hash>` a scored evaluation; the same question gives the same id. Frames: `val`, `small.val`, `spread:1000:large.val`, `random:3:7:val`,
`scenes:a,b:val`, `frames:scene-0061/12:val` (the set is last).

| method | command line | what it does |
|---|---|---|
| `check(network, frames, metrics)` | `check NETWORK [FRAMES]` | the dry run: scenes and frames, what is stored, whether the network would run, the id it would give, `problems` (exit 1 when there are) |
| `predict(network, frames, ...)` | `predict NETWORK [FRAMES]` | run the network on what is not stored yet and store it |
| `score(results, frames, gt_key, metrics)` | `score RESULTS [FRAMES]` | score stored predictions against the ground truth; no network |
| `run(network, frames, gt_key, metrics, ...)` | `run NETWORK [FRAMES] --gt-key v8` | `predict` then `score`: a Result (a dict with the id, whose methods read the evaluation) |
| `quant(id, match)`, `by(id, axis)`, `worst(id, match, n)` | `quant ID`, `by ID`, `worst ID` | the numbers, the numbers per scene or frame, where they come from |
| `compare(control, variant)` | `compare CONTROL VARIANT` | paired bootstrap, variant - control; refused on another GT or other frames |
| `cost(id)`, `report(id)`, `card(id)` | `cost ID`, `report ID`, `card ID` | ms per frame, the html page, the metric card in the project's contract |
| `qual_frame(id, frame, view)` | `qual-frame ID FRAME` | one picture (the network runs on that frame with its scene's history) |
| `time(network)`, `probe(network)` | `time NETWORK`, `probe NETWORK` | ms per module on an idle GPU, the module probe |
| `sets()`, `list(network)` | `sets`, `list NETWORK` | the sets a project defines, what is stored for a network |

`run` here is not `autoresearch/tools/run.py`: it books no ledger row and takes no GPU slot; a number that feeds a verdict is launched with `run.py ... -- python -m evalsuite Evaluation.run ...`.

'''
rep("## Where the other suites stand\n", new_section + "## Where the other suites stand\n")
rep("- **evalsuite** is built from adapters (`Source`, `Subject`, `Task`) and a command line (`qual`, `quant`); it has no function-level `check` / `run` / `status` yet. Same remark: a change\n  belongs with the evaluation chat.",
    "- **evalsuite** has the class `Evaluation` since 2026-11-10 (the section above): `check` / `predict` / `score` / `run` and the reading methods, hyphenated commands, `--json` with nothing else on\n  stdout, exit codes 0 / 1 / 2, `EVALSUITE_TRACEBACK=1`. What it lacks, seen from here: a queue (the trainsuite one does not know evaluation jobs) and a `status` of a running `run`. The older\n  adapter commands (`qual`, `quant`, `list`, `probe`, `selftest`) keep their words; a change belongs with the evaluation chat.")
open(path, "w", encoding="utf-8").write(s)
print("ok")
