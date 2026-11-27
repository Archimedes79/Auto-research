# Evaluation - what a network does, as numbers and pictures

One class, `Evaluation`, and one path: **predict** (the Runner is the only code that runs a network; it reads no ground truth) -> **stored
predictions** -> **score** (additive counts of every frame against the ground truth) -> **read** (dataset numbers, per scene, the worst
frames, whether B beats A, pictures, ms per frame). A project connects its network through an adapter module (`EXTENSIONS.md`). Needs numpy
and matplotlib; torch only where the network does. Run from `autoresearch/`.

## Minimal example
The adapter is `minimal.py` (a moving disc in noise, a thresholding "network", one metric); a network is a run folder with a `config.json`.
```python
import json, os, tempfile
from Evaluation import Evaluation

project = tempfile.mkdtemp()                                    # a project: its networks are run folders in runs/
for name, threshold in (("thr05", 0.5), ("thr09", 0.9)):
    os.makedirs(f"{project}/runs/{name}")
    with open(f"{project}/runs/{name}/config.json", "w") as f:
        json.dump({"threshold": threshold}, f)

ev = Evaluation(project=project, adapter="Evaluation.minimal")  # a real project names its adapter in evaluation/evaluation.json
a = ev.run("thr05", "val", device="cpu")["id"]                  # predict, store, score
b = ev.run("thr09", "val", device="cpu")["id"]
print(a, ev.show_numbers(a)["numbers"]["iou.iou"], ev.split_numbers(a, "scene"))
difference = ev.compare(a, b)["metrics"]["iou.iou"]["diff"]     # b - a, with a paired scene bootstrap
picture = ev.draw_frame(a, "demo-1/3")["image"]                 # a png from the stored prediction
```

## Commands
`python -m Evaluation <command>`: every method of the class, `_` written `-`; `--project DIR` (or `AR_PROJECT`), `--adapter`, `--json`;
exit 0 done, 1 `check` found problems, 2 an error.

| command | what |
|---|---|
| `check NETWORK FRAMES` | the dry run: what would run, what is stored, the id, the problems |
| `run NETWORK FRAMES [--gt-key V] [--metrics a,b] [--taps t] [--device cpu]` | predict what is missing, store, score; gives the id |
| `score RESULTS FRAMES` | score stored predictions again (another ground truth or metric), no network |
| `show-numbers ID`, `split-numbers ID [--axis frame]`, `get-cost ID` | the numbers, per scene or frame, ms per frame |
| `find-worst ID`, `report ID` | the scenes and frames that drag the numbers down; the same as an html page |
| `compare A B` | b - a, paired scene bootstrap; refused when source, task or ground truth differ |
| `draw-frame ID SCENE/INDEX [--view debug]` | one picture from the stored prediction |
| `list-sets`, `list NETWORK`, `tap RESULTS FRAME [--name T]` | the sets, what is stored, a stored intermediate result |
| `time NETWORK`, `probe NETWORK` | the adapter's timing per module and module probe |
| `methods`, `selftest [word]` | every command with its arguments; the tests |

## What holds
- **Ids**: `<network>@r<hash>` are predictions (network: checkpoint, config, code; the data it was fed), `<network>@e<hash>` an evaluation
  (predictions, frames, ground truth, metrics, evaluator code). The same question gives the same id and costs no network.
- **Frames**: `val`, `small.val`, `spread:N[:SET]`, `random:N[:SEED][:SET]`, `scenes:a,b[:SET]`, `frames:scene/3,scene/7[:SET]`.
- **Memory**: a network always runs whole scenes in time order from its own start (D70); the frames asked for decide what is kept and scored.
- **Numbers** are reductions of summed counts, never means of per-frame numbers. A holdout split opens only in a run of `research test`.
- One network per process: run each network on its own and compare by ids. `run` here writes no research record; a number that feeds a
  verdict comes from the evaluation step of a registered run (`research/evaluate.py`).

## Files
| file | what |
|---|---|
| `api.py`, `cli.py` | the class `Evaluation` and `Frames`; the command line generated from the class |
| `netrunner.py`, `results.py` | the Runner; the stored predictions (`ResultSet`) |
| `suite.py`, `quantitative.py`, `store.py` | the counting engine: counts per frame (`RunStore`), numbers, outliers, comparison |
| `stats.py` | the one paired scene bootstrap of the repository |
| `contracts.py`, `runner.py`, `sampling.py` | the boxes `Source`, `Subject`, `Task`; frame order and memory state; seeded subsets |
| `comparison/` | where a prediction meets the ground truth (its README) |
| `qualitative.py`, `render/` | pictures: the debug view, the figure engine (`figure.py`), tap pictures (`taps.py`), html (`report.py`) |
| `torch_taps.py` | forward hooks and operation counting for torch networks |
| `defaults.py` | names, folders and identities of run folders; the holdout gate (protected) |
| `minimal.py` | the example adapter |
| `selftest.py`, `tests/`, `selftest_comparison.py` | `python -m Evaluation.selftest` |
