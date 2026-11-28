# problems - one folder per research problem

Each is run by the core (`python3 -m research <verb> <problem>`, [`../autoresearch/research/README.md`](../autoresearch/research/README.md));
its best networks are in `../champions/<name>/`. A new problem: [`../autoresearch/EXTENDING.md`](../autoresearch/EXTENDING.md).

| folder | problem |
|---|---|
| [`toy/`](toy/README.md) | the reference problem: a disc in noise, small enough for the CPU; the chain test of the system and the template of a new problem |
| [`lane_center/`](lane_center/README.md) | Spurmitte: a lane-centre-line graph from camera + radar, end to end differentiable |
| [`bev_fusion/`](bev_fusion/README.md) | BEV Fusion: camera + radar bird's-eye-view perception on nuScenes (maps, objects, motion) |

## What a problem folder holds
Code and text only; data and run outputs are in the two roots of the machine file (`research machine`). What a problem writes for the core is
listed in the core's README ("Files a problem writes"); launching, records, verdicts, reports and provenance are the core's.

```
<name>/
  README.md  PROBLEM.md  DESIGN.md  DECISIONS.md  HANDOVER.md  TEST_LOG.md    (the toy: README.md and TEST_LOG.md only)
  problem.json   ledger.csv (generated)
  __init__.py                                                                 (empty: the folder is the package <name>)
  dataset/ gt/ net/{prep,model,loss}/ training/ evaluation/ tests/            (as far as the problem needs them; an empty __init__.py in each)
  experiments/E###_slug/{experiment.md, judge.py?, report/?, verdict.json, runs/<run>/{config,run,result}.json}
  literature/ (topics/<slug>/, see ../autoresearch/literature/README.md; documents, they never block a run)
  reports/<report>/ (a report's report.json and parts)   temp/ (scratch, not in git)
```

## How its code is found and started
A problem's code is one package named like its folder: a module imports another as `from <name>.net.model.net import DiscNet`, never by
a bare file name, never relatively, and nothing edits `sys.path`. Nothing is installed: `research` gives every process it starts the
module path of ONE tree (`problems/`, `autoresearch/`, the top of the repository; a run gets those of its snapshot, so it cannot import
the working tree). Steps and tests of `problem.json` are modules, started as `{python} -P -m <name>.<module>`.

```
python3 -m research tests <name>                                   # its tests (problem.json "tests") and its drawing, from autoresearch/
python3 -m research py <name> -m <name>.tests.test_x [arguments]   # python by hand with the problem's interpreter and module path, from any folder
```
`research/pkgcheck.py` checks the style; the problem's own test calls it (the toy: `tests/test_toy.py`). The toy is laid out this way;
`bev_fusion` and `lane_center` are not packages yet (no `__init__.py`): until their rewrite the core gives them the path they were
written for (script paths in `problem.json`, the lane's `pythonpath`). `research py <name> <script.py>` starts them as their tests are
started, without `-P`, so a script still finds the modules next to it; a package is started with `-P`.
