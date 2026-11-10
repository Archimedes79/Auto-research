import sys
path = sys.argv[1]
s = open(path, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep("| `--all` | the evaluation chain of the t16 bring-up:",
    "| `--sets` | the sets that exist (the table below), no run needed |\n| `--all` | the evaluation chain of the t16 bring-up:")
rep("Also `--seed S` (default 1) and bev_suite's `--device cuda|cpu --workers N --split val|train --ckpt-dir DIR --out DIR` (the TEST split is not offered).",
    "Also `--seed S` (default 1) and bev_suite's `--device cuda|cpu --workers N --split SET --ckpt-dir DIR --out DIR` (SET: the next section; the TEST split is not offered).")
rep("Global options: `--device cuda|cpu --workers N --split val|train --ckpt-dir DIR",
    "Global options: `--device cuda|cpu --workers N --split SET (val | train | <split>.<part>, see \"The sets\") --ckpt-dir DIR")
rep("## Use\n\n**Quick start", '''## The sets: one definition, the training's

Every evaluation runs on a set of scenes. **There is one definition of the scene lists in the repository: `splits/<name>.json`** (a split = a `train` and a `val` list; `large`, `small`, `xl`; the file the training config's `"split"` names, read by `training/splits.py`). The evaluation adds no list of its own: a set is `<split>.<part>`, `part` = `train` or `val`. `python evaluation/evaluate.py --sets` (= `python evaluation/bev_sets.py`) prints them:

| set | scenes | what |
|---|---|---|
| `val` (the default), `train` | those of the run | the lists the run itself was trained and validated on (its `config.json`): as before this section existed. For a run trained on `large`, `val` and `large.val` are the same 82 scenes |
| `large.val`, `large.train` | 82, 428 | the baseline split |
| `small.val`, `small.train` | 10, 60 | the small split (every cache exists): the quick set for a first look, a smoke test or a timing |
| `xl.val`, `xl.train` | 82, 700 | the larger training split (same val scenes as `large`) |
| `test` | - | not a set: only through `run.py --split test` after a logged row under Mirko's recorded OK; `test` and `<split>.test` are refused |

A new set is a new file in `splits/` (and is then also a split the training can name), never a list in the evaluation. How to use one: `--split SET` of `evaluate.py` and of `bev_suite.py` (every command: `eval`, `qual`, `quant`, `list`, `modules`, `compare --split SET`), e.g. `python evaluation/evaluate.py runs/t16_full_s0 --quant --split small.val --n 200` or `python evaluation/bev_suite.py runs/t16_full_s0 --split large.train qual random 6`. The run, its checkpoint and its caches are the same; only the scenes change (`bev_adapter.build` puts the set's lists into the run's config before the data is made, `bev_sets.py`). What a set changes in the outputs: a named set writes files of its own (`eval/card_small_val.json`, `metrics_small_val.json`, `table_small_val.txt`, the store `eval/suite/quant/eval_small_val_n200`), so the run's plain evaluation is never touched; its card says `"split": "small.val"` and `provenance.data` names the set, the scenes and the frame hash, so `decide.py` (a card's split must be the one the experiment declares) and `compare --split small.val` keep sets apart. Not offered on a named set (they are the run's own evaluation): `--all`, `generalisation`, `report`, `memory`. Whether the features and GT of a set's scenes exist is the data's business, as for a training: a scene without cached features cannot be run (`bev_suite.py RUN --split SET list` shows what the run can read).

## Use

**Quick start''')
rep("| `bev_suite.py` | the command line of this folder:",
    "| `bev_sets.py` | the sets the evaluation runs on (`val` / `train` = the run's lists, `<split>.<part>` of `splits/`; the one definition is the training's, no list of its own): `parse`, `tag`, `lists`, `names`, `table`; `--list` / `--selftest` | `bev_sets.py --selftest`, `../tests/test_evaluate_door.py` (`--real` runs a named set against the run's own) |\n| `bev_suite.py` | the command line of this folder:")
open(path, "w", encoding="utf-8").write(s)
print("ok")
