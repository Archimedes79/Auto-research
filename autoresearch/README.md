# autoresearch - the research system

Agents plan and run experiments on perception problems; code decides what counts as a result. Problem-independent: it names no problem and
no data set. A problem is a folder `../problems/<name>/`.

| entry | what | read |
|---|---|---|
| `METHOD.md` | the loop, the hard rules, the roles, git | one page |
| `research/` | the core: experiments, snapshots and runs, the worker, the evaluation step, the judge, adoption, the ledger, gates and the hook, reports | `research/README.md` |
| `trainkit/` | the training engine: fit loop, EMA, schedule, AMP, checkpoints with resume, probes, the stop protocol | `trainkit/README.md` |
| `Evaluation/` | the evaluation engine: Source / Subject / Task, one Runner, stored predictions and additive counts, the paired scene bootstrap, pictures, timing | `Evaluation/README.md`, `Evaluation/EXTENSIONS.md` |
| `reporter/` | a report from markdown parts and result files, its numbers checked; `archdraw.py` draws a system from a spec | `reporter/README.md` |
| `literature/` | research on a topic: the procedure, the table of the notes, skeletons of foreign code | `literature/README.md` |
| `agents/` | the four agent roles and their installer | `agents/README.md` |
| `env/` | the python environment `gcar` as a conda export, and how a new machine builds it | `env/README.md` |
| `templates/` | the experiment card and the other file templates | `templates/README.md` |
| `EXTENDING.md` | how the system grows: a problem, a metric, a picture, a run step, a report part, ground truth, a role | one page |
| `INTEGRATION_2026-11-26.md` | the restructure of D154 - D156: its state, what was measured, kept and deleted | - |

## Tests
From this folder, with the interpreter that has torch:
```
python -m research.selftest             # the whole chain on the toy in a throw-away repository, a few minutes; --quick: the parts only
python -m Evaluation.selftest
python -m trainkit.selftest
python reporter/reporter.py selftest
python reporter/archdraw.py --selftest
python literature/topic_table.py --selftest
python literature/code_digest.py selftest
```
A problem's own tests: `python3 -m research tests <problem>`.
