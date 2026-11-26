# autoresearch - the research system

Agents plan and run experiments on perception problems; code decides what counts as a result. Problem-independent: it names no problem and no
data set. The method in one page: `METHOD.md`. A problem is a folder `problems/<name>/` (the toy, `problems/toy/`, is the template).

| folder | what | entry |
|---|---|---|
| `research/` | the core: experiments, snapshots and runs, the worker, the evaluation step, the judge, adoption, the ledger, gates and the hook, reports | `python -m research <verb>`; `research/README.md` |
| `trainkit/` | the training engine: fit loop, EMA, schedule, AMP, atomic checkpoints with resume, the stop protocol (exit 75) | `trainkit/README.md` |
| `Evaluation/` | the evaluation engine: Source / Subject / Task, the default Runner, stored predictions and additive counts, the paired scene bootstrap, cost, pictures | `Evaluation/README.md` |
| `reporter/` | the report builder (markdown parts and result files to one html, every number checked) and `archdraw.py` (system drawings from a spec) | `reporter/README.md` |
| `agents/` | the four roles (`ar-planner`, `ar-researcher`, `ar-coder`, `ar-critic`); `install.py` copies them to `.claude/agents/` | `agents/README.md` |
| `templates/` | `experiment.md` (the card), `problem.md`, `paper_note.md`, `group_summary.md` | - |
| `suitecli.py` | the command line generator behind `python -m Evaluation` | - |
| `INTEGRATION_2026-11-26.md` | the restructure plan of D154 and its state | - |

## One experiment
```
python -m research new <problem> <slug> "<title>"      # experiments/E###_slug/experiment.md with the card template
python -m research check <problem> E<n>                # schema, settings of the base config, gates, the frozen card
python -m research start <problem> E<n>                # one snapshot + tag, one worker per run (steps from problem.json)
python -m research status <problem> [E<n>]             # state, heartbeat, last log row;  stop / resume
python -m research judge <problem> E<n> --write        # the snapshot's rule -> verdict.json, ledger.csv, one commit
python -m research adopt <problem> E<n> <arm> --decision D<k>    # the owner's decision: anchor test, champions/<problem>/
python -m research report <problem> <report folder>    # the report built again from records and stored predictions
```
Tests: `python -m research.selftest` (the whole chain on the toy in a throw-away repository, a few minutes; `--quick`: seconds),
`python -m Evaluation.selftest`, `python -m trainkit.selftest`, from this folder with the interpreter that has torch.

## The machine
`~/.research/machine.json` (`research machine init --data <dir> --runs <dir> --python gcar=<interpreter> --gpu-slots 2`) names where data and
run outputs live and which interpreters exist; GPU slots, holds and grants are next to it. Nothing of it is in a checkout.
