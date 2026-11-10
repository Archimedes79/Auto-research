# Job layer on top of the existing suites - proposal (2026-11-10)

**Status (2026-11-10, later the same day):** Mirko ordered it built ("create a directory for a JobRunner ... and generate the code there"). Version 1 is `autoresearch/jobrunner/` in the thin form of the revised recommendation below (no planner, no store of its own, run.py is the queue); its `README.md` is the reference, this text is the design history. What the sections below promise beyond version 1 (priorities, `stop` / `hold` / `resume`, a CPU gate, the GT suite in the catalogue) is not built.

Asked by Mirko: how a job is defined and how it is built on top of the system, "without redefining" what exists. This is a proposal for his reading; it is not a decision (no D<n>) and not a card. D74 holds all experiments until he says the rebuild is complete; this layer is rebuild work, so it starts only on his word. Written from the READMEs and the `run.py` docstring, not from a full read of the code (see "Not checked").

## Revised recommendation (same day, after Mirko asked "what is the best architecture?")
The sections below describe the larger variant with a Planner. On a second look the Planner duplicates `run.py`: a launcher already waits for its `--after` ids, for a GPU slot (one FIFO ticket line shared by both problems), the RAM gate and the machine lock, and `hold.py` can stop a waiting launcher. A job layer without a daemon is therefore enough and has fewer ways to fail (the review's 14 findings in the Training queue were all in a layer that starts and orders runs on top of `run.py`):
- **Spec + catalogue** (new knowledge: what is a job, what does it need, `launch` or `process`),
- **`submit`**: validate, write `job.json`, start `run.py` in the background (it waits in its own line), return the id,
- **`list` / `status`**: read `run.py`'s own run record (status, log, ledger row, outputs); no second store.
About 300 - 400 lines, no process that has to stay alive. The Planner, a separate job store and priorities are added only if an incident shows that FIFO and the waiting launchers are not enough. Open point to settle first with a small spike (one throwaway script, not started because of D74): that `run.py` accepts `python -m evalsuite Evaluation.run ...` with `--no-card`, `--gpu-max-mb`, `--ram-mb`, `--after` as its docstring says.

## The rule: add one layer, change nothing below it
Nothing that exists is renamed, re-signed or moved. The suites keep their classes and commands, `run.py` keeps its gates, the training job files keep their keys, `Training.queue_add` keeps working. The new layer only reads them.

```
agents / people        python -m jobs ...   or   Jobs(project)
------------------------------------------------------------------ NEW
job layer              spec | catalogue | store | planner
------------------------------------------------------------------ exists, unchanged
run.py                 GPU slots, one FIFO ticket line, RAM gate, machine lock, --after, ledger row, tag, card gate, TEST gate
suites                 AutoGT.generate | Training.run | Runner.predict | Evaluation.run   (every method is also `python -m <suite> Class.method`)
resources              GPU (2 slots, shared with the lane project) | RAM | CPU (no gate) | disk
```

## What a job is
A job is data, not a class: a json file (or the same dict), in the style the suites already use ("a job is a json file or the same dict", `autoresearch/README.md` rule 4; unknown key = error; keys starting with `_` are comments).

| field | answers | notes |
|---|---|---|
| `id` | which job | unique; becomes the job folder and, for a run, the run id |
| `call` | what | `Class.method` of a suite, e.g. `Evaluation.run` |
| `args` | with which parameters | exactly the parameters of that method; checked against its signature (the registry `suitecli.py` already builds the commands from) |
| `needs` | what it takes from the machine | `gpu`, `gpu_mb`, `ram_gb`, `cpu`, `est_hours`; the suite's `check` proposes it, the author may override |
| `after`, `priority` | when | `after` = job ids that must be done OK first |
| `outputs` | what it leaves | folder / flag that means done; present = the job is skipped; same `id` with other `args` is refused |
| `card` or `no_card` | may it feed a verdict | required for a call the catalogue marks `verdict_capable`; a TEST split needs its logged row |
| `kill`, `on_fail` | guard | kill rules only where the suite supports them (training) |

Old training job files stay valid as they are (`kind: train` is read as `call: Training.run`); the new fields are optional for them.

Two examples:
```json
{"id": "t23.C_s1.eval", "call": "Evaluation.run",
 "args": {"network": "t23.C_s1", "frames": "large.val"},
 "needs": {"gpu": 1, "gpu_mb": 4000}, "after": ["t23.C_s1"], "outputs": ["runs/t23_C_s1/eval/"]}

{"id": "gt.l6", "call": "AutoGT.generate",
 "args": {"data_root": "~/data/nuscenes_full", "version": "l6"},
 "needs": {"cpu": 6, "ram_gb": 10}}
```
A job lives as a folder `records/jobs/queue/<id>/` with `job.json` (the recipe, in git), `status.json` and `log.txt` (not in git). States: planned, queued, running, done, failed, stopped, held.

## The four parts of the layer (a new small suite, `autoresearch/jobsuite/`, class `Jobs(project)`)
1. **Spec**: load, validate (keys, types, signature of the called method, `after` ids exist), `to_argv(job)`.
2. **Catalogue**: each suite declares which of its methods are jobs and their default `needs` / `verdict_capable` in a small table next to its class (owned by the suite's chat; a dozen lines each). Methods not listed are not jobs.
3. **Store**: the job folders; `submit`, `list`, `status`, `stop`, `hold`, `resume`, `remove`.
4. **Planner**: takes the next job whose `after` is done and whose `needs` fit, and launches it through `run.py` as a separate process: `run.py --id <id> [--gpu --gpu-max-mb N] [--ram-mb N] [--after ...] -- python -m <suite> Class.method --json <args>`. The result json goes to `status.json`; the ledger row, tag, card and TEST gates are `run.py`'s own, so a job cannot bypass them (hard rule 3).

Verbs follow the suites' common core (`autoresearch/README.md` rule 6): `check` looks and starts nothing, `run` does it now, `submit` queues, `status`, `stop`, `selftest`; every method is also a command generated by `suitecli.py`.

## How it sits next to the Training queue
`Training.queue_add` / `queue_list` / `queue_hold` stay as they are. For `call: Training.run` the planner hands the job to that queue (no second queue for training). Whether the training queue is folded into the job layer later is a separate question and not needed for step 1 - 4.

## Build order (each step additive, with its own selftest and a planted-bug test; a critic pass before use)
1. Spec, validator, `to_argv`: pure functions. Tests: unknown key, wrong type, signature mismatch, same id with other args, golden argv for the examples.
2. Catalogue entries for `Training.run`, `Evaluation.run`, `AutoGT.generate`, `Runner.predict`; `check` returns the proposed `needs`.
3. Store plus `submit` / `list` / `status`, and ONE real job through `run.py` as a diagnostic (`no_card`): an evaluation of a stored run, then a GT dry run.
4. Planner: priority, `after`, fit of `needs`; `hold` / `resume` / `stop`.
5. CPU gate (the only change to something that exists, additive and off by default): a `--cpu N` slot like the GPU slot, plus thread limits in the job's environment. An older planner had `cpu_slots`; it was lost in the D16 simplification. Skip this step if no incident shows the need.

## Who builds what (suggestion)
Training chat: steps 1, 3, 4 (it owns `run.py`, the queue and the training job files). Each suite's chat: its catalogue entries (step 2). Evaluation chat: the `needs` of `Evaluation.run`. GT chat: `AutoGT.generate`. Me: this text, the review of the spec, the comparison rule between trainer versions stays as written in memory.

## Where the classes sit (Mirko, 2026-11-10: "a job can also contain these classes, it just has to execute them somehow")
Composition, not inheritance: a job names a suite class and a method, the job layer holds a registry `suite name -> class` and builds the instance when it needs one (`Training(project)`, `Evaluation(project)`). Two uses of the instance:
- `check` runs in the planner's own process (cheap, no GPU): validation of the arguments and the proposed `needs`.
- `run` has two forms, declared per catalogue entry: `launch` (the method itself starts a `run.py` run and returns, as `Training.run` does) or `process` (the planner starts `python -m <suite> Class.method` through `run.py`; for long or GPU methods such as `Evaluation.run`, so that a crash or an out-of-memory does not take the planner down and the slots, ledger row and gates apply). An in-process call of a long method is not offered.
```python
class Job:                                   # a record plus how to run it; no base class for the suites
    def __init__(self, spec, project): self.spec, self.project = Spec.load(spec), project
    def suite(self):  return SUITES[self.spec.suite](self.project)          # the class the job contains
    def check(self):  return getattr(self.suite(), "check")(**self.spec.check_args())
    def run(self):
        entry = CATALOGUE[self.spec.call]
        if entry.how == "launch":  return getattr(self.suite(), self.spec.method)(**self.spec.args)
        return run_py(self.spec.id, entry.needs(self.spec), argv=self.spec.to_argv())      # subprocess under run.py
```
A job is ONE call. A chain such as GT -> train -> evaluate is several jobs joined by `after` (a named list of jobs may be submitted together), because each step needs other resources, is restarted on its own and gets its own ledger row.

## Limits and risks
- The need is not shown by an incident: GT and evaluation have not failed for lack of a queue; the real driver is many chats on one machine, where CPU and RAM run on agreement only. A smaller option exists (B): one helper `box_ok(cpu, ram)` that lets heavy starts wait. This proposal is the larger option (C) in its light form.
- The review found 14 findings in the Training queue (hold race, kill rules not evaluated by the queue, a fixed settle clock); the Training chat reports them fixed with tests (7c540436). The planner must not be built on top of that code before it is re-read.
- Resume differs per call (training: checkpoints; GT and evaluation: stores). It is declared per catalogue entry, not guessed.
- Two chats must not both submit a job that holds the same output: `outputs` plus the id rule catches it, the machine lock does not.

## Not checked
`run.py` was read only in its docstring (1308 lines): that it accepts any command (not only training) with `--gpu`, `--ram-mb`, `--after` and the card gate is taken from there. `suitecli.py` was read as an outline. Nothing was run. The step estimates are mine and untested.

## Decisions for Mirko (nothing written to `decisions/` yet)
1. Is this layer wanted at all, or is B (a load helper) enough until an incident?
2. New suite `jobsuite` (recommended, `runsuite` is the Training's `RunJobs`) or inside `runsuite`?
3. Is the CPU gate (step 5) wanted now?
4. When: after the rebuild (D74) is complete by his word, or as a part of it.
