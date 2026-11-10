# The interface of the suites: one pattern for autogt, trainsuite, runsuite, evalsuite

Mirko's wish (2026-11-09): the training, its queue and the runner should look like the other tools, simple and the same everywhere; a class whose methods are also usable from the
command line. This page is the pattern; each suite's own README is the reference for its details.

| suite | what it does | module / command line |
|---|---|---|
| `autogt` | the ground truth of the keyframes | `import autogt` / `python -m autogt` |
| `trainsuite` | train a network, the queue of trainings and runner jobs, watch, stop, hold | `from trainsuite import Training` / `python -m trainsuite` (from `autoresearch/`) |
| `runsuite` | run a trained network on frames, write the data the evaluation reads | `from runsuite import Runner` / `python -m runsuite` |
| `evalsuite` | pictures, numbers and cost of what a network does | `python -m evalsuite` (adapters) |

## The rules
1. **One class per suite, its methods are the interface.** `Training(project)`, `Runner(project)`; the project folder is the only state (default `AR_PROJECT`). Method names are snake_case,
   verb first (`check`, `run`, `status`); a family has a noun prefix (`queue_add`, `queue_list`). The same methods are also functions of a default object for short scripts:
   `trainsuite.check(job)` is `Training().check(job)`. (autogt has module functions today, `autogt.generate`, `autogt.load`: the same pattern without the class.)
2. **Plain data in and out.** A function takes paths, ids and dicts and returns a dict or a list (json-serialisable: numbers, text, lists, dicts). It never prints. The command line prints.
3. **One error per suite** (`TrainingError`, `RunnerError` = the same class) whose message says what to fix; every method turns the queue's own error into it. The command line prints it to
   stderr and exits with 2.
4. **A job is a json file or the same dict.** It says what to do and with what: `id`, `kind`, `config`, `preset`, `set`, `card`, `after`, `priority`, `kill` ... An unknown key is an error
   (a typo shows). Keys that start with `_` are comments.
5. **Every method is a command with the same name.** `python -m trainsuite Training.queue_add job.json` is `Training().queue_add("job.json")`. The arguments called `job`, `i`, `a`, `b`
   are positional, every other one is an option (`--words`, `--wait`, `--kill` repeatable, `--no-start` for a switch that is on by default); `--project` names the project folder.
   `python -m trainsuite methods` lists the methods with their arguments. A method whose name no older command uses also works without the class name (`queue_add job.json`). Every
   command takes `--json` and then prints exactly what the method returns, otherwise a short text (the older words keep their own output). Exit codes: 0 ok, 1 a check found problems, 2 an
   error.
6. **The same verbs everywhere.** `check` looks and changes nothing (a preview file at most). `run` / `generate` does the work now. `status` reads one thing. `list` / `get` / `remove`
   for the items of a queue. `stop` asks one run to checkpoint and stop. `selftest` checks the suite on synthetic data.
7. **`i` addresses an item.** The id (`t23.C_s1`) or the run folder name (`t23_C_s1`) first; the number shown by `queue_list` (1 = first) when the text is no id or `i` is an int (an id
   made of digits stays an id). Without `i`, `status` and `stop` mean the running training, also one that was started outside the queue (it holds a GPU slot of this project). The numbers
   shift when another chat adds an item with a higher priority: a script uses the id, the number is for a person who has just read the list. `queue_hold` and `queue_resume` without `i` mean all.
8. **The rules of the project stay rules.** Hold and resume need the person's own words (`words` is required); a pause (`queue_hold` of all) and a held item also forbid a direct `run`; a run
   that may feed a verdict needs its card; the TEST split needs its logged row; a runner job is no training (hold leaves it alone). An interface that made them optional would be wrong.

## Training: the methods (trainsuite), with Mirko's sketch next to them
The command line column is the short form; `Training.` in front of it always works (`Training.queue_add job.json`). `status`, `stop`, `plot` and `compare` are also older commands with
another meaning: there the class name is needed (`Training.status 3`); the nested form `queue status 3` does the same.

| sketch | method | command line | what it does |
|---|---|---|---|
| `Training.run(network.json)` | `run(job)` | `run job.json` | start now, outside the queue; the same call again resumes |
| (dry run) | `check(job)` | `check job.json` | compose the config, static check, data check; starts nothing |
| `queueAdd(Training.json)` | `queue_add(job)` | `queue_add job.json` | check, then put it in the queue; the runner starts it |
| `queueGetListOverview` | `queue_list()` | `queue_list` | the items in their order, the running one, slots, machine lock, RAM, GPU memory, holds |
| `queueGet(i)` | `queue_get(i)` | `queue_get I` | one item: job, config, exact command, state |
| `queueRemove(i)` | `queue_remove(i)` | `queue_remove I` | take out an item that has not started |
| `queueStart` | `queue_start()` | `queue_start` | make sure the runner runs |
| `queueHold` / `queueResume` | `queue_hold(words)` / `queue_resume(words)` | `queue_hold --words ...` / `queue_resume --words ...` | all trainings stop with a checkpoint / continue from it (`i=` for one) |
| `queueGetStatus(i)`, `queueRunningTrainingId`, `queueRunningTrainingStatus` | `status(i=None)` | `Training.status [I]` | state, ETA, loss, gradient, skipped steps, data, health, kill rules, speed; no `i` = the running one |
| `QueueLogSpeed (?)` | `compare(a, b)` and the `speed` of `status` | `Training.compare A B` | s/step, data wait, GPU busy, peak memory, loss at the last shared step |
| (curves) | `plot(i)` | `Training.plot RUN` | the loss curves as a png |
| (stop one) | `stop(i)` | `Training.stop RUN` | checkpoint and stop one run; its id goes on hold |

Left on the command line only: `watch` (a loop for a Monitor, `--enforce` stops a run on a kill rule), `early` (advice from a control run), `compose` / `start` (the older form of
`check` / `run`), `queue add ... --id` (the older form of a job file), `queue serve` / `queue drop` / `curves` / `speed` (the older words of `queue_start` / `queue_remove` / `plot` / `compare`).

### The job file
```json
{
  "id": "t23.C_s1",
  "kind": "train",
  "config": "training/configs/t23_lane_bce.json",
  "preset": ["fast"],
  "set": {"seed": 1, "steps": 1500},
  "card": {"hyp": "...", "pred": "...", "rule": "..."},
  "after": ["t22.B2_s0"],
  "priority": 3,
  "kill": ["group.lines@1000 > 1.48"],
  "note": "second seed",
  "prepare": false
}
```
`no_card: true` instead of `card` is a diagnostic run (it never feeds a verdict). `kill` rules are checked when the job is read and appended to the run's `kill.txt` once the launcher has
survived its first seconds (where `watch --enforce` and `status` read them). `run` honours `after` (run.py waits for those runs) and ignores `priority` and `note`, which belong to the queue.
A key of the other kind (`n`, `out` in a training; `card`, `set` in a runner job) is an error.
`run_args` are arguments for `run.py` itself, `no_check` skips the project's static check, `project` names another project folder (default `AR_PROJECT`).

## Runner (runsuite): the same, for kind `run`
`Runner(project)` with `check(job)`, `run(job)`, `status(i)`, `queue_add(job)`; `queue_list`, `queue_get`, `queue_remove`, `queue_start`, `queue_hold` and `queue_resume` are the same methods as
in `Training`: one queue for trainings and runner jobs. `python -m runsuite Runner.run job.json`. A runner job names the mode
(`check`, `precompute`, `network`, `full`), the trained `run`, the frames (`split` and `n`, or a `scene`), `out`, `dts`, `device`. `full` writes `objects.json` and `maps.npz`:
the data the evaluation reads. The project says which command is its runner in `project.json` (`"runsuite": {"command": "...{args}..."}`).

## Where the other suites stand
- **autogt** already has the pattern: `generate` (run), `load`, `scene_tokens` (get / list), `data` (check), `report`, `view`, `video`, `clean`, `selftest`. What it lacks is `status` of a
  running generation and plain-dict returns for `data` and `report`. It was NOT touched: the ground-truth signature is a hash of the generator's source, and the lane chain reads it.
  Aliases (`check` for `data`, ...) belong into the next GT version, with the GT chat.
- **evalsuite** is built from adapters (`Source`, `Subject`, `Task`) and a command line (`qual`, `quant`); it has no function-level `check` / `run` / `status` yet. Same remark: a change
  belongs with the evaluation chat.
