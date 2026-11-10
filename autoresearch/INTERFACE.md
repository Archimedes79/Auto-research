# The interface of the suites: one pattern for autogt, trainsuite, runsuite, evalsuite

Mirko's wish (2026-11-09): the training, its queue and the runner should look like the other tools, simple and the same everywhere; a class whose methods are also usable from the
command line. This page is the pattern; each suite's own README is the reference for its details.

| suite | what it does | module / command line |
|---|---|---|
| `autogt` | the ground truth of the keyframes | `import autogt` / `python -m autogt` |
| `trainsuite` | train a network, the queue of trainings and runner jobs, watch, stop, hold | `from trainsuite import Training` / `python -m trainsuite` (from `autoresearch/`) |
| `runsuite` | run a trained network on frames, write the data the evaluation reads | `from runsuite import RunJobs` / `python -m runsuite` |
| `evalsuite` | pictures, numbers and cost of what a network does | `python -m evalsuite` (adapters) |

## The rules
1. **One class per suite, its methods are the interface.** `Training(project)`, `RunJobs(project)`; the project folder is the only state (default `AR_PROJECT`). Method names are snake_case,
   verb first (`check`, `run`, `status`); a family has a noun prefix (`queue_add`, `queue_list`). The same methods are also functions of a default object for short scripts:
   `trainsuite.check(job)` is `Training().check(job)`. (autogt has the class `AutoGT` and the same calls as module functions: `autogt.generate_bev_gt`, `autogt.gt_key`.)
2. **Plain data in and out.** A function takes paths, ids and dicts and returns a dict or a list (json-serialisable: numbers, text, lists, dicts). It never prints. The command line prints.
3. **One error per suite** (`TrainingError`, `RunJobsError` = the same class) whose message says what to fix; every method turns the queue's own error into it. The command line prints it to
   stderr and exits with 2.
4. **A job is a json file or the same dict.** It says what to do and with what: `id`, `kind`, `config`, `preset`, `set`, `card`, `after`, `priority`, `kill` ... An unknown key is an error
   (a typo shows). Keys that start with `_` are comments.
5. **Every method is ONE command, the same call as in Python, with the class name in front.** `python -m trainsuite Training.queue_add job.json` is `Training().queue_add("job.json")`
   (evalsuite: `python -m evalsuite Evaluation.run ...`). Short forms exist where no older command has the word: `queue_add job.json`, and the hyphenated `queue-add job.json` (autogt and
   evalsuite hyphenate their multi-word commands too). The arguments called `job`, `i`, `a`, `b` are positional, every other one is an option (`--words`, `--wait`, `--kill` repeatable,
   `--no-start` for a switch that is on by default); `--project` names the project folder. `python -m trainsuite methods` lists the methods with their arguments; the map is in the head of
   `trainsuite/apicli.py`. Every command takes `--json` and then prints exactly what the method returns (NaN and Infinity become null), otherwise a short text (the older words keep their own
   output). Exit codes: 0 done, 1 `check` found problems, 2 an error (a message on stderr, no traceback; `TRAINSUITE_TRACEBACK=1` shows it). Hold and resume without an item mean every item
   in Python; on the command line that needs `--all`, so a forgotten id does not hold everything.
6. **The same verbs everywhere.** The core every suite has: `check` looks and changes nothing (a preview file at most), `run` / `generate` / `predict` does the work now, `status` reads
   one thing, `stop` asks one run to checkpoint and stop, `selftest` checks the suite on synthetic data. The `queue_*` family (`add`, `list`, `get`, `remove`, `start`, `hold`, `resume`)
   belongs to `Training` and `RunJobs` only: the other suites have no queue.
7. **`i` addresses an item.** The id (`t23.C_s1`) or the run folder name (`t23_C_s1`) first; the number shown by `queue_list` (1 = first) when the text is no id or `i` is an int (an id
   made of digits stays an id). Without `i`, `status` and `stop` mean the running training, also one that was started outside the queue (it holds a GPU slot of this project). The numbers
   shift when another chat adds an item with a higher priority: a script uses the id, the number is for a person who has just read the list. `queue_hold` and `queue_resume` without `i` mean all (the command line wants `--all`).
8. **The rules of the project stay rules.** Hold and resume need the person's own words (`words` is required); a pause (`queue_hold` of all) and a held item also forbid a direct `run`; a run
   that may feed a verdict needs its card; the TEST split needs its logged row; a runner job is no training (hold leaves it alone). An interface that made them optional would be wrong.

## Where the methods are listed
The rules above are the convention; each suite's README is the reference for its methods, its job file and its commands (one place each, `python -m <suite> methods` lists them):
- Training: `trainsuite/README.md` section 0 (the methods next to Mirko's sketch, the job file, the commands).
- RunJobs: `runsuite/README.md`.
- Evaluation: `evalsuite/README.md`.

## Where the other suites stand
- **autogt** has the class `AutoGT` and the module functions `generate_bev_gt`, `generate_lane_gt`, `gt_key`; its commands are hyphenated (`generate-bev`), each one call of a function. Open from
  here: all commands as members of the class, `--json` on every command, a `status` of a running generation. The GT signature is a hash of the generator's source, so changes belong with the GT chat.
- **evalsuite** is built from adapters (`Source`, `Subject`, `Task`) and a command line (`qual`, `quant`). The evaluation chat is building the class `Evaluation` in this pattern (not committed when
  this page was written; its own README is the reference). A change belongs with the evaluation chat.
