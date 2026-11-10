"""Scratch: bring records/jobs READMEs to the state of 2026-11-10 (the Runner is bev_fusion/runner, runsuite is commented out)"""
R = "/mnt/c/Auto-research/bev_fusion/records/jobs/"


def edit(path, pairs):
    s = open(R + path, encoding="utf-8", newline="").read()
    eol = "\r\n" if "\r\n" in s else "\n"
    for old, new in pairs:
        old, new = old.replace("\n", eol), new.replace("\n", eol)
        assert s.count(old) == 1, (path, old[:80])
        s = s.replace(old, new)
    open(R + path, "w", encoding="utf-8", newline="").write(s)


edit("README.md", [
    ("# jobs - the job files of the training system, and where the other two suites are called from",
     "# jobs - the job files of the training system, and where the Runner and the Evaluation are called from"),
    ("| `running/` | `runsuite` (`RunJobs`) | job files of kind `run`: the planner-facing runner `runner/bev_runner.py` (boxes and dt-0 maps, no ground truth) in the same queue | built (same) |",
     "| `running/` | `bev_fusion/runner` (`Runner`) | no job files: `python -m runner Runner.predict` takes arguments (the network on frames, no ground truth, stored for the evaluation); the one example file belongs to the planner's runner `runner/bev_runner.py`, whose job layer `runsuite` is commented out (2026-11-10) | built 2026-11-10 (one class, a command line above it) |"),
    ("running/*.json    --RunJobs.run / queue_add---->   objects.json + maps.npz of chosen frames (the planner's data)",
     "Runner.predict(network, frames)  -->  runs/<name>/eval/results/<id>   (the stored predictions: a ResultSet, evalsuite/results.py)"),
])

R_RUNNING = '''# jobs/running - the Runner (`bev_fusion/runner`, class `Runner`)

The trained network on frames, from the run's own code snapshot, with no ground truth and no validation loss (D72), always warm (D70). The Runner takes arguments, not job files; the command line is
`python -m runner` (from the `bev_fusion` folder, inside wsl, interpreter of the project), every method of the class is a command:

```
python -m runner Runner.plan    runs/t23_C_s1 small.val                    # the dry run: what is stored, what would run, what is wrong
python -m runner Runner.predict runs/t23_C_s1 small.val --taps fused       # run what is not stored, store it (the id <network>@r<hash>), Evaluation.score reads it
```

A run that feeds a verdict goes through `autoresearch/tools/run.py` with that command as its command; `predict` called directly is a diagnostic (no ledger row, no GPU slot).
Reference: `../../../runner/README.md`.

| file | what |
|---|---|
| `t20_val8.json` | a job file of the planner's runner `runner/bev_runner.py` (frozen): the reference network `t20_base_s0` on 8 VAL frames, dt 0 and +0.5 s, mode `full`; its job layer (`autoresearch/runsuite`, class `RunJobs`) is commented out since 2026-11-10 (Mirko: "auskommentieren, nicht loeschen"), the file is kept as the record of the format. The same runner is called by hand: `python runner/bev_runner.py full runs/t20_base_s0 --n 8 --dts 0 0.5 --out temp/runner_try` |

Memory (D70): a network with a memory is initialised by its own cold start at the start of every scene and then carries its state from keyframe to keyframe. `n` frames spread over the split (no `scene`)
and `--cold` of `bev_runner.py` break that chain: use them for the planner's smoke tests only, never for a number that is compared. Checked on `t20_base_s0` (3 frames of scene-0003, CPU, 2026-11-10, through
`python -m runner`, `tests/test_runner_real.py`): the memory confidence is 0.00 at the first frame and 0.15, 0.29 at the next two, so the state is empty at the start and then used.
'''
open(R + "running/README.md", "w", encoding="utf-8", newline="").write(R_RUNNING)

edit("evaluation/README.md", [
    ("AR_PROJECT=../bev_fusion python -m evalsuite Evaluation.predict runs/t23_C_s1 spread:1000:large.val     # only run and store (the Runner of Mirko's sketch)",
     "AR_PROJECT=../bev_fusion python -m evalsuite Evaluation.predict runs/t23_C_s1 spread:1000:large.val     # only run and store (the Runner of Mirko's sketch)\n# the Runner alone, from the bev_fusion folder:   python -m runner Runner.predict runs/t23_C_s1 small.val --taps fused"),
    ("the class `evalsuite.Runner` and the delegation\nof `predict` / `run` to it are being built by the Evaluation chat.",
     "the class is `Runner` in `runner/runner.py` (built 2026-11-10, `python -m runner`, `tests/test_runner_real.py`), and the delegation\nof `predict` / `run` to it is being built by the Evaluation chat."),
    ("Until the class is in, `runner/bev_runner.py` is the ground-truth-free runner (planner data only).",
     "`runner/bev_runner.py` stays the planner's runner (boxes and dt-0 maps, frozen)."),
    ("**To check when the `Runner` is finished**", "**To check when the `Runner` is wired in** (items 3 and 4 are done: `tests/test_runner_real.py`)"),
])
print("ok")
