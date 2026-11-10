# jobs/running - the Runner (`bev_fusion/runner`, class `Runner`)

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
| `t20_val8.json` | a job file of the planner's runner `runner/bev_runner.py` (frozen): the reference network `t20_base_s0` on 8 VAL frames, dt 0 and +0.5 s, mode `full`; its job layer (`autoresearch/runsuite`, class `RunJobs`) is commented out since 2026-11-10 (Mirko: "auskommentieren, nicht loeschen"), the file is kept as the record of the format. The same runner is called by hand: `python runner/bev_runner.py full runs/t20_base_s0 --split val --n 8 --dts 0 0.5 --out temp/runner_try` |

Memory (D70): a network with a memory is initialised by its own cold start at the start of every scene and then carries its state from keyframe to keyframe. `n` frames spread over the split (no `scene`)
and `--cold` of `bev_runner.py` break that chain: use them for the planner's smoke tests only, never for a number that is compared. Checked on `t20_base_s0` (3 frames of scene-0003, CPU, 2026-11-10, through
`python -m runner`, `tests/test_runner_real.py`): the memory confidence is 0.00 at the first frame and 0.15, 0.29 at the next two, so the state is empty at the start and then used.
