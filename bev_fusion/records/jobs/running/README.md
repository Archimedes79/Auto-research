# jobs/running - job files of kind `run` (suite: `autoresearch/runsuite`, class `RunJobs`)

The trained network on frames, from the run's own code snapshot; writes `objects.json` (boxes at every dt, planner records) and `maps.npz` (probability maps of dt 0), no ground truth read. It is the
planner-facing runner (`runner/bev_runner.py`, frozen). The predictions that the evaluation scores are made by `Evaluation.predict` instead (see `../evaluation/README.md`).

```
cd autoresearch                                                  # from the workspace root, inside wsl
AR_PROJECT=../bev_fusion python3 -m runsuite RunJobs.check ../bev_fusion/records/jobs/running/t20_val8.json
AR_PROJECT=../bev_fusion python3 -m runsuite RunJobs.run   ../bev_fusion/records/jobs/running/t20_val8.json
```

| file | what |
|---|---|
| `t20_val8.json` | the reference network `t20_base_s0` on 8 VAL frames, dt 0 and +0.5 s, mode `full` (precompute + network in one process) |

Fields: `id`, `kind` = `run`, `mode` (`check`, `precompute`, `network`, `full`), `run`, `split` + `n` or `scene` + `first`, `out`, `dts`, `device`, `cold`, `frames`, `code`
(reference: `../../../runner/README.md`, `../../../../autoresearch/runsuite/README.md`).

Memory (D70): a network with a memory is initialised by its own cold start at the start of every scene and then carries its state from keyframe to keyframe. `n` frames spread over the split (no `scene`)
and `cold: true` break that chain: use them for the planner's smoke tests only, never for a number that is compared. Checked on `t20_base_s0` (3 frames of scene-0003, CPU, 2026-11-10): the memory confidence
is 0.00 at the first frame and 0.15, 0.29 at the next two, so the state is empty at the start and then used.
