# Conventions

One scheme for names, ids and flags, so that a path tells what a thing is. New work follows these; old files keep their names (mapping of the old layout in `experiment_execution/history/PATH_MAP.md`).

## Ids

| thing | scheme | example |
|---|---|---|
| loop | `loop<NN>` in text, `p<NN>_` in code | loop 16, `p16_builder.py` |
| experiment (catalogue) | `E<number>`, never reused; letters for variants | E94, E94b, E120a |
| experiment folder | `E<first id, 3 digits><letter>_<slug>`; the other ids of a row are in its README | `E094b_wrong-direction-states-four-heads-wavefront` |
| run (ledger) | `E<id>.<arm or letter>` or the ladder form `E123.<design>.r<rung>.<step>` | `E209.a`, `E123.D7.r3.f1` |
| run directory | `runs/e<id>_<tag>` (ladder: `runs/e123/e123_<design>_r<rung>_<step>`) | `runs/e37_final` |
| technique | `K<number>` in the register, before it is used in an experiment | K185 |
| building block / stage | `BB<nn>`, `SP<n>` (catalogue tables) | BB11, SP5 |
| paper note | `<cluster prefix><number>` (GR growth, TP topology, VM vector-map, VS verification, MH hypotheses, SN sensors, PL planner, A / RB / C loop-17 clusters) | GR01, RB01 |
| seeds | `s0`, `s1`; at least 2 route-head seeds for any comparison (seed spread up to 0.09) | `_route_s0` |
| splits | TRAIN 8,414 / VAL 873 / TEST 833 frames, scene-disjoint (`data/splits.json`); TEST (21 scenes) once | |

Loop 17 note: its work packages were run under the internal names E130-E144 (`runs/e17/e130_*` ...) and are catalogued as E193-E212 (mapping in `experiments/E193_*/README.md`).

## Names

- Folders and files: lower-case, `snake_case` or `kebab-case`, no spaces, ASCII. Dates `YYYY-MM-DD`. Documents in English, chat in German.
- Code (`training/`, `evaluation/`, `tools/`, `reports/build/`, short-lived in `temp/`): `p<loop>_<module>.py` = loop-specific library or entry point; `fig_p<loop>_*.py` figures; `loop<NN>_text*.py` + `build_loop<NN>_report.py` report parts; `_run_*.sh` launchers of experiments; a leading `_` = helper nobody imports; `training/data/prep/` = builders of cached data; short-lived scripts live in `temp/<kind>/loopNN/`. Flat imports (`import graph_eval as GE`): the launcher puts every code folder on `PYTHONPATH`, so a module can move between folders.
- Reports: `Loop<NN>_<Topic>_Report_EN.pdf` or `<Scope>_Report_EN.pdf`; build sources in `reports/source`.
- Docs per folder: every folder that humans open has a `README.md` (what is here, how it is made).

## Runs and flags

- A finished run directory holds `log.txt`, `status.json` or `metrics.json`, the model and - only for a run that ended with exit code 0 - a `.done` flag. The launcher `tools/experiment/run.py` does this. A flag that is known to be false is renamed `<name>.done.FALSE` and explained in the ledger (E209: `runs/e17/logs/e142.done`).
- Long trainings write a checkpoint every 400-500 iterations and can resume; wrappers never write the flag after a crash.
- At most 2-3 GPU jobs, GPU memory below 11 GB (RTX 3060 12 GB); timing only on an idle GPU, batch 1, cached foundation features.

## Experiments

- Ledger row BEFORE the run (`tools/experiment/ledger.py add ...`), result + decision after; catalogue row (E-id) and K-id for every new technique; every finished run stays in the ledger with its reason, also failed and cut ones.
- A comparison needs two seeds, differences below the noise band (0.010 in S) are ties; select on VAL only; TEST once, winner + control, only with an explicit OK of the user.
- Every experiment reports quality and ms per frame; optimisations must give identical outputs (`p16_fast_check.py` is the pattern).
- Graph outputs are scored with `evaluation/sparse_lanes` (graph dict: `pts`, `edges`, `nodes`), dense maps with `evaluation/lane_grid`; do not write a new metric - extend those.

## Files and deletion

- Nothing is deleted without the user's OK: unneeded files move to `_to_delete/` with a line in its README; the user purges.
- Generated files say so in their first lines and name the generator (`STRUCTURE.md`, `models/REGISTRY_ALL.md`, `experiments/README.md`, `experiment_execution/LEDGER.md`, `literature/papers/INDEX.md`).
- Rollback hazard: a conversation rollback in Claude Code removes the files that session created with Write / Edit (the loop-17 code was lost this way). New code and results of a session are copied with `cp` (or written by a script) as soon as they matter, and `HANDOVER.md` is updated at every milestone.
