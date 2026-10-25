# temp - short-lived scripts (safe to delete when the loop is filed)

Everything here was needed for one loop or one question and is **not** part of the system. Rule: when a loop's experiments are filed (`experiments/E<id>_*/README.md` has the result, the report is built), its folders here may be deleted; `training/` and `evaluation/` never import from here. Moving a script to a role folder (`training/`, `evaluation/`, `tools/`) promotes it.

| folder | what | files |
|---|---|---|
| `analysis/loop08` ... `loop16`, `misc` | diagnostics, probes, explorations, figure experiments of one loop (`p13_*diag*`, `p14_explore*`, `p14_tmp*`, `p16_graph_try`, `p16_prof_asm`, ...) | 86 |
| `launchers/loop15`, `loop16`, `loop17` | the shell wrappers that launched a batch of runs (`_run_*.sh`), smoke tests (`_smoke_*.sh`); their command lines are the record of how a run was made; loop 17: `_run_e142.sh` (E209 rerun), `_run_final17.sh` ... | 44 |
| `archive/` | historic scripts of finished loops that nothing imports: `loop01_foundation_runs`, `reports_build`, `bundles_and_run_commands`, `diagnostics_and_sweeps`, `misc` (reference; copy one back into `training/` to rerun it) | 472 |
| `_old_scripts_README.md` | the README of the former `scripts/` folder (obsolete layout) | |

Experiment-specific scripts of old experiments are not here but in `experiments/E<id>_<slug>/code/`.
Not here and not kept: one-off patches, job control, bytecode caches (staged in `_to_delete/scripts_junk/`).
