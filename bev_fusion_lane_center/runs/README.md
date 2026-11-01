# runs - raw run outputs

Formerly `out/loop2`. What the live code reads and writes: checkpoints (`model.pt`), prediction arrays (`prob_*.npy`, `dir_*.npy`), graph pickles, per-run `metrics.json` / `status.json`, logs. 348 run folders and about 200 loose logs / flags (`*.out`, `*.log`, `*.done`) of the loops up to 12; loop 18 writes `e18/` and `e17/logs/`. What matters for the record is copied into `../experiments/<id>/`. Candidates for slimming after loop 18 (loose old logs, diverged and smoke runs): `../tools/maintenance/CLEANUP_LIST.md`.

| entry | what |
|---|---|
| `e<id>_<tag>/` | one run of experiment E<id> (loops 2-15); `e94_a` control trace network, `e37_final` dense head, `e120_*` refinement, `e117`-`e121` loop 14-15 |
| `e123/e123_<design>_r<rung>_<step>/` | ladder runs of loop 16 (model, `log.txt`, `status.json`); `e17_map_*` map-conditioned builders of loop 17 |
| `prototype/<name>/` | output of `../tools/prototype/lane_prototype.py`: `scenes/<scene>.json` (planner format), `summary.json` / `.md`, optional video (rewritten by each call) |
| `e123_ladder/` | loop 16 workspace: `cards/` (one json per finished run), `diag/`, `gate/`, `state.json`, `queue.json`, rung summaries (the human-readable `TRIAGE_STATE.md` moved to `../experiment_execution/history/indexes/`) |
| `e18/` | loop 18 workspace (`p18_*` outputs, fork pruning, smoke runs) |
| `e17/` | loop 17 workspace: `e130_champion` ... `e144_*` (internal names; catalogue ids E193-E212, see `../experiments/`), `results_master.json`, `results_extra.json`, `logs/`, `latency/`, `consumer/`, `tables/` |
| `p16_*`, `p16chain_*`, `timing`, `diag_*` | loop 16b: route heads (`p16_<cfg>_route_s<seed>`), chain data and consumers, timing |
| `_loop1_crn/` | loop-1 CRN results (formerly `out/crn`) |
| `LOOP*_INDEX.md`, `EXPERIMENT_TABLE.*` | moved to `experiment_execution/history/indexes/` and `experiments/` |

**2026-11-01 archive:** the files >= 1 MiB of 92 old run folders (19.4 GB: loop 1 - 15 runs, failure sweeps, loop-17 intermediate arrays, ladder runs off the champion lineage) were moved to `D:\Auto-research\lane_archive_2026-11-01` because C: was full; their small files (metrics, status, logs) stay here. What stayed and what moved, the manifest and the restore command: `../experiment_execution/history/lane_archive_2026-11-01/README.md`. The condensed account of the experiments that matter: `../reports/source/Condensed_Record_2026-11-01.md`.

Rules: a run directory is written once (new run = new name); a `.done` flag exists only for a run that finished. `runs/e17/logs/e142.done.FALSE` is the flag of the crashed E209 run.
