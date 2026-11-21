# runs - raw run outputs

Formerly `out/loop2`. What the live code reads and writes: checkpoints (`model.pt`), prediction arrays (`prob_*.npy`, `dir_*.npy`), graph pickles, per-run `metrics.json` / `status.json`, logs. 348 run folders and about 200 loose logs / flags (`*.out`, `*.log`, `*.done`) of the loops up to 12; loop 18 writes `e18/` and `e17/logs/`. Since the lane work of E271 and later every run has its own folder `e<id>_<arm>_s<seed>` (about 930 entries in 2026-11) with the files of the Training contract (`config.json`, `log.jsonl`, `status.json`, `model.pt`, `ckpt.pt` for resume); the weights are not in git, the metric cards are (`eval/`). What matters for the record is copied into `../experiments/<id>/`. Candidates for slimming after loop 18 (loose old logs, diverged and smoke runs): `../tools/maintenance/CLEANUP_LIST.md`.

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
| `e365_planb/` | `planb_s<S>.pkl`: plain B's scored VAL records (E324 s base read-out), the reference of the paired interval `paired_vs_plain_b` in every `../evaluation/jtoken_eval.py score` card (the champion's included); written only by `jtoken_eval.py check-layer1` on a PASS, an existing file kept unless `--overwrite`; copied from `temp/e365/` on 2026-11-17 (D117), sha256 unchanged |
| `eval/<name>/` | the metric cards of the scorers: `<split>.json` with all metrics, the settings and the code commit, `results.md`; in git (the text record of a number) |
| `e292_a_s<S>/`, `e273_p_s<S>/` | the lane heads: B' (E292: pyramid and node head, the front end of the token champion) and plain B (E273, the baseline) |
| `e373_cb_s<S>/`, `e372_s4_s<S>/` | the token champion (D113 - D120): the stage-1 head (E373, none weight 0.25, trained on the cached tokens) and the stage-2 weights (S4, E372: the head and the node head of B' trained together; also the token cards and the threshold fits of the decode, `token_card_*.json`, `thresh_fit_*.json`); the source of the champion is exported to `../../winners/e372_s4_champion` (no weights) |
| `e365_cache/` | the predicted-node cache of a frozen lane head, one folder per run, K and version (`be292_a_<S>_K512_<hash>_v1`, plain B `be273_p_<S>_...`); read by `../training/data/lane_ptok_data.py`; not in git |
| `e324_s_s<S>/` | the pair head of the reference system E324 s (the system of record until D113); `../training/lanegraph/` loads it together with `e273_p_s<S>` |
| `e389_m_s<S>/`, `e389_z_s<S>/`, `e389_route/`, `e389_rules/` | E389: the route-association head with the map (m) and the same head without it (z, the matched control), the route cache (`train`, `val`; not in git) and the cards of the geometric rule on the route (`rules_card_*.json`) |
| `LOOP*_INDEX.md`, `EXPERIMENT_TABLE.*` | moved to `experiment_execution/history/indexes/` and `experiments/` |

**2026-11-01 archive:** the files >= 1 MiB of 92 old run folders (19.4 GB: loop 1 - 15 runs, failure sweeps, loop-17 intermediate arrays, ladder runs off the champion lineage) were moved to `D:\Auto-research\lane_archive_2026-11-01` because C: was full; their small files (metrics, status, logs) stay here. What stayed and what moved, the manifest and the restore command: `../experiment_execution/history/lane_archive_2026-11-01/README.md`. The condensed account of the experiments that matter: `../reports/source/Condensed_Record_2026-11-01.md`.

Rules: a run directory is written once (new run = new name); a `.done` flag exists only for a run that finished. `runs/e17/logs/e142.done.FALSE` is the flag of the crashed E209 run.
