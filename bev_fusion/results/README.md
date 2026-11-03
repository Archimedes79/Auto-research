# results/ - the result package of each run

One folder per run id (`results/<run id>/`), written by `autoresearch/tools/package.py` and `autoresearch/tools/provenance.py` (D40). It answers: how was this run made, what came out, and where is the deeper material. Next to the run folders: `experiments/` (the finished experiments, below) and report pages (`t16/`, `BEV_result_report/`).

| file | what | in git |
|---|---|---|
| `package.json` | the package: code tag and commit, GT version and signature, model bundle hash, checkpoint and step, seed, run status, tier (best three or normal), the top-level numbers of `eval/metrics.json` and `eval/latency.json`, and the list of DERIVED files (pictures, videos, pickles, per-scene folders; name and size) | yes |
| `provenance.json` | the full provenance block (`provenance.py manifest <run id>`): GT it trained on, final checkpoint hash, config hash, environment of the run record, the model bundle (the network and loss files with their hash) | yes |
| `split.json` | the scene lists the run trained and validated on, the split name, the GT version, the evaluation frames when the caller has them (`training/splits.py split-json`); `package.py` runs it through the hook `package.split_record` of `project.json` (into a temp folder first) and refuses to finish the package when the file is not written; a run without `config.json` has nulls and a `note` | yes |
| `bundle/` | a copy of the network and loss files and the config of the run, so the model can be rebuilt from this folder and the checkpoint; the tag holds the same code | no |

The heavy files stay in `runs/<run folder>/` (checkpoints, `eval/`, `report/`). Outside the best three (`records/keep3.json`) the derived files are deleted by `package.py clean --apply` when the run is final, its tag and checkpoint exist and the files are older than two days; the package then says `deleted` with the list and how to rebuild them. Old GT stores go the same way (`package.py gt`). Checkpoints are never removed by it (best three complete, the others slimmed or archived). Every deletion is logged in `records/CLEAN_LOG.md`.

Reproduce a run: `python autoresearch/tools/provenance.py repro <run id>`.

## experiments/ - the finished experiments (D48, 2026-11-03)

`results/experiments/<folder>/` holds the card, summary and result files (tables, images, small json) of an experiment that is finished; the open cards stay in `../experiments/`. The name of the folder is the old one, so the move is `experiments/<folder>/` -> `results/experiments/<folder>/`. The tools (`experiment.py`, `decide.py`, `package.py`, `metrics.py`) look in both folders (`project.json` "experiment_dirs"). The relative links inside the moved cards were fixed in the same step (one level deeper); the `experiment` field of an older `package.json` and old ledger notes, run records, configs and dated records still say the old path: follow this map.

| old path | new path |
|---|---|
| `experiments/t1_full_chain/` | `results/experiments/t1_full_chain/` |
| `experiments/t1_r4_soa/` | `results/experiments/t1_r4_soa/` |
| `experiments/t2_block_swaps/` | `results/experiments/t2_block_swaps/` |
| `experiments/t3_improve_loop/` | `results/experiments/t3_improve_loop/` |
| `experiments/t4_soa_front/` | `results/experiments/t4_soa_front/` |
| `experiments/t5_lane_probe/` | `results/experiments/t5_lane_probe/` |
| `experiments/t6_soa_gap/` | `results/experiments/t6_soa_gap/` |
| `experiments/t8_bevfeat/` | `results/experiments/t8_bevfeat/` |
| `experiments/t11_vel_cause/` | `results/experiments/t11_vel_cause/` |
| `experiments/t11_vel_hurdle/` | `results/experiments/t11_vel_hurdle/` |
| `experiments/t12_motion_block/` | `results/experiments/t12_motion_block/` |
| `experiments/t17_more_data/` | `results/experiments/t17_more_data/` |
| `experiments/t18_motion_lr/` | `results/experiments/t18_motion_lr/` |
