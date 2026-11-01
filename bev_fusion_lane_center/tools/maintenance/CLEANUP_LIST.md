# Cleanup list (open items, 2026-10-25)

Nothing here is urgent for disk space: the WSL disk has 715 GB free of 1007 GB (242 GB used, 155 GB of it in `~/exp_gcar_b123`). The point is order and clarity. Rule: move to `_to_delete/` first (project folders); caches in the WSL home are deleted only on the user's OK. Do not touch anything a running loop reads (loop 18 until Sunday 11:20: `runs/`, `p17/`, `p18/`, `pert/`, `feats/`, ledger).

## A. Project folders (Windows)

| item | size | what | recommendation |
|---|---|---|---|
| ~~`data/nuscenes/`~~ (done 2026-10-25) | 3.0 GB | Windows copy of nuScenes **mini** (samples, sweeps, maps). The code reads the WSL copy `~/data/nuscenes`; only the default of `extract_features.py` (loop 1) names `data/nuscenes` | move to `_to_delete/` (duplicate); keep `data/splits.json` and `data/README.md` |
| ~~`runs/` loose files~~ (done 2026-10-25) | 185 files | moved to `_to_delete/runs_2026-10-25/loose/`; the 8 logs that experiments name and the small json / npz files stay | - |
| `runs/` dead runs | ~1 GB | done 2026-10-25 for the smoke runs (`e120_smoke`, `e121_smoke`, `e18_smoke_joint` -> `_to_delete/runs_2026-10-25/smoke/`); KEPT: `e30_s4_diverged` and `e30_scene11` (read by `experiments/E030_*/code/make_e30.py`), `e34_*` / `e35_B` (named in E034) | move the kept ones only together with their experiment code |
| ~~`experiment_execution/loop17_recovery/`~~ (done) | 0.6 MB | already in `experiment_execution/history/loop17_recovery/` | - |
| `literature/sources/` | 4 MB | multi-paper notes without per-paper structure | the literature chat is reorganising `literature/`; leave to it |
| `temp/literature_backup_2026-10-24/` | small | safety copy made by the literature chat | done 2026-10-31: moved to `_to_delete/cleanup_2026-10-31_review/temp/` (its copied `tools/catalogue.py` shadowed the live one through the temp/ search path of `_py14.sh`) |

## B. Caches in the WSL home (derived data, all regenerable; hours of GPU time)

Which ones the live code reads was checked by grep of `training/`, `evaluation/`, `tools/` (a name that appears nowhere may still be built from a pattern, so "unused" below means "no sign of use").

| item | size | tied to | read by live code | recommendation |
|---|---|---|---|---|
| `~/exp_gcar_b123/feats/*_fused32.npy` | 24 GB | feature-group option (`--feats`, ablations of the dense head); the whole pipeline uses `dec32 + logit7` (`p10_data`, `p16_data`, `dense_head` default) | no | **largest single candidate**; delete after the final report (re-extraction needs the GaussianCaR pass) |
| `~/exp_gcar_b123/feats/*_dec32, *_logit7` | 29 GB | everything | yes | keep |
| `~/exp_gcar_b123/pert/` | 36 GB | failure features (loops 5, 13, 14, 17) | yes (`p17_series`, `p17_dense`, `eval_pert_val`) | keep until the loop 17 report is final; `fog` (5.7 GB) and `extr3` (4.5 GB) are used by fewer experiments |
| `~/exp_gcar_b123/p17/` | 38 GB | failure series of loop 17 / 18 | yes (loop 18 running) | keep; check again after loop 18 |
| `~/exp_gcar_b123/p8/` | 10 GB | `ds_*` datasets of loops 8-9 (templates, segment variants, ICP / none) | `p8_assign`, `p12_calib` only (loop 12 calibration) | candidate: ~9 GB of variants (`*_seg[B,C,L,P,R,S]`, `icp`, `none`) not used since loop 12; keep `train_oof.pkl`, `ds_test_tmpl.pkl` |
| `~/exp_gcar_b123/p10/` | 5.3 GB | token caches of route heads: `tok_e94_a_*` (control), `tok_e120_a_*`, `tok_e94_ar_*`, `tok_e94az_*`, `tok_p16_*` of the loop 16 ladder designs | control arm yes; ladder designs no | candidate: `tok_p16_f2*`, `f1m`, `rc1v`, `smoke` (~3 GB) and `tok_e94az_*`; keep the control and champion tokens |
| `~/exp_gcar_b123/p14/` | 6.4 GB | `chain_a/r/r2_*` (loop 14), `chainfail*` (loop 14 failure chains), `p16chain_*` of ladder designs | `p14_chain_consumer` default `chain_a`; `p16chain_f2hr_*` | candidate: `chain_r*`, `chainfail*`, `p16chain_f2l`, `f2h`, `smoke` (~4.5 GB) |
| `~/exp_gcar_b123/*.npy, *.pkl` | ~5 GB | ground truth, priors, direction / node / route maps | partly (`graph_gt_*`, `dir_val`, `route_heat_train`, `sd_*`, `prior_*`) | keep; `node_*` (0.8 GB) and `dir_train` (1.3 GB) have no sign of use: check, then candidate |
| `~/exp_gcar_b123/p18/` | 0.2 GB | loop 18 | yes | keep |
| `~/exp_crn_b123/feats` | 12 GB | loop 1 CRN features AND the exact-distance map `*_gt_dist_exact.npy`, the `gd` target of `p16_data` / `p16_loss` (seed, direction and builder losses) | yes | **keep** (corrected 2026-10-25: not a purge candidate; its distance map is the subject of D10) |
| `~/exp_crn_mini`, `~/exp_gcar_mini`, `~/data/nuscenes` | 5.6 + 2.5 + 3.1 GB | mini set: loop 1, smoke tests, prep demos | prep scripts with `--mini` | candidate; the mini archive was unpacked from a download that is gone, re-download only with the user's OK |
| `~/work/CRN`, `gaussiancar`, `bevfusion`, `gaussiancar_ckpt` | 2.4 GB | foundation code and weights | yes | keep (weights are the foundation) |
| `~/_review_*`, `~/_verify_*`, `~/extract_blob.sh`, `run_full_stage.sh`, `run_py.sh`, `setup_env.sh`, `trim_sweeps.sh` | < 1 MB | review rounds and setup scripts of loops 1-9 | no | delete (nothing in them is data) |

If everything marked "candidate" went: about 24 (fused32) + 9 (p8) + 3 (p10) + 4.5 (p14) + 12 (CRN feats) + 11 (mini) = ~64 GB. Order of value: `fused32` first, then the loop-1 CRN features and the mini copies; the loop 8 / 10 / 14 variants are small gains and cost regeneration time if a reviewer asks for an old number.

## C. training/ (audit of 2026-10-25; do after loop 18, loop 18 imports from here)

Size is small (about 1 MB of code, 82 files + 28 in `legacy/` + 38 in `data/prep/`), so the issue is not volume but naming and purpose: files are stamped with the loop that made them (`p8_` ... `p18_`), not with their function, and about 25 modules (most of `data/prep/`, `dense_head`, `graph_from_dense`, `assoc_lib`, the old modules) have German docstrings, against the English rule.

| group | files | state | recommendation |
|---|---|---|---|
| core of the champion chain | `trainer.py`, `models/p16_{builder,geom,time,tokens,ref}`, `losses/p16_loss`, `data/{p16_data,p16_common,p16_chain_data,p8_lib,assoc_lib,sd_lib}`, `pipeline/p16_{infer,runtime,pipeline,tok_resample}`, `graph_from_dense`, `models/p10_{net,expert,wave}` (control arm E94), `p17_mapnet`, `p18_regis*`, `train/{p16_train,p10_route_train,dense_head}` | used by everything, always runnable | keep as they are; renaming the `p<loop>_` prefixes is optional and only worth it if it is cheap (CLAUDE.md: keep it simple) |
| loop-specific trainings | `train/p17_{ft,gate,joint,mapft,dense_train}`, `p18_{joint,regis_train}`, `p16_ctrl`, `p10_train`, `p14_chain_consumer` | each trains one experiment; `p18_joint` is a copy of `p17_joint` | merge `p17_joint` / `p18_joint` into one script with options; the rest go to `train/loops/loopNN/` like `evaluation/loops/` |
| old models | `models/p13_{jhead,junction,phead}` (loop 13 verifiers), `p8_assign`, `p8_seg` (reference system A) | imported once each, only by old drivers | move to `models/loops/` |
| inference variants | ~~`pipeline/p16_{ensemble,hyst}`~~ (moved to `evaluation/loops/loop16/` 2026-10-31, D24), `data/p14_chain_data.py`; `line_tracer.py` is live (imported by `graph_from_dense.py`) | `p14_chain_data` is superseded by `p16_chain_data` but `data/rebuild.json` names it as the builder of the loop-14 chain data | move to `_to_delete/` unless a report names them |
| `data/prep/` | 38 files | one-time builders of the caches (loops 1-5), nothing imports them; needed only to rebuild a cache | keep as one reproducibility folder, translate the German docstrings, drop `crn_prep_mini.sh`, `crn_prep_b123.sh`, `run_prep_graph.sh`, `run_centerline_gt.sh`, `gcar_pipeline1/2.sh` after folding them into a README with the order of the steps |
| functions without a caller (scan 2026-11-01) | `data/p8_lib.py`: `perceived_vertices`, `sd_to_ego`, `row_features`, `piece_labels`, `piece_points`; `models/p8_seg.py`: `row_features_seg`, `seg_labels`; `models/p10_expert.py`: `sample_check`; `models/p10_wave.py`: `ExpertPolicy` | no tracked file (code, scripts, text, json) names them outside their definition | remove in the loop-8 / E94 clean-up with the lead chat (live modules, so not touched by the scan) |
| `legacy/` | 28 files, 390 KB | 10 modules are live dependencies (`eval_pert_val`, `p13_temporal_val`, `refiner`, `temporal_b`, `stability`, `route_head`, `route_head3`, `temporal_eval`, `lf_inputs`, `prior_tools`); the rest (`lane_former`, `exp_trace`, `report2`, `dense_report_lib`, `node_assemble`, `node_head`, `coast_eval`, `plan_eval`, `hold_eval`, `assoc_*`, `align_*`, `heat_align`, `field_tools`, `centerline_estimator`, `postproc`, `models_cnt`) serve the code of experiments E19-E72 | keep the live ones as `training/legacy/`; move them to `experiments/_lib/` (they belong to those experiments) |

## E. Purge list for Mirko (2026-10-25; decision D5) - permanent deletes, run them yourself
Agents only move things to `_to_delete/` (CLAUDE.md); these deletes are yours. Checked on 2026-10-25: nothing live reads them (grep of `training/`, `evaluation/`, `tools/`), no job runs from them. Not on the list on purpose: `~/exp_crn_b123/feats` (training target), `data/nuscenes_additions/` and the `~/data/nuscenes*` link views (owned by the bev_fusion chat), anything a running download writes.

| what | size | how |
|---|---|---|
| `bev_fusion_lane_center/_to_delete/` (contents listed in its README) | 54 MB | PowerShell in the problem folder: `Get-ChildItem _to_delete -Exclude README.md | Remove-Item -Recurse -Force`, then empty the table of `_to_delete/README.md` |
| WSL home: review and verify rounds of loops 1-9, setup scripts | ~140 MB | `wsl bash -c "cd ~ && rm -rf _review_* _verify_* extract_blob.sh run_full_stage.sh run_py.sh setup_env.sh trim_sweeps.sh"` |
| `~/exp_gcar_b123/feats/{train,val}_fused32.npy` | 25 GB | regenerable (GaussianCaR pass, `training/data/prep/gcar_tools.py`); read only by the `--feats fused32` option of the dense head: `wsl bash -c "rm ~/exp_gcar_b123/feats/train_fused32.npy ~/exp_gcar_b123/feats/val_fused32.npy"` |
| `~/exp_crn_mini`, `~/exp_gcar_mini` (mini-split copies of loop 1) | 8.3 GB | used only by prep scripts with `--mini`: `wsl bash -c "rm -rf ~/exp_crn_mini ~/exp_gcar_mini"` |
| the Claude project key of the old folder name (`.claude/projects/C--ML-Experiment` in your user folder: memory and transcripts) | small | the memory was copied to the key of the new name (`C--Auto-research`); delete the old key when no session of the old name is open any more |

## D. Known findings of `check_structure.py` that wait for the end of loop 18
- `experiment_execution/PROGRAM.md` names `indexes/LOOP17_STATE.md` (the loop-17 state file is `loop17_recovery/out__loop2__e17__LOOP17_STATE.md`) and a memory note outside the project; PROGRAM.md belongs to the running loop, fix it in the next plan.
