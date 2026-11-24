# Scratch files that the cards name, after D140 (2026-11-24)

D140 emptied `bev_fusion/temp/` and `bev_fusion/_to_delete/` (what went where: `temp/README.md`). The owner's rule that came after the deletion (relayed by the lead, 2026-11-24): a scratch script that a card names as how a number was made stays (small, copied into the card's folder) or the card gets the exact command; the local `bf/*` and `lcr/*` tags stay. This is the check of the cards and result folders of this problem (`experiments/`, `results/experiments/`) against the tags, made with `git cat-file -e <tag>:bev_fusion/<path>` over the 114 local `bf/*` tags.

**42 distinct scratch names** are named by the cards (41 files and one pattern, `temp/overview/*_result.txt`). How to read one that is in a tag: `git show <tag>:bev_fusion/<path>`. The scratch files that the cards named and that D140 moved (not deleted) are cited at their new place since the same commit: the t26 card (`records/t26_g0/`, `runs/t26_prep/`, `runs/t16_soa_ref/`) and `evaluation/` (`runs/junction_ref/`, `records/intersection_iou_result.txt`).

| group | files | where it is now |
|---|---|---|
| scripts and queue files in a tag (23) | `temp/lane_tests/chain4.sh` ... `chain8.sh`, `check_step1.py`, `fn_fp.py`; `temp/overview/lane_readout.py`; `temp/study/queue.txt`, `queue3.txt`; `temp/t16/prep_more.sh`, `rule_t17.py`, `rule_t18.py`; `temp/t22/chain/compare_gate.py`, `chain/make_standin_run.py`, `recovery_B2.py`, `time_student.py`; `temp/train/run_ar.sh` | tag `bf/chain.t20` (the scratch of 2026-11-10, taken at that run's start) |
| | `temp/pitch/ladder.py`, `temp/pitch/maps_shares.py`, `temp/soa_gap/map_ref_dev.py` | tag `bf/diag.overfit8_after` |
| | `temp/t4/compare_front.py` | tag `bf/pitch.maps_lfl` |
| | `temp/timing/time_image_sides.py` | tag `bf/diag.t21_rec_speed` |
| outputs that a card cites, copied into the tree (5) | `temp/t16/compare_t17.txt`, `compare_t18.txt`, `compare_t18_vs_base.txt` | `results/build/inputs/` |
| | `temp/t5/results.md`, `results_v2.md` | `results/experiments/t5_lane_probe/` |
| outputs of one-off runs, gone (11) | `temp/box_vel/out/alive_table.txt`, `doppler_at_boxes_val.txt`, `p1_t1_r4_soa_val.txt`, `p1_t8_ctl_s0_val.txt`, `probe_fit_t8_ctl_s0.txt` (the probe outputs of `results/experiments/t11_vel_cause/`); `temp/lane_tests/t23_s1_readout.txt`; `temp/overview/l4_readout.txt`; `temp/pitch/maps_lfl.md`; `temp/t21/drift_scene-0003_pair_config.json`, `drift_scene-0003_recurrent_config.json`, `timing_full_cuda.json` | not kept: they are read-outs; the numbers they gave stand in the cards and result files that cite them (`experiments/t23_lane_line_loss`, `t24_lane_pull`, `t21_one_frame_temporal/stream_timing_full_cuda.txt`, `results/experiments/t11_vel_cause`) |
| an input list, gone (1) | `temp/t5/frames.json` | the frames of the lane probe t5 (a closed study); not kept |
| planned, never written (1) | `temp/t22/chain_twin_R.sh` | the t22 card says it was "not written yet"; it never existed |

Scratch scripts named by the ledger rows and `records/runs/*/run.json` as commands (the composed configs `temp/compose/*.json`, `temp/t26/job_*.sh`, `temp/*/eval_arm.sh` ...) are history, not cards: the tags `bf/<run id>` hold what was in the scratch folder at that run's start (up to 1000 kB and 2000 files: a later run's ledger note says "scratch script(s) not in the tag" when the limit cut it), the composed config of every run is `runs/<run>/config.json`, and `runs/t26_prep/job_*.sh` keeps the t26 job commands.
