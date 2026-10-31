# Repository review 2026-10-31, pass 2: architecture, code, cleanup

Second pass of the same evening, after the first review (`repo_review_2026-10-31.md`) and its fixes (D23, D24, 945bacf). Four fresh read-only reviewers looked at one area each: the bev_fusion model chain; its training loop, data and losses; its evaluation and ground truth (`evaluation/`, `autogt/`); the autoresearch tools with the bev_fusion launch path. Their findings were re-checked in the code. A fix was made when it changes no training trajectory and no stored number. Fixes that would change future numbers are listed as proposals or decisions.

**The `ar-reviewer` pass (mode code) over the fixes.** All seven claims of the fixes held. Its findings were fixed in the same commit:
- `--force-resume` could not work for a recorded GT;
- a cut evaluation was not made up on a resume;
- the checkpoint fallback was too broad, and a restart at 0 when every checkpoint was damaged;
- superseding the log needed a loaded checkpoint;
- a valid legacy pickle was renamed `.stale`;
- the log rewrite failed on /mnt/c;
- `write_shard` left temporary files;
- the prior-baseline store was not checked;
- missing tests.

**Two consequences on purpose:**
1. The validation loader no longer draws from the global random stream. A run of the new code is therefore not bitwise comparable with an older one after its first evaluation (same seed, other flip masks). The seed noise band covers this.
2. The single-sensor arms now train without modality dropout, so their stored results cannot be reproduced with the new code.

Running jobs were not touched: the `t10.dual_s0` training, the autogt trainval report, `queue_fused.sh`, the `move_ckpts.sh` loop. The t7 re-evaluation (`bev_fusion/temp/t7/finish2.sh`) started on the GPU at the same moment as t10 (both waited for a free GPU; 11.5 GB in use). It was stopped and re-queued behind t10 as `finish3.sh`.

Line numbers are those of 2026-10-31, 21:00.

## 1. Architecture

**The research system has three layers.**
1. **`autoresearch/`, problem-independent:**
   - method, agent roles, templates and prompts;
   - tools: the launcher `run.py` (snapshot commit and tag per run, GPU slots, retries), ledger, schedule, `decide.py`, card and experiment checks, the guard hook, `clean_temp.py`, `check_structure.py`;
   - `evalsuite/`: the source / subject / task contract, run stores, tables, the one paired scene bootstrap, the comparison core;
   - `trainsuite/`: status, curves, kill rules, the CONTROL / STOPPED files, exit code 75.
2. **Two problems:** `bev_fusion_lane_center/` (tag prefix `lcr`) and `bev_fusion/` (`bf`). Each has its `project.json`, its code, its evaluation (the truth kit), its records and experiments, and `runs/` (not in git).
3. **Data underneath:**
   - nuScenes raw on D: through the WSL views;
   - derived caches on C:, per problem;
   - checkpoints in `runs/` (some moved to D:);
   - git tags per run.

**The bev_fusion perception chain:**
1. **Camera features:** the frozen DualBEV image heads (cached, stride 16), or cached BEVFusion features (stride 8), or both ("dual", t10).
2. **Lift PD:** a depth-weighted pull into a 0.5 m grid, 0-60 m x ±26 m.
3. **Camera temporal fusion:** a window of frames plus the world memory. The t7 lane branch (stride 4, zero start) is added after this step.
4. **Fusion:** concatenation with the radar raster, with modality dropout.
5. **BEV trunk:** a ResNet pyramid with an FPN.
6. **Time mover:** the motion head plus a splat to t + dt.
7. **Timed heads:** maps, objects (CenterPoint) and motion, decoded at dt -0.5 / 0 / +0.5 s.

**The bev_fusion code and data flow:**
1. **`autogt/`** generates the GT. Its signature is the sha1 of the five generator files plus the options.
2. **Two GT caches:** per sample in `data/cache/gt`, per keyframe in `data/autogt/<sig>`.
3. **The dataset** reads the GT caches, the camera feature caches and the radar cache.
4. **`train.py`** runs inside a `git archive` snapshot of `training/` (`run.py` -> `temp/train/run_ar.sh`).
5. **The run folder** holds the checkpoints, `log.jsonl` and `gt.json`.
6. **`evaluation/bev_suite.py`** loads the model from the run's own snapshot. The GT comes from that snapshot's dataset code.
7. **The stored evaluation:** store, `metrics.json`, `card.json`.
8. **`compare` and `decide.py`:** the paired scene bootstrap and the verdict.

**Weak joints (section 2):**
- the GT is built inside loader workers, into a cache that holds 13 versions;
- training is not pinned on the GT;
- old snapshots evaluate with their own GT generator;
- camera feature caches have no identity;
- `run.py` and trainsuite are two run-state machines that disagree.

## 2. Findings and what happened to them

### The box velocity output (found from the stored detections; the most important finding)
| finding | evidence | status |
|---|---|---|
| **The object head's velocity is about 0 in every main-chain run.** | **Max predicted speed** of the vehicle detections in the stored evaluations (`scene_records*.pkl`, 56-64 k detections per dt): 0.040 m/s for t2_ref_s0, 0.031 for t3_vel10 (`w_obj_vel` 10), 0.012 for t7_rgb_s0, 0.024-0.051 for t8_ctl_s0. 22 % of the GT vehicles move faster than 1 m/s.<br>**The evaluation path carries velocities:** t4_dualbev_front and t3_img_long reach 13-14 m/s through the same code. So the chain's box-velocity branch does not learn.<br>**What this does to the reported AVE:** it is the error of always predicting 0. The gap between dt 0 (2.35-2.6 m/s in 34 evaluations) and dt ±0.5 (1.2-1.7) is a selection effect: without a velocity the moving vehicles are not matched within 2 m at ±0.5 s, so the matched set at ±0.5 is mostly parked cars (t8_ctl_s0: mean GT speed of the matches 2.49 m/s at dt 0, 0.43 at ±0.5). | **Open, model.** Next step: log the `objhead.vel` term per step and check the gradient of the vel branch. As a test, read the box velocity from the motion head (`v_hat` at the box centre). Until then report AVE next to `ave0`; the suite writes `ave0` since 8eb5587. |

### Model chain (`bev_fusion/training/`)
| # | where | finding | severity | status |
|---|---|---|---|---|
| A1 | `fusion.py:35-42`, `model.py:203-206` | Modality dropout (p = 0.2) is hard-wired and stays on in the single-sensor arms. About 10 % of the samples of t2_cam_off, t2_radar_off, t3_radar_only, t3_cam_adapter and t3_cam_zsum had no input at all, so those arms understate the remaining sensor. | medium | **Fixed:** `ModelConfig.p_drop`, forced to 0 when a sensor is off. The finished arms are as they were trained. |
| A2 | `evaluation/bev_adapter.py:176-180`, `bev_probe.py:115-128` | `zero_cam` / `shuffle_cam` left `prob2` / `ctx2` (t10) and `lane_img` (t7) untouched. | medium | **Fixed**, with a self-test. |
| A3 | `feature_cache.py:42-65` | The camera feature caches have no identity. Swapped caches (DualBEV features read as BEVFusion ones) train on the wrong camera geometry without an error. | medium | **Partly fixed:** a feature-size guard (DualBEV 16 x 44, BEVFusion 32 x 88). **Proposal 5:** `meta.json` per cache. |
| A4 | `model.py:330-337` | The lane branch reads the image even with `camera_off`. | low | **Fixed:** the combination is refused. |
| A5 | `temporal_fusion.py:149,198` | The memory read is untrained in `seq_len` 1 runs but still used by warm evaluations. | low | **Open:** the evaluation should refuse carry mode for `seq_len` 1. |
| A6 | `blocks_new.py:55` | RadarPillars left its point net in eval mode after a single detection. | low | **Fixed**, with a self-test. |
| A7 | `feature_cache.py:115`, `cache_bevfusion_features.py:124` | Shard writes are not atomic: a cut extension pairs old index rows with the new array. | low | **Fixed:** `write_shard` (temporary names, index last) and a row-count check. |
| A8 | `train.py:535`, `model.py:170-187` | Seed twins are not paired across arms: the creation order of optional modules shifts the random stream. | low | **Proposal 7.** |
| A9 | `nuscenes_dataset.py:85` | The DualBEV input is resized BILINEAR. BEVDet used Pillow's default, BICUBIC since Pillow 7 (unverified). | ? | **Proposal 10:** a 100-frame probe. |

### Training loop, data, losses
| # | where | finding | severity | status |
|---|---|---|---|---|
| B1 | `train.py:201-206,622-629` | **The validation loader:** same worker count as training, workers kept alive, and GT built inside them. t7_rgb_s1 died of ENOMEM in validation worker 4 at step 1000.<br>**The order:** the evaluation ran before the save of the same step, so 500 steps were lost. | medium | **Fixed:** at most 2 workers, not kept alive, with their own generator; the save comes before the evaluation, and a resume makes up an evaluation that a crash cut. |
| B2 | `temp/train/run_ar.sh:22`, `autoresearch/tools/run.py:638,660` | `run_ar.sh` prints `tail -n 40` of a log that holds every attempt. A stop (exit 75) after a CUDA crash is then retried and trains on. | medium | **Open:** `run_ar.sh` after t10 ends; the `run.py` side belongs to the owner of the tools work in progress. |
| B3 | `train.py:555-563` | A resume starts a new data permutation, and the sampler seed depends on the model's size. | medium-low | **Proposal 7** (it changes the random stream). |
| B4 | `train.py:557-560` | A resume does not check the configuration: a changed `steps` restarts the cosine schedule (factor 0.001 -> 0.54). | medium-low | **Fixed:**<br>- refused unless `BF_RESUME_ANY=1`;<br>- a damaged checkpoint (zip, unpickling or EOF error only) is kept as `.damaged` and the one before is used;<br>- if no checkpoint is readable the run refuses to restart at 0.<br>`run_cfg.sh` still overwrites `train.log` (legacy launcher). |
| B5 | `losses.py:375` | The documented "no-time control" `dt_weights: [[0, 1]]` weighs dt ±0.5 like dt 0. | low-medium | **Fixed:** an unlisted \|dt\| is refused; the configs README is corrected. |
| B6 | `train.py` | A resumed run had duplicate log rows. | low | **Fixed:** the rows above the resumed step go to `log_superseded.jsonl`, also after a crash before the first save. If a Windows reader holds the log open, a warning is printed and the rows are appended. |
| B7 | `losses.py:133-175,326-361` | Boxes of unknown velocity are trained as 0 m/s (0.4 % of the boxes). | low | **Proposal** (it changes numbers). |
| B8 | `feature_cache.py` `_radar` | The radar cache read had no F1 protection. | low | **Fixed:** an unreadable entry is rebuilt; no temporary files are left behind. |
| B9 | `train.py:601-608,555` | A skipped non-finite step skipped the save; an unreadable newest checkpoint had no fallback. | low | **Fixed.** |
| - | `augment.py:113-120,174` | The self-test failure "radar mirror" was two test defects:<br>- `np.allclose` without `equal_nan` on `vel_sigma` = NaN;<br>- a bit-exact `torch.equal` of two scatter-summed rasters, which fails intermittently on several CPU threads.<br>`flip_meta` itself is exact. | low | **Fixed.** |

### Evaluation and ground truth
| # | where | finding | severity | status |
|---|---|---|---|---|
| C1 | `bev_suite.py:214-226` | Re-running `eval` resumes a store made without a signature, or on another GT, and labels the old counts with the current GT signature. | medium-high | **Fixed**, also for the prior baseline's store:<br>- a store of another recorded GT is always refused (the evalsuite store refuses it too);<br>- a store made before the signature was recorded is resumed only with `--force-resume`, labelled `mixed:unrecorded+<current>`. |
| C2 | `bev_suite.py:183-190` | After a resume, the detection part of the `scene_records` pickle covers only a subset of the scenes. | medium | **Fixed on this side:** the pickle is written only when this process saw every stored frame. An existing pickle that fits the store (same frames, same GT) is kept; otherwise it is renamed `.stale`. The evalsuite side (`on_frame` for resumed frames) is in another chat's uncommitted work. |
| C3 | = A2 | | | |
| C4 | `bev_suite.py:215-231` | Results are overwritten by an evaluation with another n. The new `board.py` compares without checking n. | low-medium | **Fixed:** copies in `eval/by_n/` (`metrics<tag>_n<n>.json`, `card...`, `table...`); a subfolder, so globs such as `metrics_ctl_*.json` do not count them twice. `board.py` (another chat's) should check `info/n_samples`. |
| C5 | `nuscenes_gt.py:309`, `bev_views.py:39-43` | The FOV mask uses the full image, but the BEVFusion crop drops 3.4 % of the GT divider cells, which biases against t8_bevfeat. | low-medium | **Decision** (the metric changes). |
| C6 | `metrics.py:202-206` | The detection wedge reaches beyond the grid (\|y\| <= 26 m for x > 41 m), so 1.8 % of the GT boxes can never be matched. | low-medium | **Decision.** |
| C7 | `metrics.py:56,192` | The moving threshold is 0.5 m/s, autogt's is 1 m/s. `ave_skill_static` is unbounded. | low | **Decision.** |
| C8 | `autogt/cli.py:98` | `build --workers` uses spawn: the workers re-import the generator, so an edit can be stored under the old key. | low | **Open.** |
| C9 | `autogt/clean.py:51-60` | `index.json` is overwritten by the last build and there is no lock, so `clean` can move a folder that is still being written. | low | **Open.** |
| C10 | `bev_adapter.setup_paths` | **The evaluation GT comes from the run's own snapshot.**<br>Runs older than the autogt shim (t2_ref, 2026-10-30) are scored on their own generator (962e788566), newer runs on the live autogt.<br>Object numbers of old against new runs are therefore not comparable (vehicle layer differs in 94 of 120 frames). The maps are identical across 962e788566, 7fe1149566 and 15efbde2d1. | medium (design) | **Proposal 2.** The t7 decision uses the same-GT control t8_ctl. |

### autoresearch tools and the launch path
Several of these files have uncommitted work of another chat (`decide.py`, `run.py`, `project.py`, `card.py`, `experiment.py`, `evalsuite/{quantitative,store,selftest}.py`). They were not edited here; the findings are for that owner.

| # | where | finding | severity |
|---|---|---|---|
| D1 | `tools/decide.py:185` | The headline clause is taken without looking at its `vs`, so the band test against the control is skipped. Example: KEEP with +0.005 "beyond the band" (confirmed by a run). | medium |
| D2 | `tools/decide.py:188` | The T1 fix is incomplete for subset-only cards: one seed per subset still gives KEEP, and `selftest.py:241-244` asserts that KEEP. | medium |
| D3 | = B2 | | medium |
| D4 | `tools/project.py:31-44`, `ledger.py:146-151` | Without `AR_PROJECT`, a tool run from `bev_fusion/` writes into the lane problem: ledger rows, schedule items, a run record, an `lcr/` tag. | medium |
| D5 | `run_ar.sh:12,18`, `run.py:558-561` | **Pinning holes:**<br>- an explicit commit argument is ignored;<br>- the snapshot is taken when the GPU slot frees, not at launch (t10 waited from 19:38 to 21:06);<br>- a config outside `training/` is read live;<br>- a failed snapshot trains HEAD. | medium |
| D6 | `tools/decide.py:53-56` | The typed interface never cuts on one seed or on disagreeing seeds. | low-medium |
| D7 | `tools/run.py:307-325` | An `--after` id without a record waits the full 720 minutes. | low-medium |
| D8 | `evalsuite/stats.py:116-130` | Draws whose statistic leaves a key out are dropped and not counted, so `n_valid` hides them. | low |
| D9 | `run_ar.sh:23` | `.done` is not cleared before a re-run in the same folder. | low |
| D10 | `tools/guard_hook.py:135` | An MCP `move_file` is checked only by its destination, so moving a protected file away is not refused. Protected file: a decision. | low |

The fixes the first review reported as done are in HEAD (124 exit code, `wait_after`, S2, S3, S5, S6, S8), except T1 (D2) and the ghost rows (D4).

## 3. Cleanup
- **a7820ec:** 27 finished t1-era one-offs moved from `bev_fusion/temp/` to `bev_fusion/_to_delete/temp_oneoffs/` (rows in its README):
  - `audit/` launchers and configs: 9;
  - early `checks/`: 4;
  - `radar/` studies: 3;
  - 2026-10-28/29 `study/` helpers: 6;
  - t1 `train/` smoke configs: 5.

  Criteria: tracked, unchanged in the working tree, last committed 2026-10-25 .. 10-29, the file name occurs in no other tracked file, and no running process names it.
- **Pitfall:** git refuses the short exclusion form `':!_to_delete'` with "Unimplemented pathspec magic". A reference scan must use `':(exclude)...'`.
- **Updated:** `bev_fusion/README.md` (bev_fusion is under `run.py` since D23) and `bev_fusion/data/README.md` (the GT cache key is the autogt signature, not `gt_bev.py`).
- **Not moved:**
  - `temp/train/run_ar.sh` and `run_cfg.sh`: live launch code (proposal 4);
  - `temp/study/lane*.sh`: the queue history of the t2-t8 runs;
  - `temp/t7/` and `temp/t10/`: running work.
- **The lane problem:** its structure check reported 41 dead paths, from the removal of the loop 19-21 scratch. The lane chat fixed them in e3d8aa9 (check_structure 41 -> 0).

## 4. Proposals, ranked
1. **The box velocity:** find why the branch does not learn (section 2). Until then, every AVE in a table is "the zero predictor" and should be printed next to `ave0`.
2. **One measuring stick for the GT:**
   - the evaluation builds the GT from the live `autogt/` for every run, not from the run's snapshot, and records the signature;
   - training refuses to start unless the GT store covers its split under the current signature (prefill with `gt_cache_build.py`; no GT building in loader workers, which is what killed t7_rgb_s1);
   - one GT store instead of two (`data/cache/gt` has 13 versions and about 76 k files).
3. **One run-state machine:**
   - the trainsuite contract inside `run.py`: exit 75 means stopped and is never retried; one done flag; one staleness rule;
   - `run_ar.sh` prints only the current attempt;
   - the snapshot is taken at launch, not at slot time.
4. **Launch code out of `temp/`:** `run_ar.sh` and `run_cfg.sh` move to `bev_fusion/tools/`, and `project.json` `fingerprint_extra` changes with them. Today `clean_temp.py train` would move the live launcher away.
5. **Caches with identity:** a `meta.json` per camera feature cache and the lane image cache (encoder, checkpoint hash, ida, resize filter, feature size), checked against the config, as the GT and radar caches already do through their keys.
6. **Sensor switches in one place:** one sensor mask at the fusion input and one list of camera keys for the controls. A1, A2 and A4 were one class of bug.
7. **The data order belongs to the sampler:** a generator seeded by (seed, epoch), with its position in the checkpoint. That gives exact resumes and paired seed twins. Introduce it at the start of an experiment round, because it changes the random stream.
8. **Speed (ms per frame):**
   - the time mover runs once per head per dt: 6 calls for 3 dts, two of them identity splats at dt 0, plus a third motion pass for `v_hat`;
   - `cam_temporal: none` still lifts and reduces all 3 frames.
9. **Metric decisions for Mirko:**
   - C5: score the FOV of the run's own image transform, or the intersection of the compared pipelines;
   - C6: limit the detection wedge to the grid;
   - C7: one moving threshold, 1 m/s as in autogt;
   - `lane_divider:zc` F1 is degenerate (r@0.5 0.001-0.01): take it out of the headline.
10. **Cheap probes:** the resize filter (A9); refusing carry mode for `seq_len` 1 (A5); the radar-off velocity channel `aux_rad radar_stationary`, which is never supervised.

## 5. Open, by owner
- **bev_fusion lead (after t10 ends):**
  - `run_ar.sh`: print only the current attempt (B2), refuse a commit that differs from `AR_COMMIT` and a config outside the snapshot (D5), move `.done` aside at the start (D9);
  - the box-velocity investigation;
  - proposals 2-8.
- **Owner of the uncommitted tools work:** D1, D2 (and the selftest that asserts the wrong KEEP), D4, D6, D7, the `experiment.py` `KeyError: 'no card for Q'` in the work in progress, and its `code_drift` step: it runs while nothing updates the heartbeat, and it flags drift on every bev_fusion run although `run_ar.sh` trains from the snapshot.
- **evalsuite owner:** D8; C2 on the evalsuite side (in progress).
- **Mirko:** C5-C7 and the `zc` headline; D10 (protected `guard_hook.py`); proposal 2 (the GT measuring stick) as a decision, since it changes how old runs are scored.
