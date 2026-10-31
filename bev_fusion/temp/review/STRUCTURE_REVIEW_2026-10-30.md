# Structure and code review of the repository, 2026-10-30 (main BEV session, at Mirko's request)

Method: four read-only reviewers (training/, evaluation + evalsuite + autogt + tests, temp + directory hygiene, autoresearch +
lane project layering), their claims then checked by me where a command could check them. **V** = verified by me, **R** = reported by a
reviewer, not re-checked. Other chats were editing the repository during the review (evalsuite migration, camera features, lane branch):
files with uncommitted changes of others were not touched.

## 1. Done (commit 8612a56)
| what | evidence |
|---|---|
| `autogt/clean.py --all-caches` deleted EVERY sub-folder (an `or isdir` defeated the `index.json` guard) | V: code read; now only folders with `index.json` |
| autogt cache keyed by a signature that omits `res` and `ss`: `build --res 0.25` into a 0.5 m cache reused the old files | V: `NuScenesGT.signature()` hashes sources + flags only; the build is refused when `index.json` holds another res / ss |
| evalsuite `RunStore.create(resume=True)` accepted any manifest: a re-evaluation of a newer checkpoint returned the OLD stored counts | R + V (guard tested by hand, evalsuite selftest 38/38): a resume must have the same state / source / subject (device aside) / task / selection, `force=True` overrides |
| video `verify()` could not see the codec (OpenCV decodes MPEG-4 part 2 too) | V: mp4v file passes the old check; FOURCC `h264` / `avc1` required |
| `bev_suite.py compare` legacy fallback ignored `--tag`; `--state warmup` not selectable | V: `scene_records{tag}.pkl` is what `eval` writes (line 145) |
| `training/gt_bev.py` shim could not find `autogt` from a run snapshot (`runs/<name>/code/`) | V: copy of the shim imports `autogt` from a snapshot-like folder |
| `_wsum` / `_wcount` defined twice (losses.py, loss_extras.py), identical bodies | V; losses self-test passes |
| `plan.md` line damaged by a patch script (newline inside a file name) | V |
| `.gitignore` had no whitelist for problem 2 (`runs/` with 30 code snapshots, data caches, weights showed up as untracked) | V; `data/scenes_parts1_6.json` (the frozen val split, needed by the evaluation) is now tracked |
| 52 finished one-off scripts of `temp/` (patch / edit / dry-run scripts, a stale generated model copy) moved to `_to_delete/temp_oneoffs/` | V: `git grep` shows no reference except two names in `temp/setup/move_from_lcr.py` |
| README claims that contradicted the repository ("no ledger, no runs", "68 notes", lane3 calls evaluate_run); missing READMEs for `temp/`, `runs/`, `literature/`, `_to_delete/` | V |

## 2. Found, not changed (and why) - exact change given
| # | where | change | why not now |
|---|---|---|---|
| 1 | `training/nuscenes_dataset.py` `_gt` | write the GT cache atomically (tmp + `os.replace`, as the radar cache does); compute `gt.signature()` once in `__init__` (it re-reads the autogt sources on every sample, an edit mid-run changes the cache keys) | file has uncommitted changes of another chat (R) |
| 2 | `temp/train/run_cfg.sh` | also archive `bev_fusion/autogt` into the snapshot (`git archive ... bev_fusion/autogt`); today a snapshot uses the live autogt (the shim now at least finds it) | `lane3.sh` / `run_cfg.sh` are running; a running bash script must not be edited |
| 3 | `autoresearch/tools/` (`clean_temp.py`, `project.py`, `run.py`, `schedule.py`, `ledger.py`) | clean_temp `--delete` must refuse when the git snapshot returned no tag (R: it deletes the scratch even if the snapshot failed, and the snapshot covers only `code_dirs`); `project.py`: exit when the project folder has no `project.json` (a relative `AR_PROJECT` silently falls back to defaults); `run.py`: write `run.json` atomically (a torn read makes a live GPU slot look stale and stealable); `schedule.py`: reject duplicate item ids (schedule.csv has two); `ledger.py note` without an id writes a blank-id row | CORRECTION 2026-10-31: D21 removed the tools from the protected list, so no grant is needed; staged as decision D22 (`autoresearch/history/d22_tool_fixes.md`, six fixes re-read in the code, V), to be applied with selftest + `ar-reviewer` |
| 4 | `bev_fusion_lane_center/CLAUDE.md:73`, root `CLAUDE.md` | the protected list still names `../autoresearch/tools/` and `METHOD.md` although D21 narrowed `project.json`; root CLAUDE.md and the old `.gitignore` comment call bev_fusion "discussion only" | instruction files: your decision |
| 5 | `tests/run_gates.sh`, `tests/README.md` | only 6 of 13 gates are in the runner (`test_temporal`, `test_end_to_end`, `test_box_refine`, radar and S1 gates are not); no runner calls the evalsuite, autogt or bev selftests; the README omits 7 files | `tests/README.md` has uncommitted changes of another chat; a runner script may be running |
| 6 | `training/README.md`, `experiments/README.md` | 9 modules missing in the training README (temporal, object_head, blocks_new, camera_decoder, radar_decoder, augment, s1_camera, train_s1, dualbev_front); experiments README lacks t1_r4_soa, t2, t3, t4, t7 and still says t1 "running" | uncommitted changes of another chat |
| 7 | `evaluation/metrics.py` | mATE / mASE / mAVE skip classes without a true positive (nuScenes assigns 1.0), so a total failure raises the mean; selftest is 45 % of the file and imports `training/lift_pd` via sys.path | R, effect only on printed means |
| 8 | `training/model.py`, `train.py`, `losses.py` | dead `n_sem`, dead `w_correction` / `w_radar_slot`, `TaskWeights.values()` clamps a parameter in forward (forces a backward per keyframe), `amp="bf16"` puts geometry in bf16 (0.1-0.25 m error at 30-60 m; all configs use "off"), modality-dropout p=0.2 hidden in three constructors | R; model.py / train.py are being edited by another chat |
| 9 | ffmpeg and `/mnt/d` paths | `/mnt/c/Program Files/...ffmpeg.exe` is typed in `render/video.py` and `autogt/encode.py`, `/mnt/d/...` in `bev_selftest.py`; `~/miniconda3/envs/gcar/bin/python` in about 150 files | needs an env wrapper (`tools/env/py.sh`, as the lane project has `_py14.sh`) |

## 3. Proposed architecture (bigger changes)
**Layers** (imports only downward; a problem calls the process layer through CLI + JSON files):
```
L1 process   autoresearch/  METHOD, agents, templates, prompts, tools (stdlib only; run, ledger, schedule, decide, guard ...)
             + machine-wide state (GPU slots / lease) - today the slots live in ONE problem's run_records, a second problem gets its own 2 slots
L2 library   shared, problem-independent: evalsuite (core), video encode (ONE copy), bootstrap (ONE), nuScenes light loader + pose,
             autogt (GT), metrics (Seg/DetMeter, lane graph metrics), fit loop
L3 problems  bev_fusion_lane_center/, bev_fusion/: training, evaluation adapter, project.json, experiments
```
**Duplicates to collapse** (V = I saw the second copy): video / ffmpeg lookup in `evalsuite/render/video.py`, `autogt/encode.py` (V), lane `tools/video/make_video.py` and
`temp/loop21/make_results_video.py` (still mp4v, R); paired scene bootstrap in `evalsuite/stats.py`, `bev_legacy.py`, `train_s1.py`, lane
`sparse_lanes/bootstrap.py`, `hold_eval.py`, `temp/loop21/e257_simplify.py` (R); lane geometry / `match_1to1` / `resample` (6 copies of `resample`, R);
3D box corners and projection in `bev_views.py`, `autogt/view.py`, `autogt/video.py`, `autogt/flow.py` (R); trainer loops lane `trainer.py` vs bev `train.py` (R).
The root CLAUDE.md allows duplicating code between problems; that is why fixes drift between copies - the library layer is the cure.

**bev_fusion/training/ target** (R, mapping given by the reviewer; flat names stay importable by re-export shims, as `gt_bev.py` does):
`core/` (model, lift_pd, camera_encoder, splat, temporal_fusion, timed_head, fusion, bev_trunk, radar_bev raster, ObjectHead + decode, `spec.py` with the
layer / group constants that today make the model import the loss module) - `loss/` (losses split into targets / terms / BEVLoss, loss_extras, object-head
losses, S5c refiner) - `data/` (nuscenes_dataset, feature_cache, augment, clips) - `trainer/` (config, optim, checkpoint, ego, loop; `train.py` stays the entry) -
`variants/` (fallbacks split into ego_gru / lift_s / cmx, blocks_new split per block, radar_encoder) - `studies/` (temporal.py, camera_decoder, radar_decoder,
dualbev_front, s1_camera, train_s1, the S5b part of object_head, light_encoder + distill) - `tools/` (cache builders, gt_cache_build). V for the facts behind it:
22 modules carry the `try: from .x / except ImportError: from x` pattern although `training/` has no `__init__.py` (the relative branch is dead), `ModelConfig` has 19
bool / str switches, `train_s1.py` (training) imports `evaluation.metrics`, `model.py` imports `losses`. R: `temporal.py` duplicates `splat.chain_positions` / `splat.splat`
and is imported only by tests and temp; `CachedBEVFusionModel` copies `camera_bevs` in a local subclass (the copy caused the `camera_off` bug) and the parent still builds a
random ResNet-50 first.

**bev_fusion/evaluation + tests**: `metrics.py` split into seg / det / timing / report behind a re-export shim; one `geom` module for box corners and projection;
`bev_task` and `bev_views` must not import each other; `bev_suite.py` argparse only; `bev_legacy` frozen. Tests: classify `tests/` into gates (assert + exit code,
run by `run_gates.sh`, including the evalsuite / autogt / bev selftests), experiments (print numbers: move to `experiments/<id>/code/` with their result files) and helpers;
`test_*` files that serve as libraries (`test_warp_loss_minimum` imports `test_flow_refine` imports `test_temporal_stages`) move to a library folder; tests that import
`temp/radar/radar_recon_test.py` (and its cache `temp/radar/cache/`) get that module + cache builder promoted into `tests/` and `data/cache/`.

**Process**: bev_fusion grew a parallel process (hand-set `.done`, `run_cfg.sh` + `lane*.sh` queues, `chain_a..x.sh` sleep / pgrep chains, hand-written noise bands) next to the
method of `autoresearch/` (card, ledger, decide, `run.py --after`). Proposal: a `project.json` for bev_fusion, ledger backfill of the ~30 runs, new runs through `run.py` wrapping `run_cfg.sh`,
the queue scripts replaced by one `queue` driver (`name | script | args` lines, a shared wait-for-GPU-room helper), `decide.py` for verdicts.
**temp/**: promote what is used (`train/run_cfg.sh`, cache builders, queue drivers, report builders) to `bev_fusion/tools/`; move the code of each experiment next to its card
(`experiments/<id>/code/`); the rest stays scratch and its result files (`.txt`, `.log`, `.json` next to scripts, 51 tracked files) leave git.

## 4. Migration order that keeps running jobs alive
1. (done) guards, docs, ignore rules, archive of finished one-offs. 2. Tools fixes of 2.3 with a decision (D-number) and the autoresearch selftest. 3. After `lane3.sh` ends:
`run_cfg.sh` archives autogt, `tools/` created, `temp/` launchers moved, doc paths rewritten. 4. Library layer by copy (video, bootstrap, nuScenes light loader), old modules become shims.
5. `training/` split studies / variants first (nothing in the product imports them), then loss / data / trainer behind shims, `try/except ImportError` removed by a `_paths` helper.
6. Evaluation split of `metrics.py`, tests reclassified, runner extended. 7. bev_fusion adopts `project.json` + `run.py`; machine-wide GPU slots (tell the lane chat, D19 is open).
