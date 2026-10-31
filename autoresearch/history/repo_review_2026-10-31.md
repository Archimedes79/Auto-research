# Repository review 2026-10-31

The whole repository was reviewed read-only: `autoresearch/` file by file, then `bev_fusion_lane_center/`, `bev_fusion/` and `datasets/NuScenes/`. The review answers Mirko's six questions. Three sub-reviews (evalsuite + trainsuite, the lane project, bev_fusion + datasets) ran in parallel, and their high-severity claims were re-checked in the code. A claim marked "confirmed" was reproduced with a throw-away script or traced line by line; "by reading" means it was not executed. Line numbers are those of 2026-10-31.

The cleanup moves and doc fixes of this review were committed at the same time (section 4 and section 6). No tool, metric or protected file was changed.

## 1. Is the code correct?

### autoresearch/tools (standard-library core)
The core is in good shape. `selftest.py` passes, `check_structure.py` reports 0 problems after one doc fix (below), and every D22 fix is applied: `d22_tool_fixes.md` still said "staged, NOT applied", and its header now says applied in ee19026. One real bug:

| # | where | failure case | severity |
|---|---|---|---|
| T1 | `tools/decide.py:151-163` | When the headline metric has only a guard clause (`max_loss`) and no other `min_effect` clause raises a flag, an effect beyond the band on **one seed**, or on two seeds that **disagree**, gives `KEEP`. Repro: one seed each, Q 0.70 vs 0.75, rule `[{Q, max_loss 0.005}]` gives KEEP. Seeds 0.80 / 0.69 vs 0.70 / 0.70 also give KEEP, while the typed interface gives INCONCLUSIVE for the same numbers. The docstring promises INCONCLUSIVE. Seed and interval flags are only computed for `min_effect` clauses. E239, E259 and E260 have such rules; none has a `decision.json` yet, so no recorded verdict is affected. Fix: compute the seed / interval flags for the headline in `decide_experiment` itself, whatever the clause kind. | medium (confirmed) |

Minor:
- `ledger.py note <typo id>` silently creates a `planned` ghost row. `decide.py` guards against this; the CLI does not.
- `run.py` treats any exit code 124 as its own timeout. A command that itself exits 124 (e.g. `timeout`) is recorded as `timeout`.
- `wait_after` treats a dependency with status `ok` but no done flag (the flag write failed) as not finished and waits until `--wait-after` expires.

`project.json` "loops" ends with `"19": [234, null]`, so `metrics.py` counts loops 20 and 21 under loop 19. This is decision D17 (open), not a code bug, but the loop table in `LEDGER.md` is wrong until D17 is answered.

### autoresearch/evalsuite and trainsuite
Both selftests pass (evalsuite 44/44, trainsuite 15/15, in `gcar`). `trainsuite/` was committed by another chat during the review (5d3bdae); the evalsuite cost parts are still uncommitted work of that chat. The bugs:

| # | where | failure case | severity |
|---|---|---|---|
| S1 | `evalsuite/lanes/task.py:195-210` (committed) | `lane_probe` caches its result in `d.pred.meta["probe"]`, and `LaneGraph.of()` returns the object it is given. A Subject that returns one constant `LaneGraph` (an empty null baseline, a prior) gets frame 0's probe tags and error budget for every frame. Exactly the null baselines CLAUDE.md requires are corrupted in `probe budget`, `situations` and `scan`. | high (confirmed) |
| S2 | `trainsuite/cli.py:125,138` (committed in 5d3bdae during the review) | `stopped_by` is keyed by rule text only. `watch a b --kill "..." --enforce` with both runs over the limit writes `CONTROL` into `a` only; `b` trains on. Key it on (run, rule). | high; **fixed in 90d2314** (keyed by run and rule) |
| S3 | `trainsuite/kill.py:12,42` (5d3bdae) | The rule regex ends in `\s*$`; only whole-line comments are skipped. A rule copied from the README with an inline `# ...` comment raises `ValueError`, and `status` / `watch` die. | high; **fixed in 90d2314** (comments stripped, bad lines reported) |
| S4 | `trainsuite/runs.py:107-117` (90d2314) | A `Traceback` in the tail marks a run crashed unless a later line starts with `"step "` (the bev_fusion log format). A trainer that logs a handled exception and goes on with another line format is reported failed, and `watch` stops watching it. | medium (still open in 90d2314) |
| S5 | `evalsuite/store.py:114-121,174` (committed) | A metric that returns `{}` for a whole chunk gets no counts file, and `table` / `compare` / `outliers` then raise `FileNotFoundError`. | medium (confirmed) |
| S6 | `evalsuite/check.py:88-99` (committed) | The reset check compares only taps. A Subject without taps (`adapters/minimal.py`) whose `reset()` forgets nothing passes. | medium (confirmed) |
| S7 | `evalsuite/stats.py:68,84-88` (committed) | The bootstrap drops NaN draws without reporting how many, so the interval is conditional on scorable resamples (matters on 21 TEST scenes). `p_b_ge_a` ignores the metric direction, so for lower-is-better metrics it is the probability that b is worse. | medium (by reading) |
| S8 | `evalsuite/quantitative.py:238-254` (committed) | `compare` never checks that the two runs share source and task (GT version, cache). This is the bev_fusion "GT signature trap" again. | medium (by reading) |

Smaller issues:
- `quant trace a demo-1/3` raises `KeyError`; the label must be zero-padded.
- `render/video.py` calls `wslpath` under native Windows, and its `CANDIDATES` are absolute ffmpeg paths of this PC.
- `probing.situations` raises `IndexError` on an empty run.

### bev_fusion_lane_center
| # | where | failure case | severity |
|---|---|---|---|
| L1 | `evaluation/sparse_lanes/sparse_metrics.py:106` `graph_from_vertices` | The start rule lacks `len(outs[v]) != 1`. A split vertex with one predecessor is never a chain start, so every ordinary split becomes `[0,1,2]`, `[3,4]`, `[5,6]` plus the two-vertex leftovers `[2,3]`, `[2,5]` instead of three pieces. Connectivity and vertex counts stay right. But `topo_metrics.dup_counts` treats only direct siblings as non-duplicates, so near-parallel fork branches count as duplicates: 14/155 vs 0/153 on a test fork. `dup_frac` of every assembled graph is biased upward at forks, and `simplify_dp` can never drop the first vertex after a split. A fix changes graph hashes and golden numbers, so it needs a ledger row and a golden rewrite (decision). | high (confirmed by trace) |
| L2 | `evaluation/evaluate.py:66-68` `test_allowed` | `test_id in log` is a substring test, so `--split test --test-id E` (or `2026`) passes. `run.py:473` matches the whole id. Called directly, the evaluator has a weaker TEST guard than the launcher. | high (confirmed by reading; fix in the evaluator = lead chat) |
| L3 | `evaluation/lane_grid/grid_eval.py:140-153` | With `rg > 0` and `low` both set, the hysteresis drops far seed pixels. No caller uses both. | low |

Doc mismatches:
- `bootstrap.py` names a verdict the code never returns.
- `p16_chain_data.py:6` names `p14_chain_data.py`.
- `PROTOCOL.md:13` names `boot_eval.py` as the interval method, while CLAUDE.md names `bootstrap.py`.

### bev_fusion and datasets/NuScenes
See section 7 (F1 non-atomic GT cache, F2 sample/GT mismatch at a 0.75 s gap, F3 GT not pinned with the code, F4-F6).

## 2. Does the architecture make sense?

Mostly yes. The strong parts:
- **One truth per fact:** run state lives in run records, judgement in the ledger, the schedule is derived.
- **Narrow gates:** a hook on the truth files only (D21), and the tools are covered by selftest and review.
- **A code tag per run.**
- **A small agent set with templates.**

The cut "method + tools in `autoresearch/`, problem in its folder" is right for the standard-library core.

Where it bends:
- **`autoresearch/` now holds problem-specific code:**
  - `evalsuite/lanes/` (about 60 KB), `adapters/lanes_synth.py` and `selftest_lanes.py`; the only user is the lane project;
  - `probing.target_of` with lane words;
  - trainsuite defaults (11 GB, `"step "`, `total` / `group.*`) that describe bev_fusion;
  - `decide.LOWER` with lane metric names (D18).
- **The same statistics exist up to four times:** the paired scene bootstrap is in `evalsuite/stats`, `sparse_lanes/bootstrap.py` (sign a − b!), `topo_metrics.paired` and `bev_fusion/evaluation/bev_legacy.compare`. Lane measures are in `sparse_lanes` / `lane_grid` and again in `evalsuite/lanes` with slightly different `resample` / Douglas-Peucker rules. `decide.py` reads `card["compare"]["paired"]` from the project bootstrap, so an evalsuite compare written into a card would flip the sign.
- **Two run-state machines:**
  - `run.py`: done flag, stale after 300 s;
  - `trainsuite/runs.py`: `.done`, stale after 900 s, log modification times.
- **bev_fusion runs outside the launcher** (no card, ledger row or tag). The GT-signature incident of t7 / t8 is the cost of that. D23 (open) proposes the fix.
- **`bflc_env.sh` puts every `temp/` folder on PYTHONPATH,** and `_py14.sh` resolves script names inside `temp/`. Scratch can shadow live code by name, which happened with the literature backup's `catalogue.py` (moved now).

## 3. Leftovers: duplicates, dead copies, stale wrappers, old versions

| item | finding |
|---|---|
| `autoresearch/history/d16_staged/` | Already gone: removed from git in 1901e88 when D16 was deployed. Nothing refers to it. |
| `autoresearch/history/d22_tool_fixes.md` | Said "staged, NOT applied"; all six fixes are in ee19026. Header corrected. |
| `ROADMAP.md` vs `history/ROADMAP_2026-10-25.md` | Not a duplicate: the short open list and the frozen long review. |
| `bev_fusion/training/gt_bev.py` shim, `evaluation_pre_suite` | Section 7. |
| lane: `p16_infer.assemble_cluster` (lines 692-754) | Dead function, docstring "legacy", no caller. Remove in a reviewed change of the lead chat. |
| lane: `p16_infer._dp_keep` = `sparse_metrics._dp_keep` | Two identical copies kept equal by a comment, plus a third, different one in `evalsuite/lanes/geometry.simplify`. |
| lane: `drivers/boot_eval.py` vs `sparse_lanes/bootstrap.py` | Two scene bootstraps. `boot_eval` is still imported by `models/p13_*` and loops 11 / 17. |
| lane: `drivers/p17_latency.py` vs `p18_latency.py` (+ `p16_timing.py`) | `p18` is a superset (union 2, fork pruning), but `evaluate.py`, README and PROTOCOL name `p17`. Retire `p17` after those three are updated. |
| lane: `train/p18_joint.py` vs `p17_joint.py` | A copy with resume; already on CLEANUP_LIST C. |
| lane: `training/pipeline/p16_ensemble.py`, `p16_hyst.py`, `evaluation/drivers/p16_eval.py`, `p18_fork_prune_eval.py` | No importer, only README rows. Experiment scripts of loops 16 / 18: move to `evaluation/loops/loop16`, `loop18` (lead chat's code). |
| video encoding | `evalsuite/render/video.py` is a near copy of `bev_fusion/autogt/encode.py`. The lane's `tools/video/make_video.py` still writes mp4v. |
| `Subject.benchmark` vs `bev_fusion/evaluation/metrics.ms_per_frame` | The same timing statistics twice. |

## 4. What went (moved, nothing deleted)

Lane project. Moved by this review; each has a row in `bev_fusion_lane_center/_to_delete/README.md`:
- `tools/catalogue/lit16_merge.py`: its input `report_final/text/lit16/` no longer exists.
- `tools/catalogue/dump_catalogue_json.py`: no builder reads its output.
- `temp/literature_backup_2026-10-24/`: work filed on 2026-10-25; it shadowed a live script name.
- `temp/loop19_review/`: filed with `clean_temp.py`, tag `lcr/loop19_review-scratch`.

In the same README, a missing row for the older `cleanup_2026-10-24b/` was added.

Not moved, with the reason:
- `temp/loop19/`: `data/rebuild.json` names two of its scripts as builders. Move them to `training/data/prep/` first.
- `temp/loop20/`: loop 21 still uses `run_e253.sh` and `e255_matched.py`. The sub-review had called it safe; it is not.
- `temp/loop21/`: the running loop.
- The `p16` / `p18` drivers above: code of the loop-21 lead.

bev_fusion and datasets: section 7.

## 5. Recommendation: cut along trust, not along topic

Mirko's point is the key: a well-checked old evaluator is worth more than a new one an agent writes from scratch. The repository already protects truth (golden numbers, TEST log, champion), but not the code that produces it. The new evalsuite re-implements metrics that the lane project has golden-checked for 20 loops, and nothing stops a fresh implementation from becoming the source of a headline number.

The concrete recommendation is three layers with one rule each.

**1. `autoresearch/core`: frozen, reviewed, problem-independent.**
- Contents: the tools of today, plus the contracts (`card.schema.json`, the trainsuite log / CONTROL contract) and one statistics module (paired scene bootstrap, one sign convention: variant − control, seed recorded).
- Rule: changes only with selftest + `ar-reviewer`, as today.
- Moves out: `evalsuite/lanes`, `lanes_synth` and the lane words in `probing` go to `bev_fusion_lane_center/evaluation/`. Metric directions move from `decide.LOWER` to `project.json` (D18).

**2. Per problem: a "truth kit" pinned by hash.**
- Contents: the evaluator (metric code, splits, registry, golden numbers). Every metric card records the evaluator's code hash, and `decide.py` refuses cards whose hash differs from the one `golden.json` was made with.
- A new or rewritten evaluator (evalsuite included) enters in three steps:
  1. it runs next to the old one;
  2. it reproduces the golden numbers (layer 1 of the seven);
  3. a decision switches the pin.

  Until then it may draw pictures and probes, not headline numbers.
- Effect: agents keep full freedom to build analysis, but cannot silently replace the measuring stick. Bug L1 shows why the switch must be explicit: fixing it changes the golden numbers, and that should be one visible decision, not a side effect.

**3. Agent space: free.**
- Contents: `temp/loopNN`, `experiments/`, drivers.
- Rule: nothing in layers 1-2 imports from here. Drop `temp/` from the PYTHONPATH of `bflc_env.sh`, and extend `clean_temp.py` to scan `rebuild.json` and report builders, not only imports.

**bev_fusion fits the same cut:** a `project.json` + `run.py` (D23 option a) gives it cards, tags and the fingerprint that would have caught the three GT signatures. Its evaluator becomes its truth kit.

## 6. How to start a loop: newcomer walk-through and fixes

The path walked: root `CLAUDE.md` -> `bev_fusion_lane_center/CLAUDE.md` -> `METHOD.md` -> `prompts/loop.md` -> templates -> `install_agents.py`. Gaps found:

1. **Wrong relative paths.** `loop.md` wrote `tools/tick.py`, `tools/decide.py`, `tools/clean_temp.py` (relative to `autoresearch/`) next to `../autoresearch/tools/schedule.py` (relative to the problem folder). Run from the problem folder, `tools/decide.py` does not exist. `temp/README.md` named a `tools/maintenance/clean_temp.py` that was moved away on 2026-10-25.
2. **No pre-loop checklist.** The prerequisites existed only spread over METHOD.md and the lane CLAUDE.md:
   - P2 outputs: `PLAN.md`, experiment folders with `experiment.json` passing `experiment.py check`, schedule rows with commands, planned ledger rows;
   - the previous loop filed;
   - agents installed, hook, git identity.
3. **Loop number and E-id range.** The range must go into `project.json` "loops", a protected file. Nothing says so, and loops 20 and 21 were never added (D17).
4. **Where the filled prompt lives.** In practice it is `experiment_execution/history/prompts/PROGRAM_loopNN.md`; no doc said so.
5. **How to start `/loop`.**
   - Which folder: the workspace root, so the agents and the hook load.
   - Interval or not: none, the loop is self-paced.
6. **Windows.** `python tools/install_agents.py` in the README fails on this PC; there is no Windows python.

Fixed in this commit:
- `autoresearch/prompts/loop.md` now opens with a six-step "Start a loop" section and uses one path form throughout.
- `METHOD.md` points to it.
- The lane CLAUDE.md names the checklist and the `PROGRAM_loopNN.md` place.
- The lane `temp/README.md` and `experiment_execution/README.md` are corrected.
- The D23 row in `DECISIONS.md` now spells its bev_fusion paths in full, so `check_structure.py` is clean.

Open: D17 (who writes the loop range into the protected `project.json`); its recommended option (b) removes the protected edit from every loop start.

## 7. bev_fusion and datasets/NuScenes

These jobs were running during the review and were not touched:
- `lane4.sh` (`bev_suite.py` eval of t8);
- `finish2.sh`;
- `run_cfg.sh` training of `t8_ctl_s0`;
- `queue_fused.sh` (`tests/test_flow_fused.py`);
- `run_autogt_trainval.sh`;
- `move_ckpts.sh --loop`;
- a trainsuite `watch`.

### Bugs
| # | where | failure case | severity |
|---|---|---|---|
| F1 | `training/nuscenes_dataset.py:158` | `np.savez_compressed(path, ...)` writes the GT cache straight to its final name. A killed or concurrent writer leaves an empty or truncated `.npz`, and the next `exists()` -> `np.load` crashes training. It happened: `_to_delete/gt_cache_partial/..._15efbde2d1.npz`, 0 bytes, 11:45. The `finish2.sh` workaround catches only 0-byte files, not truncated ones. The radar cache and autogt already write to a temporary file and `os.replace`. | high (confirmed by reading + the file) |
| F2 | `training/losses.py:378-386` `sample_targets` | GT is bucketed by `nominal_dt(meta dt)` and the sample index is dropped. One trainval keyframe gap (scene-0994, train parts 7-10) is exactly 0.75 s; Python rounds 1.5 to 2, so it lands under ±1.0 s, and `gts[±0.5]` gets B-1 entries for B predictions. After the dt warm-up, `soft_bce` raises a size mismatch. It is latent today (configs list parts 1-6 only); a config with the default empty `train_scenes` (700 scenes) would crash. Fix: key by side (prev / next), not by rounded dt. | high (by reading; the gap measured by the sub-review) |
| F3 | `temp/train/run_cfg.sh` + `training/gt_bev.py:9-13` | A "pinned" run snapshots only `training/`, but the GT is built from the live `autogt/`, and the GT signature is recorded in neither `config.json` nor `metrics.json`. `bev_suite.py compare` does not check it either. Two arms pinned at the same commit can therefore train or score on different GT. This is the t7 / t8 incident, and the root cause the S8 compare check and the D23 fingerprint would close. | high (design) |
| F4 | `evaluation/bev_suite.py:262-266` | The legacy compare fallback ignores `--n`: `scene_records{tag}.pkl` is overwritten by the newest eval, and `bev_legacy.compare` pairs by scene without checking frame identity. A compare with a pre-suite run (`t2_ref_s0`) can mix frame sets. | medium (by reading) |
| F5 | `datasets/NuScenes/tools/link_views.sh:12` | The guard `find /proc/*/fd -lname "$HOME/data/*"` never fires: open files through a view resolve to `/mnt/d/...` (tested: 0 matches with a file open). A re-run swaps the views under running jobs. Documented in the dataset README now; the fix is to match `/mnt/d/Auto-research/NuScenes/*` and the additions path. | medium (confirmed) |
| F6 | `nuscenes_gt.py:298`, `autogt/cli.py:79-84` | `flow_sigma` is not divided by covered sub-pixels, so edge cells come out up to 4x too small. The res / ss collision guard of autogt works only after a finished build. | low |

### Leftovers
- **`training/gt_bev.py` (shim to `autogt/`) must stay for now.** It is imported live by:
  - `nuscenes_dataset.py`, `losses.py`, `feature_cache.py`, `distill_light_encoder.py`;
  - 8 temp scripts.

  Switch those imports to `autogt` when no chain job runs. Old run snapshots carry their own copy.
- **`_to_delete/evaluation_pre_suite/`** has a complete README. Nothing imports the moved modules.
- **The `evaluation/evaluate_run.py` stub has no caller left** (lane 3 ended). It can go together with its check in `bev_selftest.py:398ff` and its README rows. That is shared code (a reviewed change), so it is not moved here.
- **Two GT stores:**
  - `data/cache/gt` is per sample, so each keyframe is stored about 3 times, without atomic writes: 11 GB on C:, 10 signatures.
  - `data/autogt/<sig>/` is per keyframe and atomic.

  C: has 5.7 GB free. A dry run of `python -m autogt clean --stale-caches data/cache/gt` shows what the old signatures hold. They are what an exact re-evaluation of t2 / t3 needs, so purging them is Mirko's decision.
- **Off-chain training modules:** `temporal.py`, `s1_camera.py`, `train_s1.py`, `dualbev_front.py`, `light_encoder.py`, `distill_light_encoder.py`, `camera_decoder.py`, `radar_decoder.py` are imported only by tests / temp. They are candidates for `training/legacy/`, not for deletion.
- **Every new eval still writes the legacy `scene_records.pkl`,** only for four temp readers.
- **Gaps in the gate scripts:** `tests/run_gates.sh` does not run the selftests of model / losses / train / dataset or `autogt selftest`; `tests/run_after_t1.sh` is stale.
- **Intentional copies:** `autogt/encode.py` vs `evalsuite/render/video.py`, and the stop check in `train.py` vs `trainsuite/control.py`. These are acceptable: the pinned snapshot needs them.

### Moved by this review
- **bev_fusion:** 51 tracked one-offs to `bev_fusion/_to_delete/temp_oneoffs/`, with the existing `<origin>__<file>` naming and a README row:
  - 36 from `temp/checks/`: the t1-era `chain_a` ... `chain_x` and job probes;
  - 9 from `temp/train/`;
  - 6 from `temp/setup/`.

  Criteria: tracked, unchanged since 2026-10-27, the full file name occurs in no tracked file, and no running script names it. The sub-review had proposed about 30 more. They were left because they changed on 10-29 .. 10-31 (other chats may still use them) or because they are cited by experiment cards:
  - `study/lane*.sh`, `t7/finish.sh` and similar: move after lane 4 / finish2, together with their cards.

  A README row was also added for `gt_cache_partial/`.
- **datasets/NuScenes (not in git):** moved to `datasets/NuScenes/_to_delete/`, with a new README:
  - the byte-identical copies `tools/history/list_nuscenes_bucket.py` and `verify_s3_etag.py`;
  - the superseded `tools/move_additions.sh`.

  The README is corrected: camera sweeps complete, disk state, the derived bev_fusion folders on D:, the `link_views.sh` guard bug.

## Follow-ups, by owner
- **Tools chat (autoresearch/tools), reviewed change:** T1 in `decide.py` (flags for a guard-only headline), with a selftest case.
- **evalsuite / trainsuite chat:** S1-S8 with selftest cases; one bootstrap and sign convention shared with `decide.py`.
- **bev_fusion chats:**
  - F1: atomic GT cache write;
  - F2: key GT by side;
  - F3: GT signature into `config.json` / `metrics.json`, compare refuses on a mismatch;
  - F4;
  - switch the four `gt_bev` imports, then retire the shim and the `evaluate_run.py` stub (after lane 4 / finish2).
- **Loop-21 lead (lane code):**
  - L1, as a decision with a golden rewrite;
  - L2, by using `run.py`'s whole-id regex;
  - remove the dead `assemble_cluster`;
  - move the loop-16/18 drivers into `loops/`;
  - move the two rebuild scripts out of `temp/loop19`, then `clean_temp.py loop19` / `loop20` when loop 21 is filed.
- **Mirko:**
  - D17: loop ranges;
  - D18: metric directions in `project.json`;
  - D23: bev_fusion under `run.py`;
  - the truth-kit pin of section 5, as a new decision if wanted.

## 8. What was done (D23, D24, same day)
Mirko's answer (recorded as D23 and D24 in `bev_fusion_lane_center/experiment_execution/DECISIONS.md`): fix the bugs first, move the lane-specific code out of `autoresearch/`, keep one paired scene bootstrap, run bev_fusion inside `run.py`, remove unnecessary code. Commits 144fb94, 1db792f, 423f093, a3ea642, eae2ff4, a41df85; a fresh `ar-reviewer` pass over the tools and one over the evaluation code, their findings fixed.
- Fixed: T1, S1-S8 (S2 / S3 already by the trainsuite chat), L2, L3, F1-F5, the small evalsuite items; the reviewer's later finds (compare refused the t8 feature swap, `--force` not passed, legacy pickle name, tensors in the reset check, the exception line of a crash, `n_valid`).
- L1: the fixed cut is in `graph_from_vertices`, but the product path keeps the old cut (`legacy_cut=True`) until route success and the consumer are re-measured on the GPU; no golden number changed.
- Moved out of `autoresearch/`: `evalsuite/lanes` -> `bev_fusion_lane_center/evaluation/lane_task/`; the lane metric names of `decide.py` go to `project.json` "lower_is_better" (the lane problem's key waits for a grant: the session's permission check refused the grant command, so `decide.py` keeps the lane names as a fallback for tag prefix lcr).
- One bootstrap: `autoresearch/evalsuite/stats.py` (`scene_picks`); the lane intervals byte-identical; `topo_metrics`, `boot_eval`, `bev_legacy` use it.
- bev_fusion: `project.json`, `records/`, `temp/train/run_ar.sh` (trains from the run's snapshot commit, exits with the trainer's code); GPU slots shared with the lane problem.
- Removed / moved: dead p16 code and loop-16/18 drivers, off-chain bev_fusion modules (`training/legacy/`, `_to_delete/off_chain_2026-10-31/`), the `evaluate_run.py` stub.
- Metric directions (Mirko, later the same day): every card declares them in a `"better"` map written by its evaluator (lane: `sparse_metrics.LOWER_IS_BETTER`, 8722a94; bev_fusion: each metric's `direction()`, written by `bev_suite.py eval`, which now also writes `eval/card<tag>.json` in the card contract, because its `metrics.json` is not a contract card and `decide.py` could not read it), and `decide.py` reads them there. The prior-map baseline of `eval --prior` overwrote the model's box numbers with its own (older bug, found by the reviewer); it now adds only its map numbers. The fallback for older cards is only project.json `lower_is_better` (the lane key written under a D24 grant; `decide.py`'s built-in lane list removed), matched by the key's last part, because bev_fusion's path-like keys (`det/dt+0.00/vehicle/all/ate`) never matched its list before. Two wrong directions in `bev_task.py` fixed: `field_mae` and the loss terms are lower-is-better.
- Open: route-success measurement for L1, `training/augment.py` selftest fails (radar mirror, older than this work), the gained-key tolerance hides a GT change of a store that predates the GT signature (warning only), one intermittent failure of the autoresearch selftest seen once under full load (`S.slowA/run.log` missing; not reproduced in six runs).
