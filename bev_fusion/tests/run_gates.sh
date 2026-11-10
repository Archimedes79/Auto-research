#!/usr/bin/env bash
# Run the synthetic gates of bev_fusion block by block and report pass / fail per block (plan.md section 10.1).
# Fast self-tests always; the slow self-tests of the chain (model, losses, trainer, autoGT, radar slot, projection) only with --all
# (they take minutes on the CPU). The early studies that were once gates here (flow from frames / correlation, 2026-10) live in
# studies/early/ (D48): they train for up to 40 minutes and their recorded result is a narrow or open FAIL of the self-supervised arm.
# Usage (from the bev_fusion folder): wsl bash tests/run_gates.sh [--all]
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"     # GtGenerator lives at the repository root since 2026-11-01 (`-m GtGenerator selftest`)
PY="${PY:-$HOME/miniconda3/envs/gcar/bin/python}"
fail=0
run() {  # run <block> <command...>
  local name="$1"; shift
  local out
  if out=$("$@" 2>&1); then
    echo "PASS  $name  | $(echo "$out" | grep -v -i warn | tail -1 | cut -c1-140)"
  else
    echo "FAIL  $name  | $(echo "$out" | grep -v -i warn | tail -3 | tr '\n' ' ' | cut -c1-200)"
    fail=1
  fi
}
run "synthetic world"   "$PY" tests/support/synthetic_world.py
run "lift PD"           "$PY" training/lift_pd.py
run "camera encoder"    "$PY" training/camera_encoder.py
run "radar encoder"     "$PY" training/radar_bev.py
run "radar encoder, lossless" "$PY" training/radar_encoder.py     # delta-coded slot raster: exact round trip, side loss, monitor
run "splat"             "$PY" training/splat.py
run "BEV trunk"         "$PY" training/bev_trunk.py
run "temporal fusion"   "$PY" training/temporal_fusion.py
run "fusion"            "$PY" training/fusion.py
run "timed heads"       "$PY" training/timed_head.py
run "object head"       "$PY" training/object_head.py
run "metrics"           "$PY" evaluation/metrics.py
run "front compare"     "$PY" evaluation/front_compare.py
run "paper fallbacks"   env OMP_NUM_THREADS=1 "$PY" training/fallbacks.py
run "oracle lift"       "$PY" tests/test_oracle_lift.py
run "Doppler"           "$PY" tests/test_doppler.py
run "GT cache, targets" "$PY" tests/test_gt_cache_targets.py
run "speed flags"       "$PY" tests/test_speed_flags.py            # run_clip against its per-term float() version, switches, resume rules, profiler
run "host overhead"     "$PY" tests/test_host_overhead.py          # target window, one host read per call, multi-tensor EMA
run "grad scaler"       "$PY" tests/test_grad_scaler.py            # amp fp16: scaler exact against plain steps, overflow, bad-step rule, checkpoint
run "lr groups"         "$PY" tests/test_lr_groups.py              # a module's own lr and warm-up: groups, schedule, resume
run "hygiene"           "$PY" tests/test_hygiene.py                # w_obj as a weight, unknown velocity, splat precision, batch_ego (the flag hygiene_2026_11_02)
run "clip split"        "$PY" tests/test_clip_split.py             # grad_clip_split: each listed module and the rest clipped on their own, the global call unchanged, resume, call site
if [ -f runs/t10_dual_s0/ckpt_00003000.pt ]; then               # its G4 warm-starts from this real checkpoint (not in git)
  run "velocity hurdle" "$PY" tests/test_velocity_hurdle.py     # t11; CPU, about 1 min, 2.2 GB peak
else
  echo "SKIP  velocity hurdle  | runs/t10_dual_s0/ckpt_00003000.pt missing (G4 needs it)"
fi
run "slots, every kind" "$PY" tests/test_slots.py                   # five-layer build: contract test of every kind of every slot; CPU, no data, about 1 min
run "fusion area"       "$PY" evaluation/bev_fusion_area.py --selftest   # footprint shares, GT classes, neutral detections, coverage classes (D37 / D38); no data
run "curve metrics"     "$PY" evaluation/bev_curves.py --selftest      # AP / AUC / Brier against the exact curves of sorted scores, additivity, one-class / NaN / float16
run "module probe"      "$PY" evaluation/bev_modules.py --selftest      # tensor numbers, dead channels, planted velocity offsets, each response perturbation changes only what it names; no data
run "module timing"     "$PY" evaluation/bev_timing.py --selftest       # toy modules with known sleeps: per call / per frame, calls, hooks removed, training split, idle rule; CPU
run "eval sets"         "$PY" evaluation/bev_sets.py --selftest         # the sets of the evaluation = the splits of splits/ (grammar, TEST refused, tags, lists, a toy folder); no data
run "error search"       "$PY" evaluation/bev_errors.py --selftest       # FP / FN kinds, counts equal the metric's matching, frame trace through the files; no data
run "view helpers"       "$PY" evaluation/bev_selftest.py --only "bev_views helpers"   # error colours of boxes, faded history, radar layer, velocity hue, moving layer, loss heat; seconds
if ls data/cache/gt/*_15efbde2d1.npz > /dev/null 2>&1; then    # the evaluation on real cached GT: GT as the prediction must reach every ceiling, degraded copies fall, the GT frames agree with their poses
  run "GT against GT"   "$PY" evaluation/bev_gtcheck.py --selftest     # CPU, about 1 min, no model
  run "response test"   "$PY" evaluation/bev_response.py --selftest    # CPU, about 2 min, no model: GT worsened by known amounts, every number must move by exactly that amount
else
  echo "SKIP  GT against GT  | no cached GT of signature 15efbde2d1 in data/cache/gt"
fi
run "fusion-area loss" "$PY" training/losses.py --fusion-area              # D38 (b) option: weights, ignored objects, the gradient through BEVLoss; CPU, seconds, no optimiser step
run "fusion area, real" "$PY" tests/test_fusion_area_real.py               # the same option on real cached GT frames (skips without data/cache/gt); seconds
run "host targets"     "$PY" tests/test_host_targets.py                    # D52: build_targets / GT stacking bit for bit against the old code; CPU, seconds
run "launch saving"    "$PY" tests/test_launch_saving.py                   # D53: share_mover / splat_fused / sync_free equal to the plain code (bit for bit on the CPU); CPU, about 30 s
run "host batch"       "$PY" tests/test_host_batch.py                      # D54: the host copies (batch["host"]), the one-copy ego tensors, the keyed lift: the same numbers, no wait for the GPU (the GPU part runs when one is there); about 1 min
run "A/B timer"        "$PY" tests/test_ab_timing.py                       # the in-process A/B timing of the speed switches: arms, schedule, paired difference, a tiny model on the CPU; about 30 s
run "startable"        "$PY" training/startable.py                        # static config check, GT identity, smoke twin; CPU, seconds
run "orientation"      "$PY" tests/test_orientation.py             # every drawer: x up, +y left, arrows, colour keys, overlays
run "nuScenes dataset"  "$PY" training/nuscenes_dataset.py          # nuScenes mini; about 20 s
run "mirror augment"    "$PY" training/augment.py                   # nuScenes mini; about 30 s
run "bundle"            "$PY" tests/test_bundle.py                  # model.py + losses.py import no loader and no trainer (the network is a closed bundle); 1 s
run "camera features"   "$PY" tests/test_camera_features.py         # the CAMERAS table, feature swap, dual branches, refusals; CPU, 12 s
run "dual camera geometry" "$PY" tests/test_dual_camera_geometry.py   # both branches put an image pixel at the same BEV cell (< 0.5 px); CPU, 5 s
run "freeze, probe"     "$PY" tests/test_freeze_probe.py            # freeze.py / probe.py / debug.py on a toy network; CPU, 4 s
run "GT prepare"        "$PY" tests/test_gt_prepare.py              # scene lists, GtGenerator command lines; no data, 5 s
run "run_ar launcher"   "$PY" tests/test_run_ar.py                  # D73: tools/run_ar.sh unpacks training/ and trainkit of the commit, the trainer imports the pinned copy, exit code, .done; a toy repo, 5 s
run "radar memory ghost" "$PY" tests/test_radar_memory_ghost.py     # the Doppler write mask keeps a radial mover out of the radar memory; 3 s
run "motion block, joint" "$PY" tests/test_motion_joint.py          # the t15 motion block through the trainer's path; CPU, 7 s
run "split lists"       "$PY" tests/test_splits.py                  # splits/*.json, training/splits.py, the trainsuite hook; no data, 7 s
run "layer table"       "$PY" tests/test_layers.py                  # training/layers.py: the table changes no number, the 1 m centre-line row; CPU, 7 s
run "line BCE"          "$PY" tests/test_line_bce.py                # t23: the tolerance-aware BCE on the line presence (analytic value, gradients, head isolation, default path = dice + huber); CPU, seconds
run "centre-line eval"  "$PY" tests/test_centreline_eval.py         # the 1 m centre-line head in the evaluation (D48): oracle / null / shuffled on real cached GT, hand examples, old runs' counts unchanged, 19 mutants killed; CPU, about 100 s
run "lane branch, dual" "$PY" tests/test_lane_branch_dual.py        # L4 on camera_features 'dual' (t20): the rule, state_dict as before, zero start, gradient, geometry inputs, mirror, t20_base.json diff; CPU, no data
run "lane pull"         "$PY" tests/test_lane_pull.py                # t24: ModelConfig.lane_use_depth (lift P instead of lift PD in the lane branch): default, same tensors, weighting, oracle, gradient, mirror, config; CPU, about 15 s, no data
run "precompute"        "$PY" training/precompute.py               # layer 1: camera map, radar fan, fusion area, radar hits / static cells against their sources; CPU, seconds, no data
run "feature store"     "$PY" training/feature_cache.py shards-selftest         # atomic shard round trip, feature-size guard, a cut build is refused and rebuilt; CPU, 8 s
run "camera off"        "$PY" training/feature_cache.py camera-off-selftest     # camera_off on the cached model: zero camera BEV; CPU, 8 s
run "stream temporal"  "$PY" tests/test_stream_temporal.py         # cam_buffer pair / recurrent (t21): stream = batch, buffer rules, gradient through the carry, window unchanged, 8 mutants; CPU, about 2 min, no data
run "stream tools"     "$PY" tests/test_stream_tools.py              # the promoted t21 diagnostics (stream_drift, stream_timing, arm_guards): drift smoke on nuScenes mini, timing smoke, guards table; CPU, about 30 s
run "student encoder"  "$PY" tests/test_student_encoder.py          # t22 stage A: the student's contract, H8 geometry, gradients, distillation loss, CLI stop / resume on a synthetic tree; CPU, about 12 s, no data
run "student chain"    "$PY" tests/test_student_chain.py            # t22 stage B: camera_student in the chain: off = old code, frame-0 splice, student_side, trainable / frozen student, anchor, refusals; CPU, about 30 s, no data
run "evaluation api"    "$PY" tests/test_evaluation_api.py          # the class interface (toy project, the adapter's identity, TEST refused); no data, seconds
run "runner class"      "$PY" tests/test_runner_class.py          # the class Runner and python -m runner on a toy network: no ground truth read, plan, reuse, taps, exit codes; no data, seconds
run "suites conform"    "$PY" tests/test_suites_conform.py        # Training, Evaluation, Runner: one convention (methods, --json, exit codes 0/1/2, verbs first, layering); about 40 s
run "evaluate door"     "$PY" tests/test_evaluate_door.py           # evaluation/evaluate.py translates options into bev_suite commands (--real: the golden check, by hand); no data, 1 s
run "report build"      "$PY" tests/test_report_build.py            # results/build: a fixture report (every number in a source file, every cell in place), missing / wrong inputs stop the build, 23 mutants; CPU, 8 s, no data
run "trainkit selftest" bash -c "cd ../autoresearch && $PY -m trainkit.selftest"    # the engine of the trainer on a toy network; CPU
run "train guards"     "$PY" tests/test_train_guards.py                 # D73: stop + resume of the real train() = the uninterrupted run (EpochLoader), evaluate with non-finite batches, check_engine; CPU, toy, 13 s
run "trainkit equivalence" "$PY" tests/test_trainkit_equivalence.py # train() of commit df9d437 against the live one (trainkit): rows, weights, EMA, optimiser, RNG, files, exit codes; CPU, toy network
if [ "${1:-}" = "--all" ]; then
  run "whole chain"    "$PY" training/model.py                     # nuScenes mini, CPU; minutes
  run "losses"          "$PY" training/losses.py                    # CPU; minutes (overfit 100 steps)
  run "radar slot"      "$PY" tests/test_radar_slot.py              # delta_slots in the whole chain: side loss, gradients, old path unchanged; about 2 min
  run "trainer"         "$PY" training/train.py                     # nuScenes mini + data/cache/cam_feats_mini, CPU; minutes
  run "autoGT"          "$PY" -m GtGenerator selftest
  run "projection (R-GEO)" "$PY" tests/test_projection_general.py
  if ls runs/t16_full_s0/ckpt_*.pt > /dev/null 2>&1; then          # the front door of the evaluation against bev_suite.py eval on a copy of the real run
    run "evaluate door, golden" "$PY" tests/test_evaluate_door.py --real   # 4 val frames on the CPU, number for number; about 3 min, 2 x 6 GB
    run "evaluation api, real" "$PY" tests/test_evaluation_api.py --real   # the class interface equals the engine on a real frame (CPU, 4 frames, about 5 min)
    run "evaluate door, sets"   "$PY" tests/test_evaluate_door.py --sets   # large.val / large.train against val / train, small.val's own scenes; 4 frames on the CPU; about 6 min, 2 x 6 GB
  else
    echo "SKIP  evaluate door, golden  | no checkpoint in runs/t16_full_s0"
  fi
fi
exit $fail
