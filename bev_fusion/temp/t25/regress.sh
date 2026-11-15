#!/usr/bin/env bash
# scratch: the existing tests of the touched folders, one after the other under nice; one result line each, full output in temp/t25/out/<name>.txt
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2
export PYTHONPATH="/mnt/c/Auto-research${PYTHONPATH:+:$PYTHONPATH}"
PY="$HOME/miniconda3/envs/gcar/bin/python"
mkdir -p temp/t25/out${OUTSUF}; export SHIM="${SHIM:-}"
run() {  # run <name> <args...>
  local name="$1"; shift
  local t0=$(date +%s)
  if nice -n 15 "$PY" $SHIM "$@" > "temp/t25/out${OUTSUF}/$name.txt" 2>&1; then r=PASS; else r=FAIL; fi
  echo "$r  $name  ($(( $(date +%s) - t0 )) s)  | $(grep -v -i warn temp/t25/out${OUTSUF}/$name.txt | tail -1 | cut -c1-160)"
}
for t in "$@"; do
  case "$t" in
    interface)   run interface training/interface.py ;;
    precompute)  run precompute training/precompute.py ;;
    stream)      run test_stream_temporal tests/test_stream_temporal.py ;;
    slots)       run test_slots tests/test_slots.py ;;
    lanepull)    run test_lane_pull tests/test_lane_pull.py ;;
    speed)       run test_speed_flags tests/test_speed_flags.py ;;
    host)        run test_host_overhead tests/test_host_overhead.py ;;
    hostbatch)   run test_host_batch tests/test_host_batch.py ;;
    radarslot)   run test_radar_slot tests/test_radar_slot.py ;;
    motion)      run test_motion_joint tests/test_motion_joint.py ;;
    guards)      run test_train_guards tests/test_train_guards.py ;;
    memghost)    run test_radar_memory_ghost tests/test_radar_memory_ghost.py ;;
    hygiene)     run test_hygiene tests/test_hygiene.py ;;
    dual)        run test_lane_branch_dual tests/test_lane_branch_dual.py ;;
    clip)        run test_clip_split tests/test_clip_split.py ;;
    student)     run test_student_chain tests/test_student_chain.py ;;
    camfeat)     run test_camera_features tests/test_camera_features.py ;;
    model)       run model_selftest training/model.py ;;
    dataset)     run nuscenes_dataset_selftest training/nuscenes_dataset.py ;;
    augment)     run augment_selftest training/augment.py ;;
    fcache)      run feature_cache_selftest training/feature_cache.py ;;
    e2e)         run test_end_to_end tests/test_end_to_end.py ;;
    new)         run test_radar_recursive tests/test_radar_recursive.py ;;
  esac
done
echo done
