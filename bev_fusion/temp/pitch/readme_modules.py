"""README edits of the module probe / timing / taps unit: apply to a file given as argv[1] (the blob of HEAD for the commit, the working copy for the folder)"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)


rep("python evaluation/bev_suite.py runs/t1_r4_soa time                                  # ms per frame, idle GPU only -> eval/latency.json\n",
    "python evaluation/bev_suite.py runs/t1_r4_soa time                                  # ms per frame, idle GPU only -> eval/latency.json\n"
    "python evaluation/bev_suite.py runs/t16_small_s0 modules [--frames 3]               # module probe on CPU (bev_modules.py): output statistics, dead channels, the response when the camera / all radar detections / only the Doppler are removed, the velocity field on static and moving cells, a warm second frame, one PNG per frame -> eval/modules/\n"
    "python evaluation/bev_suite.py runs/t16_small_s0 timing [--n 30]                    # ms per module at batch 1 + forward / backward of a training micro-batch (bev_timing.py), idle GPU only (refused otherwise) -> eval/timing_modules.json\n")
rep("forward hooks for ten taps, batch-1 latency,",
    "forward hooks for the taps (camera, radar incl. the delta-slot features and the lossless slot raster, the radar memory, the fused features, the trunk, the motion block's three levels, the hurdle's moving map, the radar aux head: `qual debug` shows what the network has), batch-1 latency,")
rep("| `bev_selftest.py` | known-answer tests",
    "| `bev_modules.py` | module probe of a run (`bev_suite.py RUN modules`, CPU): per module output (every tensor inside it) shape / finite / mean / std / zero share / dead channels, the warm second frame for the memories, the RESPONSE test (camera removed, all radar detections removed, only the Doppler removed: relative change of the memories, fusion, trunk, motion block and heads), the velocity field on static and moving GT cells (offset along the ego velocity?), one PNG per frame; -> `eval/modules/{modules,response,static_check}.json` + `frame_<k>.png` | `--selftest` (known tensor numbers, planted velocity offsets, each perturbation changes exactly the keys it names, the recorder restores the trunk, pictures; in `tests/run_gates.sh`); on `t16_small_s0` the numbers equal the stored scratch probe |\n"
    "| `bev_timing.py` | ms per module at batch 1 (`bev_suite.py RUN timing`): pre / post hook with a CUDA sync around every module of `bev_modules.MODULES` (the trunk through `.features()`), per call and per frame (the heads run once per dt), the unsynchronised total next to the sum, the forward / backward split of a training micro-batch, parameters per module; the load average, GPU utilisation and other GPU processes are stored with the numbers (`idle`) and a busy GPU is refused (`--force` marks the result not idle) -> `eval/timing_modules.json` | `--selftest` (toy modules with known sleeps: per call, per frame, calls, hooks removed, training split, idle rule; in `tests/run_gates.sh`) |\n"
    "| `bev_selftest.py` | known-answer tests")
open(p, "w", encoding="utf-8").write(s)
print("ok")
