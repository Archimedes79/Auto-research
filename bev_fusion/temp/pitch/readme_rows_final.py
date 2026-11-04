"""set the final text of the bev_modules / bev_timing rows (and the adapter tap sentence) in the README given as argv[1]"""
import sys

p = sys.argv[1]
lines = open(p, encoding="utf-8").read().split("\n")
rows = {
    "| `bev_modules.py` |": "| `bev_modules.py` | module probe of a run (`bev_suite.py RUN modules`, CPU unless `--device`): for every direct child module of the model (lift AND lift2 of a dual camera, lane, ... ; the trunk through `.features()`) the output tensors' shape / finite / mean / std / zero share / dead channels, a warm second frame for the memories, the RESPONSE test (camera removed incl. the lane images, all radar detections removed, the Doppler of every detection set to 0 = \"all stand still\": relative change of the memories, fusion, trunk, motion block and heads; a perturbation that changes nothing prints a WARNING), the velocity field on static and moving GT cells (offset along the ego velocity?), one PNG per frame; heads are recorded at their first call (dt -0.5), `OUT dt0:` holds dt 0; `aux_cam` / `aux_rad` run only in training; -> `eval/modules/{modules,response,static_check}.json` + `frame_<k>.png` | `--selftest` (tensor numbers, planted velocity offsets, each perturbation changes exactly the keys it names, the recorder restores the trunk and copies what it stores, pictures; in `tests/run_gates.sh`); on `t16_small_s0` (4 frames) the numbers equal the stored output of the scratch probe |",
    "| `bev_timing.py` |": "| `bev_timing.py` | ms per module at batch 1 (`bev_suite.py RUN timing`): pre / post hook with a CUDA sync around every direct child of the model (the trunk through `.features()`), EXCLUSIVE time (a module called inside another is taken off the caller), per call and per frame (the heads run once per dt), the unsynchronised total next to the module sum and `unattributed_ms` (host work outside the modules + sync overhead: read shares, not an exact budget), parameters per module, the forward / backward split of a training micro-batch (median, min, max; BatchNorm statistics restored); the fp32 cold-memory path of `benchmark()`. Idle GPU only: utilisation (max of 3), memory in use, `ps` for trainers / evaluations (nvidia-smi lists no process under WSL2), load, checked before the model load and again before the timed loop; a busy GPU is refused (`--force` marks the result not idle) -> `eval/timing_modules.json` | `--selftest` (toy modules with known sleeps: per call, per frame, exclusive nesting, calls, hooks removed and outputs unchanged, training split, idle rule incl. unknown = busy and the `ps` parser; in `tests/run_gates.sh`) |",
}
out = []
for ln in lines:
    for prefix, new in rows.items():
        if ln.startswith(prefix):
            ln = new
    out.append(ln)
open(p, "w", encoding="utf-8").write("\n".join(out))
print("ok")
