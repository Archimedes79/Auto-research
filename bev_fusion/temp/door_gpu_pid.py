"""throw-away: run bev_timing.run_timing with a tracing gpu_state: what does each idle check see, and is the model on the GPU at the second one?"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "autoresearch"))
sys.path.insert(0, ROOT)
from evaluation import bev_timing as B  # noqa: E402

real = B.gpu_state
t0 = time.time()


def traced(samples=3, gap=0.5):
    st = real(samples, gap)
    reserved = None
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        reserved = round(sys.modules["torch"].cuda.memory_reserved() / 2**20)
    print(f"[t+{time.time() - t0:5.1f}s] gpu_state(samples={samples}): util {st['util_percent']} mem {st['mem_used_mb']} MB, torch reserved {reserved} MB, "
          f"others {st['other_processes']} busy {st['busy_processes']} (own pid {os.getpid()})", flush=True)
    if samples == 2:                                                    # the second check: what follows it
        import subprocess
        for i in range(8):
            time.sleep(0.5)
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            print(f"   +{0.5 * (i + 1):.1f}s: {q}", flush=True)
    return st


B.gpu_state = traced
run = os.path.join(ROOT, "temp", "door_check", "door", "t16_full_s0")
res = B.run_timing(run, n=5, warmup=2, device="cuda", out=os.path.join(ROOT, "temp", "door_check", "probe_timing"), ckpt_dir=os.path.join(ROOT, "runs", "t16_full_s0"))
print("total ms", res["inference_ms"]["total_ms"], "idle", res["idle"])
