"""scratch: the t26 G2 smoke: s per step, peak memory, losses, non-finite, the student anchor and jitter phase"""
import json
import os
import math

B = "/mnt/c/Auto-research/bev_fusion/"
rj = json.load(open(B + "records/runs/smoke.t26_one_s0/run.json"))
print({k: rj.get(k) for k in ("status", "exit_code", "started", "ended", "duration_s", "commit", "tag", "attempts")})
rows = [json.loads(l) for l in open(B + "runs/smoke_t26_one_s0/log.jsonl")]
tr = [r for r in rows if "val" not in r]
va = [r for r in rows if "val" in r]
print("train rows:", len(tr), "| val rows:", len(va))
keys = ["step", "s_per_step", "gpu_gb", "total", "grad_norm", "student_anchor", "grad.student", "t_data", "t_step", "lr", "lr.student"]
print(" ".join(f"{k:>14s}" for k in keys))
for r in tr:
    print(" ".join(f"{(r.get(k) if r.get(k) is not None else float('nan')):14.4f}" for k in keys))
for r in va:
    print("val", r["step"], {k: round(v, 3) for k, v in r["val"].items() if k in ("total", "group.areas", "group.lines", "group.objects", "group.motion", "student_anchor", "n", "n_skipped")})
bad = [r["step"] for r in tr if not all(math.isfinite(v) for v in (r.get("total", 0.0), r.get("grad_norm", 0.0)))]
print("non-finite rows:", bad)
# per-phase step time: the s_per_step is a running mean since the start of this process; derive the incremental time per step from the "t" stamps
ts = [(r["step"], r["t"]) for r in tr]
for (s0, t0), (s1, t1) in zip(ts, ts[1:]):
    print(f"steps {s0:4d} -> {s1:4d}: {(t1 - t0) / (s1 - s0):6.2f} s per step")
print("log files:", sorted(os.listdir(B + "runs/smoke_t26_one_s0"))[:20])
