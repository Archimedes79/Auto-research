"""scratch: the R gate of the card section 'Seed twin and R arm': exit 0 when the ResNet-18 distillation run t22.A2R_s0 exited 0 and its validation depth skill is
skill16 >= 0.695 and skill8 >= 0.652 (not more than 0.02 below the EfficientViT-B1 student's 0.715 / 0.672); prints the numbers; exit 1 otherwise."""
import json
import math
import sys

RUN = "runs/t22_A2R_s0"
REC = "records/runs/t22.A2R_s0/run.json"
try:
    rec = json.load(open(REC))
    m = json.load(open(RUN + "/metrics.json"))
except Exception as e:                                   # noqa: BLE001
    print("R gate: no result to read:", e)
    sys.exit(1)
v = m["val"]
s16, s8 = v["skill16"], v["skill8"]
ok_exit = rec.get("status") == "ok" and rec.get("exit") == 0
finite = all(math.isfinite(x) for x in (v["total"], s16, s8))
print(f"R gate: backbone {m['backbone']}, run status {rec.get('status')} exit {rec.get('exit')}, {m['steps']} steps, {m['seconds']:.0f} s, peak {m['peak_gpu_reserved_gb']:.2f} GB, params {m['params_m']:.2f} M")
print(f"R gate: skill16 {s16:.4f} (gate 0.695; B1 student 0.7153), skill8 {s8:.4f} (gate 0.652; B1 student 0.6721), total {v['total']:.4f} (B1 student 1.2006), cos16 {v['cos16']:.3f} cos8 {v['cos8']:.3f}")
passed = ok_exit and finite and s16 >= 0.695 and s8 >= 0.652
print("R gate:", "MET" if passed else "MISSED")
sys.exit(0 if passed else 1)
