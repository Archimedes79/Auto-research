"""One-off: summary of a short run's log.jsonl: speed, GPU memory, group losses, gradient norms, validation terms."""
import json
import sys

for l in open(sys.argv[1]):
    r = json.loads(l)
    if "val" in r:
        print("VAL", {k: round(v, 3) for k, v in r["val"].items()})
        continue
    if r["step"] in (4, 12, 24) or "diag" in r:
        print(r["step"], {k: round(v, 3) for k, v in r.items() if isinstance(v, (int, float)) and k in ("grad_norm", "s_per_step", "gpu_gb", "total", "group.areas", "group.lines", "group.objects", "group.motion")})
        if "diag" in r:
            print("  norms", dict(zip(r["diag"]["groups"], [f"{x:.1e}" for x in r["diag"]["norm"]])))
