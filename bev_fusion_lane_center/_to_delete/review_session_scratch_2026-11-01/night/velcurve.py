import json, sys
run = sys.argv[1] if len(sys.argv) > 1 else "t11_vel_hurdle"
rows = [json.loads(l) for l in open(f"/mnt/c/Auto-research/bev_fusion/runs/{run}/log.jsonl")]
for r in rows:
    if r.get("step", 0) % 100 == 0 or any(k.startswith("val") for k in r):
        print(r.get("step"), {k: round(v, 3) for k, v in r.items()
                              if any(s in k for s in ("vel", "mov", "motion")) and isinstance(v, float)})
