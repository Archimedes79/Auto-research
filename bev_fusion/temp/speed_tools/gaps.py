import json, sys
# seconds per step between the logged rows of runs (read-only), with the probe columns when present
for name in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"/mnt/c/Auto-research/bev_fusion/runs/{name}/log.jsonl") if l.strip()]
    rows = [r for r in rows if "t" in r and "val" not in r]
    print(name, "rows", len(rows))
    for a, b in zip(rows, rows[1:]):
        extra = " ".join(f"{k}={b[k]:.2f}" for k in ("t_data", "t_step", "gpu_gb") if k in b)
        print(f"  step {b['step']:>3}  {(b['t'] - a['t']) / (b['step'] - a['step']):6.2f} s/step  {extra}")
