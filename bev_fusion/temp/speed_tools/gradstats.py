import json, statistics
for name in ("speed_a3", "speed_a", "speed_b", "speed_c2"):
    rows = [json.loads(x) for x in open(f"/mnt/c/Auto-research/bev_fusion/runs/{name}/log.jsonl") if x.strip()]
    rows = [r for r in rows if "grad_norm" in r]
    g = [r["grad_norm"] for r in rows]
    tot = [r["total"] for r in rows if "total" in r]
    print(f"{name:9} grad_norm median {statistics.median(g):8.1f}  max {max(g):9.1f}   total: first {tot[0]:.2f}  mean(last 6) {statistics.mean(tot[-6:]):.2f}  min {min(tot):.2f} max {max(tot):.2f}")
    if "t_data" in rows[-1]:
        print("          last row keys:", sorted(k for k in rows[-1] if k.startswith(("t_", "gpu", "s_per"))))
