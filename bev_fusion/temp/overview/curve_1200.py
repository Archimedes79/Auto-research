"""scratch: what changes around step 1200 in the training log (loss groups, lr, learned balancing weights, warm / cold, grad norm), several runs"""
import json, os, sys
R = "/mnt/c/Auto-research/bev_fusion/runs"
runs = sys.argv[1:] or ["t20_base_s0", "t16_full_s0", "t1_r4_soa"]
for run in runs:
    rows = []
    for ln in open(f"{R}/{run}/log.jsonl"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "val" not in r and "step" in r:
            rows.append(r)
    keys = [k for k in rows[len(rows) // 2] if k.startswith("s.") or k.startswith("w.") or k.startswith("loss.s") or k.endswith(".s")]
    print("==", run, "rows", len(rows), "s-like keys:", keys[:10])
    cols = ["total", "group.areas", "group.lines", "group.objects", "group.motion", "total_last_warm", "total_last_cold", "grad_norm", "lr"] + keys[:5]
    print("step-window " + " ".join(f"{c[:15]:>15s}" for c in cols))
    for a in range(0, 3400, 200):
        w = [r for r in rows if a <= r["step"] < a + 200]
        if not w:
            continue
        def m(c):
            xs = [x[c] for x in w if c in x and isinstance(x[c], (int, float))]
            return sum(xs) / len(xs) if xs else float("nan")
        print(f"{a:5d}-{a+200:5d} " + " ".join(f"{m(c):15.4g}" for c in cols))
