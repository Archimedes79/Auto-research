"""scratch: validation loss against training loss by step for the long runs (loss = lower is better)"""
import json, os
R = "/mnt/c/Auto-research/bev_fusion/runs"
for run in ["t1_r0", "t1_r4_soa", "t3_ref_long", "t16_full_s0", "t17_data", "t18_fix", "t20_base_s0", "t22_A2_s0"]:
    p = f"{R}/{run}/log.jsonl"
    if not os.path.exists(p):
        continue
    tr, va = {}, []
    for ln in open(p):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "val" in r:
            va.append((r["step"], r["val"]))
        elif "total" in r or "group.areas" in r:
            tr[r["step"]] = r
    print("==", run)
    steps = sorted(tr)
    for s, v in va:
        near = [tr[k] for k in steps if s - 200 < k <= s]
        def m(key):
            xs = [x[key] for x in near if key in x]
            return sum(xs) / len(xs) if xs else float("nan")
        print(f"  step {s:5d}  val total {v.get('total', float('nan')):7.3f} (train {m('total'):7.3f})  areas {v.get('group.areas', float('nan')):.3f} ({m('group.areas'):.3f})  lines {v.get('group.lines', float('nan')):.3f} ({m('group.lines'):.3f})  objects {v.get('group.objects', float('nan')):.3f} ({m('group.objects'):.3f})  motion {v.get('group.motion', float('nan')):.3f} ({m('group.motion'):.3f})")
