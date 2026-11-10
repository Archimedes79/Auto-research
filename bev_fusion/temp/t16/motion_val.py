"""scratch: the in-training validation of the motion term at steps 2000 / 4000 / 6000 for t17_data and t18_fix (kill criterion of the t18 card: step 2000 above the control by more than 50 %)"""
import json

for r in ("t17_data", "t18_fix"):
    rows = [json.loads(l) for l in open(f"bev_fusion/runs/{r}/log.jsonl", encoding="utf-8") if l.strip()]
    print("==", r, len(rows), "rows")
    for x in rows:
        if x.get("step") in (2000, 4000, 6000):
            ks = [k for k in x if "motion" in k.lower()]
            print(x.get("step"), {k: (round(x[k], 4) if isinstance(x[k], float) else x[k]) for k in ks[:6]}, "| keys sample:", list(x)[:12])
