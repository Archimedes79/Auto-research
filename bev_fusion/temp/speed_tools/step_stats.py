"""throw-away: data wait vs step time in the logs of the full-network runs, per 1000-step block (median of the logged rows)"""
import json, statistics as st
for run in ["t16_full_s0", "t17_data", "t18_fix", "t20_base_s0"]:
    rows = [json.loads(l) for l in open(f"runs/{run}/log.jsonl")]
    tr = [r for r in rows if "s_per_step" in r and "t_data" in r]
    print(run, "rows with t_data:", len(tr), " sample row t_data/t_step/s_per_step:", tr[len(tr)//2]["t_data"], tr[len(tr)//2]["t_step"], round(tr[len(tr)//2]["s_per_step"], 2))
    for a in range(0, 6000, 1000):
        b = [r for r in tr if a < r["step"] <= a + 1000]
        if not b: continue
        s = st.median(r["s_per_step"] for r in b); d = st.median(r["t_data"] for r in b); t = st.median(r["t_step"] for r in b)
        print(f"   steps {a+1:5d}-{a+1000:5d}  s/step {s:5.2f}  t_data {d:5.2f}  t_step {t:5.2f}  data share {100*d/s:4.1f} %  max s/step {max(r['s_per_step'] for r in b):5.1f}")
