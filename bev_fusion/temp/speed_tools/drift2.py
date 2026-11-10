"""throw-away: s/step every 300 steps of t17.data, t18.fix, t20.base_s0 (median of the logged rows in the window) and the clock time of the window"""
import json, statistics as st, datetime as dt
print("window      " + "   ".join(f"{r:>20s}" for r in ["t17_data", "t18_fix", "t20_base_s0"]))
data = {}
for run in ["t17_data", "t18_fix", "t20_base_s0"]:
    data[run] = [json.loads(l) for l in open(f"runs/{run}/log.jsonl")]
    data[run] = [r for r in data[run] if "s_per_step" in r]
for a in range(0, 6000, 300):
    cells = []
    for run, rows in data.items():
        b = [r for r in rows if a < r["step"] <= a + 300]
        s = st.median(r["s_per_step"] for r in b)
        clock = dt.datetime.fromtimestamp(b[len(b)//2]["t"]).strftime("%d %H:%M")
        cells.append(f"{s:5.2f} s  {clock}")
    print(f"{a+1:5d}-{a+300:5d}  " + "   ".join(f"{c:>20s}" for c in cells))
