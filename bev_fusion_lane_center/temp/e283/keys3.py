import json
c = json.load(open("runs/e279/e279_w_s0.plain_t0.5/card.json"))
print("g:", json.dumps(c["g"])[:400])
print("assign_error:", c["p"]["assign_error"])
t = json.load(open("runs/e273/e273_p_s0/timing.json")) if __import__("os").path.exists("runs/e273/e273_p_s0/timing.json") else None
print("timing:", json.dumps(t)[:500] if t else None)
a = json.load(open("temp/e276/assignment_uncertainty_e273_p_s0.json"))
lk = a["result"]["links"]
print("links keys:", list(lk.keys())[:12]); print(lk["p>=0.5 long"], [v for v in lk["risk_coverage"] if v["threshold"] == 0.5])
