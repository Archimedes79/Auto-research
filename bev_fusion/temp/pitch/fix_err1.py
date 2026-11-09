p = "evaluation/bev_errors.py"
s = open(p, encoding="utf-8").read()
old = '''    sc = Counter(_bin(r["score"], SCORE_BINS) if False else next((f"{lo:g}-{min(hi, 1):g}" for lo, hi in SCORE_BINS if lo <= r["score"] < hi), "<0.3") for r in rows if r["what"] == "FP")'''
new = '''    sc = Counter(next((f"{lo:g}-{min(hi, 1):g}" for lo, hi in SCORE_BINS if lo <= r["score"] < hi), "<0.3") for r in rows if r["what"] == "FP")'''
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new))
