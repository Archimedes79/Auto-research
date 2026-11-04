"""throw-away: remove the five table rows of the deleted held runs from records/LEDGER.md (a generated file; the next `ledger.py render` rebuilds it from the csv)"""
ids = ["t16.ctl_s0", "t16.ctl_s0.eval", "speed.d", "speed.e", "speed.f"]
p = "records/LEDGER.md"
raw = open(p, encoding="utf-8", newline="").read()
lines = raw.split("\n")
keep = [ln for ln in lines if not any(ln.startswith(f"| {i} |") for i in ids)]
print("removed", len(lines) - len(keep), "lines")
open(p, "w", encoding="utf-8", newline="").write("\n".join(keep))
