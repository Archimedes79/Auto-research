"""throw-away: every mutant of mutate_host.py must find its anchor exactly once in the real file (run from bev_fusion/)"""
src = open("temp/speed_tools/mutate_host.py", encoding="utf-8").read().split("survived = 0")[0]
ns = {}
exec(src, ns)
bad = []
for name, path, old, new, test in ns["MUTANTS"]:
    if path:
        n = open(path, encoding="utf-8").read().count(old)
        if n != 1:
            bad.append((name, n))
print("mutants", len(ns["MUTANTS"]), "| anchors not found exactly once:", bad)
