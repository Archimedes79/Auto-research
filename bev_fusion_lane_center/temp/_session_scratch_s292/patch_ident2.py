p = "temp/loader/loader_identity.py"
s = open(p, encoding="utf-8").read()
a = '''    res = {}
    for name, w in (("w3a", 3), ("w3b", 3), ("w6", 6), ("w0", 0)):
        res[name] = run(w, a.seed, a.nb, not a.no_ppts, not a.no_scales)
        print(f"{name}: {a.nb} batches in {res[name][1]:.1f} s ({res[name][1] / a.nb:.2f} s per batch), fields {sorted(res[name][0][0])}", flush=True)
    for x, y in (("w3a", "w3b"), ("w3a", "w6"), ("w3a", "w0")):
        bad = compare(res[x][0], res[y][0])
        print(f"{x} against {y}: fields that differ (number of batches of {a.nb}): {bad if bad else 'NONE: every field of every batch is bit-identical'}", flush=True)
    rep = set(compare(res["w3a"][0], res["w3b"][0]))
    d6 = set(compare(res["w3a"][0], res["w6"][0]))
    print("\nnot reproducible even with equal workers (repeat of workers 3):", sorted(rep) or "none")
    print("differ between 3 and 6 workers but ARE reproducible with equal workers:", sorted(d6 - rep) or "none")
'''
assert s.count(a) == 1
b = '''    names = {"w3a": 3, "w3b": 3, "w4": 4, "w6": 6, "w0": 0}
    res = {}
    for name in a.runs.split(","):
        res[name] = run(names[name], a.seed, a.nb, not a.no_ppts, not a.no_scales)
        print(f"{name}: {a.nb} batches in {res[name][1]:.1f} s ({res[name][1] / a.nb:.2f} s per batch), fields {sorted(res[name][0][0])}", flush=True)
    for y in res:
        if y == "w3a" or "w3a" not in res:
            continue
        bad = compare(res["w3a"][0], res[y][0])
        print(f"w3a against {y}: fields that differ (number of batches of {a.nb}): {bad if bad else 'NONE: every field of every batch is bit-identical'}", flush=True)
    if "w3b" in res:
        print("\nnot reproducible even with equal workers (repeat of workers 3):", sorted(compare(res["w3a"][0], res["w3b"][0])) or "none")
'''
s = s.replace(a, b)
a2 = '    ap.add_argument("--no-scales", action="store_true")\n    a = ap.parse_args()'
assert s.count(a2) == 1
s = s.replace(a2, '    ap.add_argument("--no-scales", action="store_true")\n    ap.add_argument("--runs", default="w3a,w3b,w4,w6,w0", help="comma separated: w3a, w3b (repeat), w4, w6, w0")\n    a = ap.parse_args()')
open(p, "w", encoding="utf-8").write(s)
print("ok")
