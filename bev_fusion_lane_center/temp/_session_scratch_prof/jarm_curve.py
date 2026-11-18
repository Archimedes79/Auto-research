p = "tools/jarm.py"
t = open(p, encoding="utf-8").read()
# curve check
a = "class FakeArm(Arm):"
assert t.count(a) == 1
t = t.replace(a, '''BAND = {500: 0.05, 2000: 0.03, 12000: 0.02}


def curve(run, ref, band=BAND):
    """the card's loss-curve kill band of a speed-change arm: the logged `total` of runs/<run>/log.jsonl against the same step of runs/<ref> at the steps of `band` -> (ok, text); a missing step or file is not ok"""
    def load(r):
        f = os.path.join(RUNS, r, "log.jsonl")
        return {j["step"]: j["total"] for j in (json.loads(l) for l in open(f)) if "total" in j} if os.path.exists(f) else {}
    a, b = load(run), load(ref)
    rows = {s: (a.get(s), b.get(s)) for s in band}
    ok = all(x is not None and y is not None and abs(x - y) <= band[s] for s, (x, y) in rows.items())
    return ok, f"{run} vs {ref}: " + ", ".join(f"step {s}: " + ("missing" if x is None or y is None else f"{x:.4f} vs {y:.4f} (diff {x - y:+.4f}, band {band[s]})") for s, (x, y) in rows.items()) + (" -> inside the band" if ok else " -> OUTSIDE the band or missing")


''' + a)
# parser + main
a = '''    sub.add_parser("selftest")
    return ap'''
assert t.count(a) == 1
t = t.replace(a, '''    c = sub.add_parser("curve")
    c.add_argument("run")
    c.add_argument("ref")
    sub.add_parser("selftest")
    return ap''')
a = '''    if a.cmd == "table":
        return table(a.runs, a.cell)'''
assert t.count(a) == 1
t = t.replace(a, a + '''
    if a.cmd == "curve":
        ok, text = curve(a.run, a.ref)
        print(text)
        sys.exit(0 if ok else 1)''')
# docstring line
a = '''    wsl python3 tools/jarm.py selftest
'''
assert t.count(a) == 1
t = t.replace(a, '''    wsl python3 tools/jarm.py curve e383_f_s0 e373_cb_s0                                                           # a speed-change arm's loss curve against its reference's (the card's kill band 0.05 / 0.03 / 0.02 at steps 500 / 2000 / 12000); exit 1 outside
    wsl python3 tools/jarm.py selftest
''')
# selftest
a = '''            f = FakeArm("E9", "d", 0, 2, fail=("",))
            assert f.train() is False and f.log == [""] * 6, f.log'''
assert t.count(a) == 1
t = t.replace(a, a + '''
            for r, rows in (("cA", {500: 0.70, 2000: 0.48, 12000: 0.36}), ("cB", {500: 0.72, 2000: 0.49, 12000: 0.37}), ("cC", {500: 0.70, 2000: 0.48, 12000: 0.40})):
                os.makedirs(os.path.join(d, r))
                open(os.path.join(d, r, "log.jsonl"), "w").write("".join(json.dumps(dict(step=s, total=v)) + "\n" for s, v in rows.items()) + json.dumps(dict(step=7, val={})) + "\n")
            assert curve("cB", "cA")[0] and not curve("cC", "cA")[0] and not curve("nope", "cA")[0], "the band: 0.02 at the last step, a missing run is not ok"''')
open(p, "w", encoding="utf-8").write(t)
print("ok")
