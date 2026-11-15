import re
p = "evaluation/jwire_verdict.py"
s = open(p, encoding="utf-8").read()
def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old, s.count(old))
    s = s.replace(old, new)
rep("import re\nimport sys\nimport tempfile\n", "import re\nimport tempfile\n")
rep('    bp = a.boot.format(seed=s) if os.path.isabs(a.boot.format(seed=s)) else f"{a.root}/{a.boot.format(seed=s)}"\n', '    bp = os.path.join(a.root, a.boot.format(seed=s))\n')
rep("    global ROOT\n    old = ROOT\n\n    def agree", "    def agree")
rep("    ROOT = old\n    print(\"jwire_verdict selftest ok\")", "    print(\"jwire_verdict selftest ok\")")
rep("return dict(dict(seed=0, pilot_train=", "return dict(dict(pilot_train=")
rep('''assert run_case({0: good, 1: dict(good, wa=agree(420, pair_run="cs0"))})[0] == "INVALID" and run_case({0: good, 1: dict(good, wa=agree(420, pair_run="cs1"))})[0] == "KEEP+"''',
    '''assert run_case({0: good, 1: dict(good, wa=agree(420, pair_run="cs0", seed=1))})[0] == "INVALID" and run_case({0: good, 1: dict(good, wa=agree(420, pair_run="cs1", seed=0))})[0] == "INVALID"
    assert run_case({0: good, 1: dict(good, wa=agree(420, pair_run="cs1", seed=1))})[0] == "KEEP+"''')
rep('''and sum("arm x (tx" in l for l in lines) == 2 and not any("no scored card" in l for l in lines), lines''', '''and sum("arm x (tx" in l for l in lines) == 2 and not any("no scored card" in l for l in lines if "arm x" in l), lines''')
open(p, "w", encoding="utf-8").write(s)
