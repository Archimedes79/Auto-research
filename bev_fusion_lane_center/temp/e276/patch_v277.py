p = "evaluation/qgraph_run_eval.py"
s = open(p, encoding="utf-8").read()
s = s.replace("Otherwise the verdict of verdict275 (CUT /\n    TIE / INCONCLUSIVE).", "Otherwise the verdict of verdict275 (CUT / INCONCLUSIVE); a verdict275 TIE (both\n    headlines inside their bands, no effect) is reported as INCONCLUSIVE, as the card says: TIE here means \"helps but trails the hard step\".")
open(p, "w", encoding="utf-8").write(s)
t = "evaluation/tests/test_verdict277.py"
s = open(t, encoding="utf-8").read()
s = s.replace('''    for n, c in zip(names_c, ctl):
        T5.put(root, "ec", n, c)
    for n, c in zip(names_h, hard):
        T5.put(root, "eh", n, c)''', '''    for k, (n, c) in enumerate(zip(names_c, ctl)):
        T5.put(root, "ec", n, dict(c, run=runs[k] if runs else f"m{k}"))
    for k, (n, c) in enumerate(zip(names_h, hard)):
        T5.put(root, "eh", n, dict(c, run=hard_runs[k] if hard_runs else f"m{k}"))''')
s = s.replace("def run(root, soft, ctl, hard, **kw):", "def run(root, soft, ctl, hard, runs=None, hard_runs=None, **kw):")
s = s.replace("def refused(root, soft, ctl, hard, **kw):\n    try:\n        run(root, soft, ctl, hard, **kw)", "def refused(root, soft, ctl, hard, **kw):\n    try:\n        run(root, soft, ctl, hard, **kw)")
s = s.replace('''        # a broken assignment guard is a CUT whatever the topo numbers''', '''        # no effect over the baseline in BOTH headlines (verdict275 says TIE): the card says INCONCLUSIVE, and the word TIE stays for "helps but trails the hard step" (critic pass 2026-11-08)
        r = run(root, soft(0.605, 0.605, junc=0.13), ctl, hard)
        assert r["verdict275"] == "TIE" and r["verdict"].startswith("INCONCLUSIVE"), r
        # a broken assignment guard is a CUT whatever the topo numbers''')
s = s.replace('''        assert refused(root, [T5.var(0, readout="close1.5_t0.5"''', '''        assert refused(root, soft(0.62, 0.62), ctl, hard, runs=["m0", "m1"], hard_runs=["m0", "other"]), "the hard-step cards must be the control's models"
        assert refused(root, [T5.var(0, readout="close1.5_t0.5"''')
open(t, "w", encoding="utf-8").write(s)
print("ok")
