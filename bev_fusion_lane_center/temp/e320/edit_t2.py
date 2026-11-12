p = "temp/e320/t_real.py"
s = open(p, encoding="utf-8").read()
old = '''    base = dict(run="e320_smoke_cpu_s0", seed=0, thr=0.1, full=f8, base=None, skip_base_check=True, chord=False, snap=False)
    for kw in (dict(), dict(attach=True), dict(snap=True)):'''
new = '''    base = dict(run="e320_smoke_cpu_s0", seed=0, thr=float(os.environ.get("THR", 0.1)), full=f8, base=None, skip_base_check=True, chord=False, snap=False)
    for kw in ((dict(attach=True),) if os.environ.get("THR") else (dict(), dict(attach=True), dict(snap=True))):'''
assert old in s
s = s.replace(old, new)
old2 = '''elif mode == "flops":'''
new2 = '''elif mode == "kperr_att":
    from importlib.machinery import SourceFileLoader
    import importlib.util
    ld = SourceFileLoader("jdiag", os.path.join(HERE, "dev", "jdiag.pysrc"))
    jd = importlib.util.module_from_spec(importlib.util.spec_from_loader("jdiag", ld))
    ld.exec_module(jd)
    jd.kperr(["e320_smoke_cpu_s0"])
elif mode == "train_smoke":
    os.environ["BFLC_SIG"] = "1.0"
    M = use("dev")
    sys.argv = ["jends_train.py", "--name", "e320_smoke_train_cpu_s0", "--seed", "0", "--maps", os.path.join(ROOT, "runs", "e307_n_s0", "model.pt"), "--e2e", "--scales", "--maps-w", "1.0", "--geom-w", "0.5", "--attach", "--bs", "2", "--workers", "0", "--smoke"]
    M["jends_train"].main()
elif mode == "flops":'''
assert s.count(old2) == 1
s = s.replace(old2, new2)
open(p, "w", encoding="utf-8").write(s)
