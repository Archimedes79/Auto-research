p = "temp/e320/t_real.py"
s = open(p, encoding="utf-8").read()
old = 'elif mode == "flops":'
new = '''elif mode == "boot_test":
    # the paired bootstrap of jdiag on the counts that the dev cmd_score wrote for two read-outs of the smoke dump (8 frames, few scenes: a pipeline check)
    from importlib.machinery import SourceFileLoader
    import importlib.util
    ld = SourceFileLoader("jdiag", os.path.join(HERE, "dev", "jdiag.pysrc"))
    jd = importlib.util.module_from_spec(importlib.util.spec_from_loader("jdiag", ld))
    ld.exec_module(jd)
    jd.boot("e320_smoke_cpu_s0", "ends_card_snap_nogeom_attach_thr0.1", "e320_smoke_cpu_s0", "ends_card_attach_thr0.1", n=30)
elif mode == "flops":'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
