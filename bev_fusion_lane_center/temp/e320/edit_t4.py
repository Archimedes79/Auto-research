p = "temp/e320/t_real.py"
s = open(p, encoding="utf-8").read()
old = 'elif mode == "flops":'
new = '''elif mode == "selftest_dev":
    from importlib.machinery import SourceFileLoader
    import importlib.util
    M = use("dev")

    def modsrc(name):
        ld = SourceFileLoader(name, os.path.join(HERE, "dev", name + ".pysrc"))
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
        sys.modules[name] = mod
        ld.exec_module(mod)
        return mod
    for n in ("qends_eval", "jmaneuver", "jends_eval", "jdiag"):
        mod = modsrc(n)
        if n == "jdiag":
            mod.selftest()
        elif n == "jends_eval":
            mod.selftest()
        else:
            mod.selftest()
        print(n, "selftest ok")
elif mode == "flops":'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
