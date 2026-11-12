p = "temp/e320/t_real.py"
s = open(p, encoding="utf-8").read()
old = '''    ev = importlib.util.module_from_spec(importlib.util.spec_from_loader("jends_eval", SourceFileLoader("jends_eval", os.path.join(HERE, "dev", "jends_eval.pysrc"))))
    ev.__loader__.exec_module(ev) if hasattr(ev.__loader__, "exec_module") else None
'''
new = '''    def modsrc(name):
        ld = SourceFileLoader(name, os.path.join(HERE, "dev", name + ".pysrc"))
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
        ld.exec_module(mod)
        return mod
    ev = modsrc("jends_eval")
'''
assert old in s
s = s.replace(old, new)
s = s.replace('''    jm = SourceFileLoader("jmaneuver", os.path.join(HERE, "dev", "jmaneuver.pysrc")).load_module()
''', '''    jm = modsrc("jmaneuver")
''')
open(p, "w", encoding="utf-8").write(s)
