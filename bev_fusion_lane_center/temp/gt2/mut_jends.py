import os, sys, types
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
src = open(os.path.join(ROOT, "training/grid/jends.py"), encoding="utf-8").read()
muts = {
    "swap degree sums": ("for deg, mt in ((T.sum(2), match_e), (T.sum(1), match_x)):", "for deg, mt in ((T.sum(1), match_e), (T.sum(2), match_x)):"),
    "swap dustbin masses": ("log_a = torch.cat([(re + 1e-6).log(), (rx.sum(1, keepdim=True) + 1e-6).log()], 1)", "log_a = torch.cat([(re + 1e-6).log(), (re.sum(1, keepdim=True) + 1e-6).log()], 1)"),
}
for name, (a, b) in muts.items():
    assert a in src, name
    mod = types.ModuleType("jends_mut")
    mod.__dict__["__name__"] = "jends_mut"
    exec(compile(src.replace(a, b), "jends_mut.py", "exec"), mod.__dict__)
    try:
        mod._selftest()
        print("MUTATION SURVIVED:", name)
    except AssertionError as e:
        print("mutation caught:", name)
