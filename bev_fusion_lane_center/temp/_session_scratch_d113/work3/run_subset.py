import importlib.util, os, sys, tempfile
testfile, names = sys.argv[1], sys.argv[2:]
spec = importlib.util.spec_from_file_location("tmod", testfile)
t = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(t)
sel = [(k, m) for k, m in enumerate(t.MUTANTS) if any(n in m[0] if testfile.endswith("test_lane_map.py") or testfile.endswith("test_jmap.py") else n in m[1] for n in names)]
print(len(sel), "mutants selected")
with tempfile.TemporaryDirectory() as tmp:
    for k, m in sel:
        print(t.one((k, m, tmp)), flush=True)
