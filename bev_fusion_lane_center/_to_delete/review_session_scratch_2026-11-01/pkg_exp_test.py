p = "autoresearch/tools/package.py"
s = open(p, encoding="utf-8").read()
old = '''        assert experiment_of("x.zzz", "x_zzz") == V.UNKNOWN'''
new = '''        os.rename(os.path.join(proj, "experiments", "E1_demo"), os.path.join(proj, "experiments", "E9_demo"))
        assert experiment_of("x.a_s0", "x_a_s0") == "experiments/E9_demo/experiment.md" and experiment_of("x.a_s01", "x_a_s01") == "experiments/E2_other/experiment.md", "an id is not found inside a longer id"
        os.rename(os.path.join(proj, "experiments", "E9_demo"), os.path.join(proj, "experiments", "E1_demo"))
        assert experiment_of("x.zzz", "x_zzz") == V.UNKNOWN'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("ok")
