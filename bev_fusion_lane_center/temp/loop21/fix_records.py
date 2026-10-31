import re
def sub(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:60], s.count(a))
        s = s.replace(a, b)
    open(path, "w", encoding="utf-8").write(s)
sub("experiments/E258_assembly-profile-and-speedup/experiment.md", [
 ("CPU, load average 3.5 - 4.2 during all timings)", "CPU, load average 3.5 - 4.2 during step 1b; step 2 ran at load 5.4 - 9.1, above the rule of no timing claim above the physical cores (6): its timing is provisional)"),
 ("the saving is relative and was measured on a machine at load 3.5 - 4.2", "the saving is relative and was measured on a machine at load 3.5 - 4.2 (step 1b; step 2 up to 9.1)"),
])
s = open("experiments/E259_assembly-cleanup-ablation/experiment.md", encoding="utf-8").read()
assert s.count("Q change +0.0016 mean") == 1
s = s.replace("Q change +0.0016 mean", "Q change +0.0015 mean (reviewer's recomputation)")
assert s.count("anchoring off saves 4.9 / 4.3 ms of 19.6 / 17.5") == 1
s = s.replace("anchoring off saves 4.9 / 4.3 ms of 19.6 / 17.5", "anchoring off saves about 5 / 4 ms of 19.6 / 17.5 (medians 5.24 / 4.29; provisional, load 4 - 15)")
open("experiments/E259_assembly-cleanup-ablation/experiment.md", "w", encoding="utf-8").write(s)
sub("models/registry.json", [("assembly about a fifth faster, relative timing)", "assembly faster by a relative, provisional amount: E258 / E259 timings were taken on a loaded machine)")])
print("ok")
