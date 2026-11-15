R = "/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/"
p = R + "jprune_probe.py"
s = open(p, encoding="utf-8").read()
a = s.index('        assert self.rec["rows"] == self.rows, "the sweep.json on disk is of other frames"')
b = s.index('        self.rec["deterministic"] = deterministic')
s = s[:a] + '        check_resume(self.rec, self.rows, deterministic)\n' + s[b:]
old = "class Sweep:"
new = '''def check_resume(rec, rows, deterministic):
    """a sweep.json on disk may be resumed only on the same frames and, once it holds passes, with the same deterministic setting"""
    assert rec["rows"] == rows, "the sweep.json on disk is of other frames"
    assert bool(rec.get("deterministic", False)) == deterministic or not rec["passes"], f"sweep.json was made with deterministic={rec.get('deterministic', False)}, this job runs with {deterministic}: the passes must share one setting"


class Sweep:'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("ok")
