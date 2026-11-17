p = "experiments/E337_attached-end-geometry-gap/experiment.md"
s = open(p, encoding="utf-8").read()
a = ' "rule": [],\n'
assert s.count(a) == 1
s = s.replace(a, ' "rule": [{"metric": "topo_F1", "vs": "k", "max_loss": 0.003}],\n')
b = "`decide.py` is not used (no run, empty rule): the reading below is by hand and written here, not decided by a tool."
assert s.count(b) == 1
s = s.replace(b, "`decide.py` is not used (no run): the one clause states the gap (the PRIMARY read-out `a` loses more than 0.003 topo F1 to `k` on both seeds: it does, -0.0069 / -0.0062); the reading below is by hand and written here, not decided by a tool.")
open(p, "w", encoding="utf-8", newline="\n").write(s)
