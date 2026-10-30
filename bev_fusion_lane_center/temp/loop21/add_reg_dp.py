p = "models/registry.json"
t = open(p, encoding="utf-8").read()
key = '  "recall": {"kind": "builder"'
assert t.count(key) == 1 and "fast_recall_dp" not in t
new = ('  "fast_recall_dp": {"kind": "builder", "design": "D7", "run": "e18_D7_j1rc", "set": {"M": 192, "K0": 96}, "tau": 0.7,\n'
       '               "asm": {"cov_r": 0.5, "vtx_r": 1.5, "junc_thr": 0.5, "reloc_thr": 0.4, "reloc_shift": 12, "reloc_tol": 1.6, "fork_prune": 0.4, "fast": 2, "dp_eps": 0.2},\n'
       '               "what": "fast_recall with the output stage simplify_dp 0.2 m (E257, D20): about 9.9 vertices per 100 m instead of 15.0, Q -0.0003"},\n'
       '  "champion_dp": {"kind": "builder", "design": "D7", "run": "e123_D7_r3_f1", "set": {"M": 288, "K0": 144}, "tau": 0.7,\n'
       '               "asm": {"cov_r": 0.5, "vtx_r": 1.5, "junc_thr": 0.5, "reloc_thr": 0.4, "reloc_shift": 12, "reloc_tol": 1.6, "fork_prune": 0.3, "fast": 2, "dp_eps": 0.2},\n'
       '               "what": "champion with the output stage simplify_dp 0.2 m (E257, D20): about 9.5 vertices per 100 m instead of 14.8, Q -0.0005"},\n')
open(p, "w", encoding="utf-8").write(t.replace(key, new + key, 1))
