p = "models/registry.json"
t = open(p, encoding="utf-8").read()
key = '  "e24_j1rc_s0": {"kind": "builder"'
assert t.count(key) == 1 and "  \"recall_dp\": " not in t
new = ('  "recall_dp": {"kind": "builder", "design": "D7", "run": "e123_D7_r3_rc1", "set": {"M": 288, "K0": 144}, "tau": 0.7,\n'
       '               "asm": {"cov_r": 0.5, "vtx_r": 1.5, "junc_thr": 0.5, "reloc_thr": 0.4, "reloc_shift": 12, "reloc_tol": 1.6, "fork_prune": 0.4, "fast": 2, "dp_eps": 0.2},\n'
       '               "what": "recall (rc1, fork pruning 0.4) with the output stage simplify_dp 0.2 m (D20)"},\n')
open(p, "w", encoding="utf-8").write(t.replace(key, new + key, 1))
