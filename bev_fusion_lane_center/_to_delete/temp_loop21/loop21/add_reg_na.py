p = "models/registry.json"
t = open(p, encoding="utf-8").read()
key = '  "recall": {"kind": "builder"'
assert key in t and "fast_recall_na" not in t
new = ('  "fast_recall_na": {"kind": "builder", "design": "D7", "run": "e18_D7_j1rc", "set": {"M": 192, "K0": 96}, "tau": 0.7,\n'
       '               "asm": {"cov_r": 0.5, "vtx_r": 1.5, "junc_thr": 0.0, "reloc_thr": 0.4, "reloc_shift": 12, "reloc_tol": 1.6, "fork_prune": 0.4, "fast": 2},\n'
       '               "what": "fast_recall with the junction anchoring switched off (E258 step 2: Q and J within the guards on four seeds, assembly about a fifth faster, relative timing)"},\n')
t = t.replace(key, new + key, 1)
open(p, "w", encoding="utf-8").write(t)
