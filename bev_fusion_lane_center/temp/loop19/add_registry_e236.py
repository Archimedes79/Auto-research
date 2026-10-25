"""One-off (cycle 19, E236): registry entry 'recall_s1' (the seed twin rc1s1 of the recall variant). Run: wsl python3 temp/loop19/add_registry_e236.py"""
import json, os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "registry.json")
t = open(p, encoding="utf-8").read()
if '"recall_s1"' not in t:
    asm = '{"cov_r": 0.5, "vtx_r": 1.5, "junc_thr": 0.5, "reloc_thr": 0.4, "reloc_shift": 12, "reloc_tol": 1.6, "fork_prune": 0.4, "fast": 2}'
    e = (f'  "recall_s1": {{"kind": "builder", "design": "D7", "run": "e18_D7_rc1s1", "set": {{"M": 288, "K0": 144}}, "tau": 0.7,\n'
         f'               "asm": {asm},\n               "what": "seed twin of the recall variant rc1 (E219.rc1s1), fork pruning 0.4"}},\n')
    i = t.index('  "control":')
    t = t[:i] + e + t[i:]
    json.loads(t)
    open(p, "w", encoding="utf-8").write(t)
print("ok")
