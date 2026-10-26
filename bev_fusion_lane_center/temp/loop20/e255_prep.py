"""E255: registry entry of the full-schedule Poisson arm (inserted as text before the "control" entry, like the loop-19 one-off scripts). The tau / vtx_r of the entry are
placeholders (0.2 / 1.5); the card uses the matched-vertex procedure of the card (temp/loop20/e255_matched.py) and writes val_m143.json."""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
p = f"{R}/models/registry.json"
t = open(p, encoding="utf-8").read()
ASM = '{"cov_r": 0.5, "vtx_r": 1.5, "junc_thr": 0.5, "reloc_thr": 0.4, "reloc_shift": 12, "reloc_tol": 1.6, "fork_prune": 0.4, "fast": 2}'
add = ""
for s in (0, 1):
    n = f"e25_j1phd2_s{s}"
    if f'"{n}"' not in t:
        add += (f'  "{n}": {{"kind": "builder", "design": "D7", "run": "e25_D7_j1phd2s{s}", "set": {{"M": 192, "K0": 96}}, "tau": 0.2,\n'
                f'               "asm": {ASM},\n               "what": "E255: j1 4 u fine-tune with the Poisson set likelihood (phd 0.65, rec 0) on the exact distance map, seed {s}; tau / vtx_r are placeholders, see the card"}},\n')
if add:
    i = t.index('  "control":')
    t = t[:i] + add + t[i:]
    json.loads(t)
    open(p, "w", encoding="utf-8").write(t)
print("ok", bool(add))
