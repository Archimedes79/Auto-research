"""One-off (cycle 19): registry entries of the E240 / E237 seed arms (new names; champion / control are not touched, so the gate hook lets it pass).
    wsl python3 temp/loop19/add_registry_e240.py
The file keeps its hand-formatted one-entry-per-block layout: entries are inserted as text before "control"."""
import json
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "registry.json")
text = open(p, encoding="utf-8").read()
reg = json.loads(text)
ASM = '{"cov_r": 0.5, "vtx_r": 1.5, "junc_thr": 0.5, "reloc_thr": 0.4, "reloc_shift": 12, "reloc_tol": 1.6, "fork_prune": 0.4, "fast": 2}'
runs = {"e24_j1rc_s0": "e18_D7_j1rc", "e24_j1rc_s1": "e18_D7_j1rcs1", "e24_j1rc_s2": "e24_D7_j1rcs2", "e24_j1rc_s3": "e24_D7_j1rcs3",
        "e24_j1eq_s0": "e18_D7_j1eq", "e24_j1eq_s1": "e18_D7_j1eqs1", "e24_j1eq_s2": "e24_D7_j1eqs2", "e24_j1eq_s3": "e24_D7_j1eqs3",
        "e24_j1ex_s0": "e24_D7_j1exs0", "e24_j1ex_s1": "e24_D7_j1exs1"}
what = {"j1rc": "E240 arm a (recall weight 3), seed %s", "j1eq": "E240 arm c (equal-budget control, recall weight 1), seed %s", "j1ex": "E237 arm a (j1eq recipe on the CPU-exact distance map), seed %s"}
add = ""
for name, run in runs.items():
    if name in reg:
        continue
    arm, seed = name.split("_")[1], name[-1]
    add += (f'  "{name}": {{"kind": "builder", "design": "D7", "run": "{run}", "set": {{"M": 192, "K0": 96}}, "tau": 0.7,\n'
            f'               "asm": {ASM},\n               "what": "{what[arm] % seed}"}},\n')
if add:
    i = text.index('  "control":')
    text = text[:i] + add + text[i:]
    json.loads(text)
    open(p, "w", encoding="utf-8").write(text)
print("added", add.count('"kind"'), "entries")
