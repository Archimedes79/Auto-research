"""E244.a0 was cut at iteration 3200 of 4800 (no final model.pt, no card): make its 50 % checkpoint evaluable for a labelled partial reading.

    wsl python3 temp/loop19/e244_prep_at50.py

Run folder runs/e123/e24_D7_j1phds0_at50/model.pt (copy of model_at50.pt), config card runs/e123_ladder/cards/e24_D7_j1phds0_at50.json (copy of the card of the
equal-budget j1eq seed 2: the same builder configuration, the loss weights are not part of the inference config), registry entry e24_j1phd_s0_at50 (fork pruning 0.4,
tau 0.5 as declared for arm a). Compare with e24_j1rc_s2_at50 / e24_j1eq_s2_at50 (E241, tau 0.7): same point of the same schedule, other seed.
"""
import json
import os
import shutil

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
reg_p = f"{R}/models/registry.json"
text = open(reg_p, encoding="utf-8").read()
src = f"{R}/runs/e123/e24_D7_j1phds0/model_at50.pt"
dst = f"{R}/runs/e123/e24_D7_j1phds0_at50"
os.makedirs(dst, exist_ok=True)
shutil.copy2(src, f"{dst}/model.pt")
shutil.copy2(f"{R}/runs/e123_ladder/cards/e24_D7_j1eqs2.json", f"{R}/runs/e123_ladder/cards/e24_D7_j1phds0_at50.json")
name = "e24_j1phd_s0_at50"
if f'"{name}"' not in text:
    asm = '{"cov_r": 0.5, "vtx_r": 1.5, "junc_thr": 0.5, "reloc_thr": 0.4, "reloc_shift": 12, "reloc_tol": 1.6, "fork_prune": 0.4, "fast": 2}'
    add = (f'  "{name}": {{"kind": "builder", "design": "D7", "run": "e24_D7_j1phds0_at50", "set": {{"M": 192, "K0": 96}}, "tau": 0.5,\n'
           f'               "asm": {asm},\n               "what": "E244 arm a seed 0 after 50 % of the 4 u fine-tune (run cut at 3200 / 4800 iterations)"}},\n')
    i = text.index('  "control":')
    text = text[:i] + add + text[i:]
    json.loads(text)
    open(reg_p, "w", encoding="utf-8").write(text)
print(name)
