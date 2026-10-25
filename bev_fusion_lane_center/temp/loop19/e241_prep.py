"""E241 (early-kill replay): make the intermediate weights of the E240 fine-tunes evaluable and register them.

    wsl python3 temp/loop19/e241_prep.py 2 3        (seeds whose model_at25/50/75.pt exist)

For every run e24_D7_j1{rc,eq}s<seed> and percent p in 25, 50, 75 with runs/e123/<run>/model_at<p>.pt: run folder runs/e123/<run>_at<p>/model.pt (copy) and the
config card runs/e123_ladder/cards/<run>_at<p>.json (copy of the run's card), and the registry entry e24_j1{rc,eq}_s<seed>_at<p> (fork pruning 0.4, tau 0.7, as
the final evaluation). Then evaluate with `evaluate.py <name>` on VAL 873 (GPU for the raw outputs, once).
Throw-away helper; the registry is inserted as text before the "control" entry like the other one-off scripts.
"""
import json
import os
import shutil
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
reg_p = f"{R}/models/registry.json"
text = open(reg_p, encoding="utf-8").read()
ASM = '{"cov_r": 0.5, "vtx_r": 1.5, "junc_thr": 0.5, "reloc_thr": 0.4, "reloc_shift": 12, "reloc_tol": 1.6, "fork_prune": 0.4, "fast": 2}'
names, add = [], ""
for seed in sys.argv[1:]:
    for arm, short in (("rc", "j1rc"), ("eq", "j1eq")):
        run = f"e24_D7_j1{arm}s{seed}"
        for p in (25, 50, 75):
            src = f"{R}/runs/e123/{run}/model_at{p}.pt"
            if not os.path.exists(src):
                print("missing", src)
                continue
            dst = f"{R}/runs/e123/{run}_at{p}"
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src, f"{dst}/model.pt")
            shutil.copy2(f"{R}/runs/e123_ladder/cards/{run}.json", f"{R}/runs/e123_ladder/cards/{run}_at{p}.json")
            name = f"e24_{short}_s{seed}_at{p}"
            names.append(name)
            if f'"{name}"' not in text:
                add += (f'  "{name}": {{"kind": "builder", "design": "D7", "run": "{run}_at{p}", "set": {{"M": 192, "K0": 96}}, "tau": 0.7,\n'
                        f'               "asm": {ASM},\n               "what": "E241: {run} after {p} % of the 4 u fine-tune"}},\n')
if add:
    i = text.index('  "control":')
    text = text[:i] + add + text[i:]
    json.loads(text)
    open(reg_p, "w", encoding="utf-8").write(text)
print(" ".join(names))
