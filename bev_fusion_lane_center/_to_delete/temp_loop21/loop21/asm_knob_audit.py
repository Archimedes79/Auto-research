"""Loop 21 audit (one-off, no data): which keyword arguments of PI.assemble() does any registry entry, TM.ASM or a code path of the product use? Unused knobs are candidates for removal (simpler assembly).
    wsl bash tools/env/_py14.sh temp/loop21/asm_knob_audit.py"""
import inspect
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import evaluate as EV  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402

sig = inspect.signature(PI.assemble)
knobs = [k for k in sig.parameters if k not in ("r", "b", "tau", "cfg")]
reg = EV.registry()
used_reg = {}
for n, e in reg.items():
    for k in (e.get("asm") or {}):
        used_reg.setdefault(k, []).append(n)
used_tm = set(getattr(TM, "ASM", {}))
src = {}
root = EV.R
for base in ("training", "evaluation", "tools"):
    for dp, dn, fn in os.walk(f"{root}/{base}"):
        if any(x in dp for x in ("legacy", "__pycache__")):
            continue
        for f in fn:
            if f.endswith((".py", ".sh", ".json")):
                try:
                    src[f"{dp}/{f}"] = open(f"{dp}/{f}", encoding="utf-8", errors="ignore").read()
                except OSError:
                    pass
body = inspect.getsource(PI.assemble)
print(f"assemble() has {len(knobs)} keyword arguments; registry entries: {len(reg)}; TM.ASM keys: {sorted(used_tm)}")
print(f"{'knob':14} {'default':10} {'registry entries':>16} {'TM.ASM':>7} {'other files naming it':>22}  uses in assemble() body")
for k in knobs:
    d = sig.parameters[k].default
    others = [p.replace(root + "/", "") for p, s in src.items() if re.search(rf"\b{k}\b", s) and "p16_infer.py" not in p and "registry.json" not in p]
    nb = len(re.findall(rf"\b{k}\b", body))
    print(f"{k:14} {str(d):10} {len(used_reg.get(k, [])):16d} {'yes' if k in used_tm else '':>7} {len(others):22d}  {nb}")
