"""audit scratch: for every ModelConfig / LossConfig field, which values do the stored configs set (count + names)."""
import ast
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = "/mnt/c/Auto-research/bev_fusion"
CLASSES = {"training/model.py": ("ModelConfig", "model"), "training/losses.py": ("LossConfig", "loss")}


def fields_of(path, cls):
    tree = ast.parse(open(os.path.join(ROOT, path), encoding="utf-8").read())
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == cls:
            for s in n.body:
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                    try:
                        d = ast.literal_eval(s.value) if s.value is not None else None
                    except Exception:
                        d = "<expr>"
                    yield s.target.id, d, s.lineno


configs = []
for p in sorted(glob.glob(f"{ROOT}/training/configs/*.json") + glob.glob(f"{ROOT}/training/configs/presets/*.json") + glob.glob(f"{ROOT}/runs/*/config.json")):
    try:
        c = json.load(open(p, encoding="utf-8"))
        if isinstance(c, dict):
            configs.append((p.replace(ROOT + "/", ""), c))
    except Exception:
        pass
print("configs", len(configs))
only = set(sys.argv[1:])
for path, (cls, sub) in CLASSES.items():
    print("\n##", cls)
    for name, default, line in fields_of(path, cls):
        if only and name not in only:
            continue
        vals = defaultdict(list)
        for p, c in configs:
            d = c.get(sub, {}) or {}
            if name in d:
                vals[json.dumps(d[name])].append(p)
        out = []
        for v, ps in vals.items():
            kind = "dflt" if v == json.dumps(default) else "NONDEF"
            out.append(f"{v}[{kind}]x{len(ps)}")
        print(f"  L{line} {name} default={default!r}: " + ("; ".join(out) if out else "never set"))
