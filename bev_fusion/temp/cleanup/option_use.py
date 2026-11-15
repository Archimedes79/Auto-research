"""Scratch: which fields of ModelConfig / LossConfig / TrainConfig does any stored config set to a non-default value?

stored configs = training/configs/*.json, presets/*.json and runs/*/config.json on this disk.
A field that no config ever changes and that only the code that defines / reads it mentions is a candidate dead option.
"""
import ast
import glob
import json
import os
import re
import subprocess
import sys

ROOT = "bev_fusion"
CLASSES = {"training/model.py": ["ModelConfig"], "training/losses.py": ["LossConfig"], "training/train.py": ["TrainConfig"]}


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
for p in glob.glob(f"{ROOT}/training/configs/*.json") + glob.glob(f"{ROOT}/training/configs/presets/*.json") + glob.glob(f"{ROOT}/runs/*/config.json"):
    try:
        c = json.load(open(p, encoding="utf-8"))
        if isinstance(c, dict):
            configs.append((p, c))
    except Exception:
        pass
print("configs:", len(configs))

files = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
src = {f: open(f, encoding="utf-8", errors="replace").read() for f in files if f.startswith(("bev_fusion/", "autoresearch/"))}

for path, classes in CLASSES.items():
    for cls in classes:
        print(f"\n## {cls} ({path})")
        for name, default, line in fields_of(path, cls):
            sub = {"ModelConfig": "model", "LossConfig": "loss"}.get(cls)
            used_by = [p for p, c in configs if name in (c.get(sub, {}) if sub else c) and (c.get(sub, {}) if sub else c)[name] != default]
            mentioned = [f for f, s in src.items() if re.search(r"\b%s\b" % re.escape(name), s)]
            in_tests = [f for f in mentioned if "/tests/" in f or "selftest" in f]
            code = [f for f in mentioned if f not in in_tests and not f.endswith(path)]
            tag = "" if used_by else ("   <-- no config changes it" + ("; only its own file and tests" if not code else ""))
            if not used_by:
                print(f"  {name} = {default!r}  [{len(used_by)} configs; own file + {len(code)} other code files, {len(in_tests)} tests]{tag}")
