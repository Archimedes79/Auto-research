"""throw-away: which assertion each mutant of test_maps trips; the unmutated exec'd copy must pass"""
import importlib.util
import os
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, "..", ".."))
for s in ("training/grid", "training", "training/data", "training/data/prep", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(R, s))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


JKs = load("jtoken", os.path.join(HERE, "jtoken_e374hook.py"))
T = load("test_jtoken_scratch", os.path.join(HERE, "test_jtoken_e374hook.py"))
T._maps_hook_ok(T._mutant())
print("unmutated exec copy: passes")
cond, call = "self.map_block is not None and maps is not None", "self.map_block(x, pos, dirv, valid, maps)"
hook = f"if n == 2 and {cond}:"
mutants = {"after layer 3": [(hook, hook.replace("n == 2", "n == 3"))], "after every layer from 2 on": [(hook, hook.replace("n == 2 and ", ""))],
           "before layer 2": [(hook, hook.replace("n == 2", "n == 0")), ("        types1, w1 = self.read_types(x, tp)\n", f"        if {cond}:\n            x = {call}\n        types1, w1 = self.read_types(x, tp)\n")],
           "{} skipped": [(hook, hook.replace("maps is not None", "maps"))], "maps None passed on": [(hook, hook.replace(" and maps is not None", ""))], "the block gets past": [(call, call.replace("maps)", "past)"))]}
for name, edits in mutants.items():
    try:
        T._maps_hook_ok(T._mutant(*edits))
        print(name, "SURVIVED")
    except AssertionError as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"{name}: line {tb.lineno}: {tb.line[:110]} | {str(e)[:80]}")
