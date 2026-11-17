"""throw-away: which card fields differ between the old module and the scratch copy on the cache path (4 VAL frames, e373_cb_s0, D115 cell and the record decode)"""
import importlib.util
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ["BFLC_ROOT"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


NEW = load("jtoken_eval", os.path.join(HERE, "e372r_jtoken_eval.py"))
OLD = load("jtoken_eval_old", os.path.join(ROOT, "evaluation", "jtoken_eval.py"))


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: only in one" for k in set(a) ^ set(b)]
        return out + [x for k in set(a) & set(b) for x in diff(a[k], b[k], f"{path}/{k}")]
    return [] if a == b else [f"{path}: {str(a)[:120]} | {str(b)[:120]}"]


with tempfile.TemporaryDirectory(dir=HERE) as tmp:
    cards = {}
    for k, E in (("old", OLD), ("new", NEW)):
        runs, out = os.path.join(tmp, "runs"), os.path.join(tmp, "out", k)
        os.makedirs(os.path.join(runs, "e373_cb_s0"), exist_ok=True)
        os.makedirs(out)
        if k == "old":
            shutil.copy(os.path.join(ROOT, "runs", "e373_cb_s0", "model.pt"), os.path.join(runs, "e373_cb_s0", "model.pt"))
            E.dump(E.parse_args(["dump", "--run", "e373_cb_s0", "--seed", "0", "--cpu", "--max", "4"]), runs)
        for v in ([], ["--link-thr", "0.3", "--link-thr-long", "0.2", "--keep-thr", "0.6", "--no-none-gate"]):
            cards[(k, bool(v))] = E.score(E.parse_args(["score", "--run", "e373_cb_s0", "--seed", "0", "--max", "4"] + v), runs, out)
    for v in (False, True):
        for dec in cards[("old", v)]:
            print(f"variant {v} {dec}:", diff(cards[("old", v)][dec], cards[("new", v)][dec]) or "identical")
