"""scratch (E379, the code critic's HEAD comparison before the move): 20 smoke steps of the champion, --pos-loss max and --none-weight-near with HEAD's jtoken / jtoken_train (e379f_base_*) against the scratch copies
(e379f_*), same process, bit for bit (the loss of every step, the final weights)"""
import os
import sys
import tempfile
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
G = os.path.join(ROOT, "training", "grid")
for sub_ in ("training", "training/grid", "training/train", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub_))
import torch  # noqa: E402
torch.set_num_threads(2)


def pair(pre):
    """(jtoken, jtoken_train) of the copies with prefix `pre`, jtoken_train bound to that jtoken"""
    mods = []
    for name, real in (("jtoken", os.path.join(G, "jtoken.py")), ("jtoken_train", os.path.join(ROOT, "training", "train", "jtoken_train.py"))):
        mod = types.ModuleType(name)
        mod.__file__ = real
        sys.modules[name] = mod
        path = os.path.join(HERE, pre + name + ".py")
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), mod.__dict__)
        mods.append(mod)
    return mods


def run(JKT, extra, runs, name):
    losses, real = [], JKT.make_loss

    def rec(*a, **k):
        f = real(*a, **k)

        def g(m, b):
            r = f(m, b)
            losses.append(float(r[0]))
            return r
        return g
    JKT.make_loss = rec
    try:
        JKT.train(JKT.parse_args(["--name", name, "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"] + extra), runs, 500, 48)
    finally:
        JKT.make_loss = real
    return losses, torch.load(os.path.join(runs, name, "model.pt"), map_location="cpu", weights_only=False)["state"]


head, new = pair("e379f_base_"), pair("e379f_")
assert head[1].JK is head[0] and new[1].JK is new[0] and not hasattr(head[0], "distill_loss") and hasattr(new[0], "distill_loss")
recipes = {"champion": ["--none-weight", "0.25"], "pos_loss max": ["--none-weight", "0.25", "--pos-loss", "max"], "none_weight_near": ["--none-weight", "0.25", "--none-weight-near", "0.73"],
           "default W 1": [], "token drop": ["--none-weight", "0.25", "--token-drop", "0.2"], "head seed 2": ["--none-weight", "0.25", "--head-seed", "2"]}
with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp", "e365")) as runs:
    for n, (rname, extra) in enumerate(recipes.items()):
        (lh, sh), (ln, sn) = run(head[1], extra, runs, f"h{n}"), run(new[1], extra, runs, f"n{n}")
        same = lh == ln and len(lh) == 20 and list(sh) == list(sn) and all(torch.equal(v, sn[k]) for k, v in sh.items())
        print(f"{rname}: HEAD vs new, 20 smoke steps: {'bit for bit' if same else 'DIFFERENT'} (last loss {lh[-1]:.6f} / {ln[-1]:.6f})")
        assert same, rname
print("headcmp ok")
