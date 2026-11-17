"""E377 throw-away probe: the smoke runs of jtoken_train.py from HEAD's code and from the scratch copy -> per-step losses + a sha256 of the final weights (compare the two outputs: must be equal).
usage: bash tools/env/_py14.sh temp/e377/e377c_identity.py head|scratch"""
import hashlib
import json
import os
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
src = ROOT if sys.argv[1] == "head" else os.path.join(HERE, "_session_scratch_e377c")
for sub in ("training/train", "training/grid"):
    sys.path.insert(0, os.path.join(src, sub))
import torch  # noqa: E402
torch.set_num_threads(2)
import jtoken as JK  # noqa: E402
import jtoken_train as JKT  # noqa: E402
assert JK.__file__.startswith(src) and JKT.__file__.startswith(src), (JK.__file__, JKT.__file__)

RUNS = {"champ_k48": (["--none-weight", "0.25"], 48), "default_k48": ([], 48), "max_k48": (["--none-weight", "0.25", "--pos-loss", "max"], 48),
        "teacher_k48": (["--none-weight", "0.25", "--teacher", "rule"], 48), "drop_k48": (["--none-weight", "0.25", "--token-drop", "0.3"], 48), "champ_k384_bs8": (["--none-weight", "0.25"], 384)}
res, real = {}, JK.jtoken_loss
with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp", "e377")) as runs:
    for name, (extra, k) in RUNS.items():
        losses = []

        def rec(out, batch, **kw):
            r = real(out, batch, **kw)
            losses.append(float(r[0]).hex())
            return r
        JK.jtoken_loss = rec
        bs = "8" if k == 384 else "2"
        JKT.train(JKT.parse_args(["--name", name, "--seed", "0", "--smoke", "--bs", bs, "--workers", "0"] + extra), runs, 500, k)
        JK.jtoken_loss = real
        sd = torch.load(os.path.join(runs, name, "model.pt"), map_location="cpu", weights_only=False)["state"]
        h = hashlib.sha256()
        for key in sorted(sd):
            h.update(key.encode())
            h.update(sd[key].detach().cpu().contiguous().numpy().tobytes())
        res[name] = dict(losses=losses, sha=h.hexdigest())
out = os.path.join(HERE, f"e377c_identity_{sys.argv[1]}.json")
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({n: (r["sha"][:16], r["losses"][-1]) for n, r in res.items()}, indent=1))
