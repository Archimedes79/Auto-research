"""E292 scratch: reference outputs of B (runs/e273_p_s0) through the node and link stage on a fixed random input, before and after the code change (flag off = bit for bit). usage: ref_b.py save|check"""
import os, sys, torch
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/train", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
sys.path.insert(0, os.path.join(ROOT, "temp", "_session_scratch_s292", "new"))
import qgraph as _QG; assert "_session_scratch_" in _QG.__file__, _QG.__file__
import qpredict as QP
torch.manual_seed(0)
x = torch.randn(1, 39, 200, 200)
m = QP.load_model(os.path.join(ROOT, "runs", "e273_p_s0"), "cpu").eval()
with torch.no_grad():
    out = m({"x": x}, upto="links", gt=False)
res = {}
for grp in ("no", "L"):
    for k, v in out[grp].items():
        if torch.is_tensor(v):
            res[f"{grp}.{k}"] = v.detach().float().cpu()
path = os.path.join(ROOT, "temp", "s292", "ref_b.pt")
if sys.argv[1] == "save":
    torch.save(res, path); print("saved", len(res), "tensors")
else:
    ref = torch.load(path)
    assert set(ref) == set(res), set(ref) ^ set(res)
    bad = [k for k in ref if not torch.equal(ref[k], res[k])]
    assert not bad, bad
    print("identical:", len(res), "tensors")
