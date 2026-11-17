"""S1 scratch: reference outputs of the E312 model on a fixed random input (CPU, eval): run before and after a code removal; usage: ref_out.py save|check"""
import os, sys, torch
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "training", "training/grid", "training/train", "training/data", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jends_eval as JV
torch.manual_seed(0)
x = torch.randn(1, 39, 200, 200)
res = {}
for run in ("e312_p_s0", "e320_a_s0"):
    m, a = JV.load_model(run, "cpu")
    with torch.no_grad():
        P, out, oe = m(x, gt=None)
    res[run] = {k: v.detach().float().cpu() for k, v in oe.items() if torch.is_tensor(v)}
path = os.path.join(ROOT, "temp", "s1", "ref.pt")
if sys.argv[1] == "save":
    torch.save(res, path); print("saved", {r: len(v) for r, v in res.items()})
else:
    ref = torch.load(path)
    for r in res:
        assert set(ref[r]) == set(res[r]), (set(ref[r]) ^ set(res[r]))
        bad = [k for k in ref[r] if not torch.equal(ref[r][k], res[r][k])]
        assert not bad, (r, bad)
    print("identical:", {r: len(v) for r, v in res.items()})
