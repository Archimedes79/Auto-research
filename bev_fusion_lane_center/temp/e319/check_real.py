"""E319 scratch: real-frame checks of the dirx block. (1) flag off == original on real VAL frames with the trained E307 weights; (2) flag on at init == flag off on the same frames; (3) the heading field of the pyramid's centre channels (1, 2)
points ALONG the lane ridge in the (rows = x, columns = y) convention of `shift_along`: the centre-lane probability read ahead / behind along t stays high, read across (the normal) it drops.
usage: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= wsl bash tools/env/_py14.sh e319/check_real.py [seed]"""
import os
import sys

import torch

torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
import jmaps_dev as D  # noqa: E402
import jmaps_orig as O  # noqa: E402
import lane_gt_data as LD  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
LD.QData.DEFAULT_JUNCTION = True
ck = torch.load(os.path.join(ROOT, "runs", f"e307_n_s{seed}", "model.pt"), map_location="cpu")
args = ck["args"]
print("E307 args ctx", args.get("ctx"), "d", args.get("d"), "sigma", args.get("sigma"))
mo = O.JModel(width=48, d=args.get("d", 64), ctx=args.get("ctx", "dil"))
mo.load_state_dict(ck["state"])
md = D.JModel(width=48, d=args.get("d", 64), ctx=args.get("ctx", "dil"))
md.load_state_dict(ck["state"])
mx = D.JModel(width=48, d=args.get("d", 64), ctx=args.get("ctx", "dil") + "+dirx")
rep = mx.load_state_dict(ck["state"], strict=False)
assert not rep.unexpected_keys and all(k.startswith("jm.dirx.") for k in rep.missing_keys), rep
for m in (mo, md, mx):
    m.eval()
ds, dl = LD.loader("val_clean", bs=2, workers=0, shuffle=False, map_mode="exact", pixels=False)
n = 0
along, across, along_sw, across_sw = [], [], [], []
for b in dl:
    x = b["x"].float()
    with torch.no_grad():
        Po, oo = mo(x)
        Pd, od = md(x)
        Px, ox = mx(x)
    assert all(torch.equal(oo[k], od[k]) for k in oo), "flag off differs from the original on real frames"
    assert all(torch.equal(oo[k], ox[k]) for k in oo), "flag on at init differs from the original on real frames"
    c1 = Po["centre1"].float()
    p = c1[:, :1].sigmoid()
    ridge = (p[:, 0] > 0.6)
    for name, t in (("cos->rows", D._unit2(c1[:, 1:3])), ("swapped", D._unit2(c1[:, [2, 1]]))):
        nrm = torch.stack([-t[:, 1], t[:, 0]], 1)
        a = 0.5 * (D.shift_along(p, t, 3.0) + D.shift_along(p, t, -3.0))[:, 0]
        c = 0.5 * (D.shift_along(p, nrm, 3.0) + D.shift_along(p, nrm, -3.0))[:, 0]
        (along if name == "cos->rows" else along_sw).append(a[ridge])
        (across if name == "cos->rows" else across_sw).append(c[ridge])
    n += x.shape[0]
    if n >= 16:
        break
cat = lambda v: float(torch.cat(v).mean())  # noqa: E731
print(f"{n} real VAL frames: flag off == original (all outputs bit-identical), flag on at init == original: ok")
print(f"centre-lane probability read 3 m along the heading (mean of ahead and behind) on ridge cells (p > 0.6): {cat(along):.3f}; read 3 m across: {cat(across):.3f}   [convention (cos, sin) = (rows, columns)]")
print(f"the same with the two heading channels swapped: along {cat(along_sw):.3f}, across {cat(across_sw):.3f}")
