import os, sys, pickle, json, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
import jnarrow as JN
import jnarrow_eval as JNE
import jprune_probe as PP
from types import SimpleNamespace
keep = JN.JEV.load_model
with tempfile.TemporaryDirectory() as ref, tempfile.TemporaryDirectory() as wrap:
    for root in (ref, wrap):
        os.makedirs(os.path.join(root, "runs", "e324_s_s0"))
        os.symlink(os.path.join(ROOT, "runs", "e324_s_s0", "model.pt"), os.path.join(root, "runs", "e324_s_s0", "model.pt"))
    old = JN.JEV.ROOT
    JN.JEV.ROOT = ref
    JN.JEV.dump(SimpleNamespace(run="e324_s_s0", thr=0.5, max=24))
    JN.JEV.ROOT = wrap
    JNE.dump(["--run", "e324_s_s0", "--max", "24"])
    JN.JEV.ROOT = old
    JN.JEV.load_model = keep
    a, b = (pickle.load(open(os.path.join(r, "runs", "e324_s_s0", "ends_val.pkl"), "rb")) for r in (ref, wrap))
    n_arr = 0
    assert len(a) == len(b) == 24
    for x, y in zip(a, b):
        assert x.keys() == y.keys() and x["token"] == y["token"]
        for k in x:
            if isinstance(x[k], np.ndarray):
                assert np.array_equal(x[k], y[k]), k
                n_arr += 1
    ha = open(os.path.join(ref, "runs", "e324_s_s0", "ends_head.json"), "rb").read()
    hb = open(os.path.join(wrap, "runs", "e324_s_s0", "ends_head.json"), "rb").read()
    print("EQUAL: 24 frames,", n_arr, "arrays exactly equal; ends_head.json byte-identical:", ha == hb, "; pickles byte-identical:",
          open(os.path.join(ref, "runs", "e324_s_s0", "ends_val.pkl"), "rb").read() == open(os.path.join(wrap, "runs", "e324_s_s0", "ends_val.pkl"), "rb").read())
# importance retained by the group choice vs the probe's per-channel choice
for S in (0, 1):
    calib = PP.load_calib(PP.rootpath(f"temp/e345/e324_s_s{S}/calib.pt"), "cpu")
    s = calib.s["jm.m05.2"].double()
    full, _ = JN.JEV.load_model(f"e324_s_s{S}", "cpu")
    sp = JN.resolve(full, {"calib": f"temp/e345/e324_s_s{S}/calib.pt", "edits": [{"unit": "jm.m05.hid", "frac": 0.75}]})
    kept_g = [c for c in range(32) if c not in sp["taps"]["jm.m05.2"]["removed"]]
    probe = PP.spec_edits({"jm.m05.hid": 0.75}, PP.build_units(full), calib)[0].idx
    kept_p = [c for c in range(32) if c not in probe]
    print(S, "importance kept (share of the total): groups", float(s[kept_g].sum() / s.sum()), "probe channels", float(s[kept_p].sum() / s.sum()), "kept channels", kept_g)
