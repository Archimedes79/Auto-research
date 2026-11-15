import os, sys, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
import jnarrow as JN
import jprune_probe as PP
t0 = time.time()
spec = {"calib": f"temp/e345/e324_s_s0/calib.pt", "edits": [{"unit": "jm.m05.hid", "frac": 0.75}]}
full, args = JN.build_narrowed("e324_s_s0", {})
nar, _ = JN.build_narrowed("e324_s_s0", spec)
print("built", time.time() - t0)
sp = nar.narrow_spec
print(json.dumps(sp["taps"]), sp["calib_sha256"])
print(nar.jmodel.jm.m05)
print(nar.jmodel.jm.m05[3].bias is not None)
es = JN.expected_saving(full, sp)
print("Cost saved", {k: v / 1e9 if "macs" in k else v for k, v in es.items()})
mf, mn = JN.measured_macs(full), JN.measured_macs(nar)
print("measured full", mf, "narrowed", mn)
print("measured saved", {k: mf[k] - mn[k] for k in mf})
print(time.time() - t0)
# probe's own channel count vs ours
calib = PP.load_calib(PP.rootpath(spec["calib"]), "cpu")
units = PP.build_units(full)
pe = PP.spec_edits({"jm.m05.hid": 0.75}, units, calib)
print("probe idx", pe[0].idx, len(pe[0].idx))
print("ours removed", sp["taps"]["jm.m05.2"]["removed"])
cache = PP.build_cache("val", workers=0, max_frames=8)
xs = [cache[0]["x"].float()]
t1 = time.time()
d = JN.difference(full, nar, xs)
print(json.dumps(d, indent=1), time.time() - t1)
