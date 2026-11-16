import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.join(ROOT, s))
ck = torch.load(os.path.join(ROOT, "runs/e324_s_s0/model.pt"), map_location="cpu")
a = ck["args"]
print({k: v for k, v in a.items()})
print(a["maps_args"])
st = ck["state"]
print(len(st), sorted({k.split(".")[0] + "." + k.split(".")[1] for k in st}))
print(sum(1 for k in st if k.startswith("jmodel.pyr.")), "pyr tensors")
print([k for k in st if "m05" in k or "s05" in k or "nodes.inp" in k or "nodes.fine" in k])
print({k: tuple(v.shape) for k, v in st.items() if "nodes.inp.0" in k or "nodes.fine.0" in k})
