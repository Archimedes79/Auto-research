import os, sys, torch
R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid"):
    sys.path.insert(0, os.path.join(R, sub))
import jmaps as JM
m = JM.JModel(width=48, d=64, ctx="dil")
m.load_base(os.path.join(R, "runs", "e273_p_s0", "model.pt"))
d = os.path.join(R, "runs", "e286_dummy_s0")
os.makedirs(d, exist_ok=True)
torch.save(dict(state=m.state_dict(), args=dict(d=64, ctx="dil", seed=0, joint=False)), os.path.join(d, "model.pt"))
print("dummy written")
