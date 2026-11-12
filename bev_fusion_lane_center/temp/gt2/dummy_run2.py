import os, sys, torch
R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/train"):
    sys.path.insert(0, os.path.join(R, sub))
import jends_train as JT
ck = torch.load(os.path.join(R, "runs", "e286_dummy_s0", "model.pt"), map_location="cpu")
m = JT.JEndModel(ck["args"], order=True, evidence=True)
m.jmodel.load_state_dict(ck["state"])
d = os.path.join(R, "runs", "e287_dummy_s0")
os.makedirs(d, exist_ok=True)
torch.save(dict(state=m.state_dict(), args=dict(seed=0, no_order=False, no_evidence=False, maps="dummy", maps_args=ck["args"])), os.path.join(d, "model.pt"))
print("dummy2 written")
