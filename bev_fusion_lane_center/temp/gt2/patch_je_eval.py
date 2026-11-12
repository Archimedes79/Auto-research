p = "evaluation/jends_eval.py"
s = open(p, encoding="utf-8").read()
s = s.replace("            for i in range(len(b[\"token\"])):\n                frames.append(", "            if a.max and len(frames) >= a.max:\n                break\n            for i in range(len(b[\"token\"])):\n                frames.append(", 1)
s = s.replace('        s.add_argument("--thr", type=float, default=0.5)\n        if name == "score":', '        s.add_argument("--thr", type=float, default=0.5)\n        if name == "dump":\n            s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; the score needs all 873)")\n        if name == "score":', 1)
open(p, "w", encoding="utf-8").write(s)
d = "temp/gt2/dummy_run2.py"
open(d, "w", encoding="utf-8").write('''import os, sys, torch
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
''')
print("ok")
