"""E354 scratch (one use): what the stored tools do with a checkpoint of jrecur_train.py (the CPU smoke run). CPU only."""
import os
import re
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, sub))
torch.set_num_threads(2)
RUN = sys.argv[1]
import jends_eval as JEV  # noqa: E402
import jrecur as JR  # noqa: E402

ck = torch.load(os.path.join(ROOT, "runs", RUN, "model.pt"), map_location="cpu")
a = ck["args"]
print("args:", {k: a[k] for k in ("seed", "state", "seq", "bs", "scales", "no_order", "no_evidence", "steps")}, "maps", a["maps"])
try:
    JEV.load_model(RUN, "cpu")
    print("jends_eval.load_model: loads")
except Exception as e:  # noqa: BLE001
    print("jends_eval.load_model FAILS:", type(e).__name__, str(e)[:400].replace("\n", " "))
m, _ = JR.load_model(RUN, "cpu")
print("jrecur.load_model: ok, gate", float(m.gate().mean()))
# the checks of jends_eval.score before qends_eval.cmd_score (lines 74-86)
mm = re.search(r"_s(\d+)[/\\]", str(a.get("maps", "")).replace("\\", "/") + "/")
base = torch.load(os.path.join(ROOT, "runs", f"e273_p_s{a['seed']}", "model.pt"), map_location="cpu")["state"]
keys = [k for k in base if k.startswith("pyr.") and ("jmodel." + k) in ck["state"]]
same = all(torch.equal(ck["state"]["jmodel." + k], base[k]) for k in keys)
print("score pre-checks: seed", a.get("seed"), "maps seed", mm and int(mm.group(1)), "pyramid tensors", len(keys), "equal to the base run's:", same)
import jmodel_cost as JC  # noqa: E402
from types import SimpleNamespace  # noqa: E402
mc = JC.build(SimpleNamespace(run=RUN))
print("jmodel_cost.build:", type(mc).__name__, "parameters", sum(p.numel() for p in mc.parameters()), "against the run's model", sum(p.numel() for p in m.parameters()))
c = JC.cost(m)
print("jmodel_cost.cost on the JRecurModel:", {k: (round(v["params_m"], 4), round(v["macs_g"], 4)) for k, v in c.items()})
