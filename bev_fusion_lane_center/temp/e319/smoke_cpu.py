"""E319 scratch: the real trainers on CPU for 20 steps (T.fit forced to device cpu, bs 2, no workers): `jmaps_train.py --dirx --smoke`, then `jends_train.py --e2e --scales --smoke` on the maps it wrote, then the loaders of jmaps_eval / junction_view
(JModel(ctx=args["ctx"])). usage: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= BFLC_SIG=1.0 wsl bash tools/env/_py14.sh e319/smoke_cpu.py"""
import os
import sys
import time

import torch

torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
import jends_train as JT  # noqa: E402
import jmaps as JM  # noqa: E402
import jmaps_train as MT  # noqa: E402
import trainer as T  # noqa: E402

fit0 = T.fit
T.fit = lambda *a, **k: fit0(*a, device="cpu", **dict(k, amp="none"))


def run(mod, argv):
    sys.argv = ["x"] + argv
    t0 = time.time()
    mod.main()
    print(f"  {mod.__name__} done in {time.time() - t0:.0f} s", flush=True)


run(MT, ["--name", "e319_smoke_cpu_m_s0", "--seed", "0", "--ctx", "dil", "--sigma", "1.0", "--dirx", "--smoke", "--workers", "0", "--bs", "2"])
ck = torch.load(os.path.join(ROOT, "runs", "e319_smoke_cpu_m_s0", "model.pt"), map_location="cpu")
print("maps checkpoint args: ctx", ck["args"]["ctx"], "dirx", ck["args"]["dirx"], "| dirx tensors in the state:", sum(k.startswith("jm.dirx.") for k in ck["state"]))
assert ck["args"]["ctx"] == "dil+dirx" and any(k.startswith("jm.dirx.") for k in ck["state"])
m = JM.JModel(width=48, d=ck["args"].get("d", 64), ctx=ck["args"].get("ctx", "dil"))                    # the line of jmaps_eval.score / junction_view
m.load_state_dict(ck["state"])
moved = sum(float(v.abs().sum()) for k, v in ck["state"].items() if k.startswith("jm.dirx.") and k.endswith(("3.weight", "3.bias")))
print("the zero-initialised last layers moved in 20 steps: sum |w| =", f"{moved:.4f}")
run(JT, ["--name", "e319_smoke_cpu_p_s0", "--seed", "0", "--maps", os.path.join(ROOT, "runs", "e319_smoke_cpu_m_s0", "model.pt"), "--e2e", "--scales", "--maps-w", "1.0", "--smoke", "--workers", "0", "--bs", "2"])
ck2 = torch.load(os.path.join(ROOT, "runs", "e319_smoke_cpu_p_s0_maps", "model.pt"), map_location="cpu")
print("maps run written by the pair stage: ctx", ck2["args"]["ctx"])
assert ck2["args"]["ctx"] == "dil+dirx"
m2 = JM.JModel(width=48, d=64, ctx=ck2["args"]["ctx"])
m2.load_state_dict(ck2["state"])
print("smoke ok")
