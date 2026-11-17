"""E330 scratch: the real trainers on CPU for 20 steps (T.fit forced to device cpu, bs 2, no workers) for the four simplification arms: S3b (maps --no-paint --no-zone, then the pair stage --no-evidence on them, a dump of 8 VAL frames, the maps loader of jmaps_eval),
S4 (maps --d 40, then the pair stage with evidence), S1 (the pair stage --no-evidence on the real E307 maps), S5 (make_init, then the pair stage from the untrained head).
usage: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= BFLC_SIG=1.0 wsl bash tools/env/_py14.sh e330/e330_smoke_cpu.py"""
import os
import runpy
import sys
import time
from types import SimpleNamespace

import torch

torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
import jends_eval as JEV  # noqa: E402
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
    print(f"  {mod.__name__} {' '.join(argv[:2])} done in {time.time() - t0:.0f} s", flush=True)


def ck(name):
    return torch.load(os.path.join(ROOT, "runs", name, "model.pt"), map_location="cpu")


common = ["--seed", "0", "--smoke", "--workers", "0", "--bs", "2"]
# S3b: the maps head without the paint branch and without the zone supervision, then the pair stage without evidence on it
run(MT, ["--name", "e332_smoke_cpu_mb_s0", "--ctx", "dil", "--sigma", "1.0", "--no-paint", "--no-zone"] + common)
c = ck("e332_smoke_cpu_mb_s0")
assert c["args"]["ctx"] == "dil+nopaint+nozone" and not any(k.startswith(("jm.m05.", "jm.head05.")) for k in c["state"]), c["args"]["ctx"]
run(JT, ["--name", "e332_smoke_cpu_pb_s0", "--maps", os.path.join(ROOT, "runs", "e332_smoke_cpu_mb_s0", "model.pt"), "--no-evidence", "--e2e", "--scales", "--maps-w", "1.0"] + common)
c2 = ck("e332_smoke_cpu_pb_s0_maps")
assert c2["args"]["ctx"] == "dil+nopaint+nozone"
m = JM.JModel(width=48, d=c2["args"].get("d", 64), ctx=c2["args"]["ctx"])                          # the line of jmaps_eval.score / junction_view
m.load_state_dict(c2["state"])
print("  maps run written by the pair stage loads as JModel(ctx=", c2["args"]["ctx"], ")")
JEV.dump(SimpleNamespace(run="e332_smoke_cpu_pb_s0", thr=0.5, max=8))                              # the dump path of the chain: JEndModel from maps_args with the flags, no evidence
print("  dump of 8 VAL frames ok", flush=True)
# S4: width 40
run(MT, ["--name", "e333_smoke_cpu_m_s0", "--ctx", "dil", "--sigma", "1.0", "--d", "40"] + common)
assert ck("e333_smoke_cpu_m_s0")["args"]["d"] == 40
run(JT, ["--name", "e333_smoke_cpu_p_s0", "--maps", os.path.join(ROOT, "runs", "e333_smoke_cpu_m_s0", "model.pt"), "--e2e", "--scales", "--maps-w", "1.0"] + common)
# S1: no evidence on the real E307 maps
run(JT, ["--name", "e331_smoke_cpu_e_s0", "--maps", os.path.join(ROOT, "runs", "e307_n_s0", "model.pt"), "--no-evidence", "--e2e", "--scales", "--maps-w", "1.0"] + common)
# S5: the untrained head, one joint stage
sys.argv = ["make_init.py", "0"]
runpy.run_path(os.path.join(ROOT, "temp", "e334", "make_init.py"), run_name="__main__")
run(JT, ["--name", "e334_smoke_cpu_o_s0", "--maps", os.path.join(ROOT, "runs", "e334_init_s0", "model.pt"), "--e2e", "--scales", "--maps-w", "1.0"] + common)
print("e330 smoke ok")
