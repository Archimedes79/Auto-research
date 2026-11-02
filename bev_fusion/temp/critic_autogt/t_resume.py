import os, json, tempfile, shutil, numpy as np
from autogt import cli
from autogt.grid import GridSpec
d = tempfile.mkdtemp(dir="bev_fusion/temp/critic_autogt")
json.dump({"res": 0.5, "ss": 4}, open(os.path.join(d, "index.json"), "w"))
for res, ss in ((0.25, 4), (0.5, 2), (0.5, 4)):
    try: cli.check_resume(d, GridSpec(res=res), ss); print("res", res, "ss", ss, "-> accepted")
    except SystemExit as e: print("res", res, "ss", ss, "-> SystemExit:", str(e)[:60])
os.remove(os.path.join(d, "index.json"))
np.savez_compressed(os.path.join(d, "a.npz"), drivable_area=np.zeros((240, 208), np.float16))
try: cli.check_resume(d, GridSpec(res=0.5), 4); print("no index, other grid -> accepted")
except SystemExit as e: print("no index, other grid -> SystemExit")
# a truncated / zero-byte final file is treated as done by _work
open(os.path.join(d, "bad.npz"), "wb").write(b"")
class Fake:
    def keyframe(self, t): raise AssertionError("regenerated")
cli._STATE["gt"] = Fake()
print("zero-byte bad.npz: _work returns", cli._work(("bad", d)), "(0 = skipped as done)")
# a corrupt index.json (e.g. from an old crash)
open(os.path.join(d, "index.json"), "w").write("{")
try: cli.check_resume(d, GridSpec(), 4)
except Exception as e: print("corrupt index.json ->", type(e).__name__)
shutil.rmtree(d)
