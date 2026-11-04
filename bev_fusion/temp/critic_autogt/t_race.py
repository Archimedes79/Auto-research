import os, tempfile, threading, shutil, collections
import numpy as np
from autogt import cli
class Fake:
    def keyframe(self, tok):
        rng = np.random.default_rng(abs(hash(tok)) % 2**32)
        g = {"drivable_area": rng.random((120, 104), dtype=np.float32), "flow": rng.random((2, 120, 104), dtype=np.float32), "meta": {"sample": tok}}
        return g
d = tempfile.mkdtemp(dir="bev_fusion/temp/critic_autogt")
cli._STATE["gt"] = Fake()
toks = [f"tok{i:03d}" for i in range(60)]
errs = collections.Counter()
def run():
    for t in toks:
        try: cli._work((t, d))
        except Exception as e: errs[type(e).__name__ + ": " + str(e)[:70]] += 1
ths = [threading.Thread(target=run) for _ in range(2)]
[t.start() for t in ths]; [t.join() for t in ths]
bad = 0
for t in toks:
    try: cli.load_keyframe(d, t)
    except Exception as e: bad += 1
print("exceptions in the two writers:", dict(errs), "| unreadable final files:", bad, "| leftover tmp:", [f for f in os.listdir(d) if ".tmp" in f][:3])
shutil.rmtree(d)
