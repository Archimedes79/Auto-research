"""scratch: keyframes of the generator of the CURRENT WORKING DIRECTORY (its autogt package) into a pickle: python dump_kf.py out.pkl [stride]"""
import os
import pickle
import sys
import time

sys.path.insert(0, ".")
import autogt
from autogt import AutoGT, scene_tokens

out, stride = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
scenes = ["scene-0061", "scene-0553", "scene-0796", "scene-1077"]
root = os.path.expanduser("~/data/nuscenes")
gen = AutoGT(root, "v1.0-mini", light=True, scenes=scenes)
res, t0 = {}, time.time()
for s in scenes:
    for tok in scene_tokens(s, "v1.0-mini", root)[::stride]:
        res[tok] = gen.keyframe(tok)
pickle.dump({"sig": gen.signature(), "kf": res, "file": autogt.__file__}, open(out, "wb"))
print(autogt.__file__, gen.signature(), len(res), "keyframes", f"{(time.time() - t0) / len(res) * 1000:.0f} ms each")
