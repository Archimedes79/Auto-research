"""scratch: how many of the 873 VAL frames are already in the l4 store?"""
import os
import sys

ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "training"))
import qgraph_eval as QE  # noqa: E402

store = sys.argv[1]
held = {f[:-7] for f in os.listdir(store) if f.endswith(".pkl.gz")}
fr = QE.val_frames()
print("VAL frames stored:", sum(f.token in held for f in fr), "of", len(fr), " scenes of VAL:", len({f.scene for f in fr}))
