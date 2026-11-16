import hashlib, inspect, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
R = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.join(R, sub))
import jends_train as JT
src = "\n".join(l.rstrip() for l in inspect.getsource(JT.JEndModel.forward).splitlines())
print(hashlib.sha256(src.encode()).hexdigest())
print(src)
