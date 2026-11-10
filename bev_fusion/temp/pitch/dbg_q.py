import sys, tempfile
sys.path.insert(0, ".")
from evalsuite import selftest_api as S
from evalsuite.api import Evaluation
S.STATE.root = tempfile.mkdtemp()
ev = Evaluation(adapter=S.ME)
r = ev.run("runs/thr0.5", "val", progress=0)
print(r.quant("iou"))
