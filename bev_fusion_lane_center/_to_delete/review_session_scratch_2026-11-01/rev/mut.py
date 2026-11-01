import sys, traceback
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion/evaluation")
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion")
from evaluation import bev_selftest as S
from evaluation import bev_task as BT
S.init_synthetic()
fn = [f for n, f in S.CHECKS if n.startswith("to_legacy")][0]
def run(label):
    try:
        fn(); print("PASS", label)
    except Exception as e:
        print("FAIL", label, type(e).__name__, str(e)[:150])
run("as is")
# extra probe: what the loss and seg keys look like
from evaluation import bev_suite
import tempfile
suite = S.make_suite(tempfile.mkdtemp()); st = suite.quant.run(scenes=["bev-00"], name="a", progress=0); tab = suite.quant.table(st)
bt = bev_suite.better_of(tab, suite.task.metrics())
print("loss keys", {k: v for k, v in bt.items() if k.startswith("val_loss")})
sys.path.insert(0, "/mnt/c/Auto-research/autoresearch/tools")
import os; os.environ["AR_PROJECT"] = "../bev_fusion"
import decide as DC
root = "/mnt/c/Auto-research/bev_fusion"
dis = [(k, v) for k, v in bt.items() if (v == "lower") != DC.named_lower(k, root)]
print("better map vs project.json fallback disagree on", len(dis), "keys:", sorted({k.rsplit('/',1)[-1] for k, _ in dis}))
suite.close()
# mutation 1: field_mae back to high
orig_seg = BT.BevSegMetric.direction
BT.BevSegMetric.direction = lambda self, key: "high"
run("field_mae high")
BT.BevSegMetric.direction = orig_seg
# mutation 2: loss direction removed
orig_loss = BT.BevLossMetric.direction
del BT.BevLossMetric.direction
run("loss direction() removed")
BT.BevLossMetric.direction = orig_loss
