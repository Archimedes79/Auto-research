import os, sys, time
ROOT = os.environ["BFLC_ROOT"]
import numpy as np, torch
sys.path.insert(0, os.path.join(ROOT, "training/data"))
import lane_live_tok as LL
import jtoken as JK
from unittest import mock
torch.set_num_threads(2)
B = 2
ds = LL.LiveData("val", 0)
b = LL.collate([ds[i] for i in range(B)])
src = LL.LiveSource(os.path.join(ROOT, "runs/e292_a_s0"), 512, True)
sys.path.insert(0, os.path.join(ROOT, "training/train"))
import jtoken_train as JTR
head, args = JTR.load_model("e365_s1_s0", "cpu")
print("champion head", args.get("steps"), head._step_py, "params", sum(p.numel() for p in head.parameters()))
def saved_bytes(fn):
    seen, total = set(), [0]
    def pack(t):
        st = t.untyped_storage(); key = (st.data_ptr(), st.nbytes())
        if key not in seen:
            seen.add(key); total[0] += st.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        r = fn()
    return r, total[0]
out = src(b["x"], b["table"], b["token"], b["row"])
batch = out["batch"]
mb = lambda n: round(n / 2**20, 1)
_, cut = saved_bytes(lambda: JK.jtoken_loss(head(batch), batch)[0])
with mock.patch.object(JK, "_const", lambda t: t):
    (loss, info), live = saved_bytes(lambda: JK.jtoken_loss(head(batch), batch))
print("head saved for backward (MB, bs 2, fp32): cut inputs", mb(cut), " live inputs", mb(live), " -> bs 8 delta", mb((live - cut) * 4))
# champion head: gradient into B vs B's own loss, step 0
src2 = LL.LiveSource(os.path.join(ROOT, "runs/e292_a_s0"), 512, True)
with mock.patch.object(JK, "_const", lambda t: t):
    out = src2(b["x"], b["table"], b["token"], b["row"])
    loss, info = JK.jtoken_loss(head(out["batch"]), out["batch"])
loss.backward()
gh = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in src2.trainable()))
for p in src2.trainable(): p.grad = None
out = src2(b["x"], b["table"], b["token"], b["row"])
lb, ib = LL.b_loss(out["no"], b["targets"])
lb.backward()
gb = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in src2.trainable()))
print("champion head loss %.3f (nll_s %.3f), B's node loss %.3f %s" % (float(loss), info["nll_s"], float(lb), {k: round(v, 3) for k, v in ib.items()}))
print("|grad| on the node head: from the head's loss %.4f, from B's own loss %.4f; the head's share of the norm sum %.3f" % (float(gh), float(gb), float(gh / (gh + gb))))
