import os, sys, time
ROOT = os.environ["BFLC_ROOT"]
import numpy as np, torch
sys.path.insert(0, os.path.join(ROOT, "training/data"))
import lane_live_tok as LL
torch.set_num_threads(2)
B = 2
ds = LL.LiveData("val", 0)
b = LL.collate([ds[i] for i in range(B)])
src = LL.LiveSource(os.path.join(ROOT, "runs/e292_a_s0"), 512, True)
m = src.model
mb = lambda n: round(n / 2**20, 1)
# pyramid under no_grad: the retained outputs and the largest block outputs
outs = []
hooks = [mod.register_forward_hook(lambda mod, i, o, outs=outs: outs.append(o.numel() * o.element_size() if torch.is_tensor(o) else 0)) for mod in m.pyr.modules() if isinstance(mod, torch.nn.Conv2d)]
t0 = time.time()
with torch.no_grad():
    P = m.pyr(b["x"].float())
print("pyramid fwd s (bs 2, cpu)", round(time.time() - t0, 2))
for h in hooks: h.remove()
P_bytes = {k: mb(v.numel() * v.element_size()) for k, v in P.items()}
print("retained P (MB, bs 2):", P_bytes, "sum", sum(P_bytes.values()))
print("largest conv outputs (MB, bs 2):", sorted([mb(o) for o in outs])[-6:])
# node head with grad: unique saved-tensor bytes
seen, total = set(), [0]
def pack(t):
    st = t.untyped_storage()
    key = (st.data_ptr(), st.nbytes())
    if key not in seen:
        seen.add(key); total[0] += st.nbytes()
    return t
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    no = m.node_stage(P)
print("node head + mnms saved for backward (MB, bs 2):", mb(total[0]), "-> bs 8:", mb(total[0] * 4))
tot2 = total[0]
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    batch = LL.live_batch(no, b["table"], b["token"], b["row"], 512)
print("live_batch extra saved (MB, bs 2):", mb(total[0] - tot2), "label_s", round(batch["label_s"], 3), "valid per frame", batch["valid"].sum(1).tolist())
n_nodes_out = sum(v.numel() * v.element_size() for v in no.values())
print("no outputs (MB, bs 2):", mb(n_nodes_out))
print("params MB: pyr", mb(sum(p.numel() * 4 for p in m.pyr.parameters())), "nodes", mb(sum(p.numel() * 4 for p in m.nodes.parameters())), "all", mb(sum(p.numel() * 4 for p in m.parameters())), "n all", sum(p.numel() for p in m.parameters()))
print("nodes params", sum(p.numel() for p in m.nodes.parameters()))
# the graph reaches the node head: gradient size / smallest nonzero
import jtoken as JK
from unittest import mock
head = JK.JTokenHead()
with mock.patch.object(JK, "_const", lambda t: t):
    out = head(batch)
    loss, info = JK.jtoken_loss(out, batch)
loss.backward()
gn = {k: float(p.grad.abs().sum()) for k, p in src.named_parameters() if p.requires_grad and p.grad is not None}
print("loss", float(loss), "node-head params with grad", len(gn), "of", len(src.trainable()), "sum|grad| range", min(gn.values()), max(gn.values()))
# gradient on tok: smallest nonzero magnitude vs float16 range
gtok = no["tok"].grad if no["tok"].grad is not None else None
print("tok leaf grad retained?", gtok is not None)
