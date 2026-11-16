import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "training/grid")
import torch, jtoken as JK
torch.set_num_threads(2)
b = JK.synthetic_batch(8, 48, 0)
m = JK.JTokenHead()
print("params", JK.count_params(m))
out = m(b)
loss, info = JK.jtoken_loss(out, b)
print(float(loss), info)
loss.backward()
print([n for n, p in m.named_parameters() if p.grad is None])
print(b["y"].sum().item(), b["ylong"].sum().item())
