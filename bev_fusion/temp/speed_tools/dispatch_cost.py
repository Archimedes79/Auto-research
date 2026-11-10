"""throw-away (2026-11-03): what does ONE small torch operation cost on this box (WSL2, RTX 3060)? Splits the ~85 microseconds per operator of the t20 step
(4.09 s / 46 thousand operators, bev_fusion/records/speed_options_2026-11-03.md) into the part every eager PyTorch pays on the host (Python + dispatcher + autograd) and the part
the CUDA path under WSL2 adds (kernel launch, device-to-host sync). One thread, tiny tensors, no data, no repo code. Usage: python dispatch_cost.py cpu | cuda"""
import sys
import time

import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.set_num_threads(1)
N = 3000  # operators per chain
REPS = 5


def chain(x, w):
    """N leaf operators (add, mul, relu, sigmoid in turn) that build an autograd graph"""
    for i in range(N // 4):
        x = x + w
        x = x * w
        x = torch.relu(x)
        x = torch.sigmoid(x)
    return x


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def best(fn):
    out = []
    for _ in range(REPS):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t)
    return min(out)


x = torch.randn(64, 64, device=dev, requires_grad=True)
w = torch.randn(64, 64, device=dev, requires_grad=True)
chain(x, w).sum().backward()  # warm up (cuda context, cublas, allocator)
sync()
res = {}
with torch.no_grad():
    res["forward, no grad (host + launch + gpu wait at the end)"] = best(lambda: chain(x, w)) / N
res["forward, autograd recording"] = best(lambda: chain(x, w)) / N
y = chain(x, w).sum()
res["backward per node"] = best(lambda: (chain(x, w).sum().backward())) / N - res["forward, autograd recording"]
if dev == "cuda":
    # enqueue time only (no wait at the end): the host cost of a launch
    t = time.perf_counter()
    with torch.no_grad():
        chain(x, w)
    t_enq = time.perf_counter() - t
    torch.cuda.synchronize()
    res["forward, no grad, ENQUEUE only (host cost per launch)"] = t_enq / N
    # a device-to-host round trip on an empty queue
    z = torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(300):
        z.item()
    res[".item() on an empty queue"] = (time.perf_counter() - t) / 300
    t = time.perf_counter()
    for _ in range(300):
        torch.cuda.synchronize()
    res["synchronize() on an empty queue"] = (time.perf_counter() - t) / 300
    # a pageable host-to-device copy of a small tensor (the 1,300 GT copies per step)
    h = torch.randn(64, 64)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(300):
        h.to(dev)
    torch.cuda.synchronize()
    res["pageable h2d copy 16 KB (incl. sync at the end)"] = (time.perf_counter() - t) / 300
    # a boolean-mask index (nonzero -> sync inside)
    m = torch.rand(64, 64, device=dev) > 0.5
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(300):
        x[m]
    torch.cuda.synchronize()
    res["bool-mask index x[m] (implicit sync)"] = (time.perf_counter() - t) / 300
print(f"device {dev}, torch {torch.__version__}, one thread, {N} operators per chain, best of {REPS}")
for k, v in res.items():
    print(f"  {k:62s} {v * 1e6:8.1f} us")
