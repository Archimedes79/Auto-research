import torch
sd = torch.load("../../runs/e284_h2_s0/model.pt", map_location="cpu")["state"]
tot = {}
for k, v in sd.items():
    if k.startswith("ends."):
        g = ".".join(k.split(".")[:2])
        tot[g] = tot.get(g, 0) + v.numel()
print({k: v for k, v in tot.items()}, "total", sum(tot.values()))
ck = torch.load("../../runs/e287_p2_s0/model.pt", map_location="cpu")
t2 = {}
for k, v in ck["state"].items():
    g = ".".join(k.split(".")[:2])
    t2[g] = t2.get(g, 0) + v.numel()
print(t2)
print(ck["args"])
