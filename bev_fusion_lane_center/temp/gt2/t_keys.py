import torch, collections
sd = torch.load("/mnt/c/Auto-research/bev_fusion_lane_center/runs/e273_p_s0/model.pt", map_location="cpu")
st = sd["state"]
c = collections.Counter(k.split(".")[0] for k in st)
print(c)
print([k for k in st if k.startswith("nodes.")][:6])
print({k: v for k, v in sd["args"].items() if k in ("width", "d", "f05", "seed")} if isinstance(sd.get("args"), dict) else sd.get("args"))
