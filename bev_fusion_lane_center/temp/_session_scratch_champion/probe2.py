import torch, collections
for p in ["runs/e324_s_s0/model.pt", "runs/e273_p_s0/model.pt"]:
    ck = torch.load("/mnt/c/Auto-research/bev_fusion_lane_center/" + p, map_location="cpu", weights_only=False)
    print(p, list(ck.keys()))
    a = ck.get("args")
    print({k: v for k, v in (a or {}).items() if k != "maps_args"})
    if a and "maps_args" in a: print("maps_args", a["maps_args"])
    st = ck["state"]
    grp = collections.OrderedDict()
    for k, v in st.items():
        g = ".".join(k.split(".")[:2])
        grp.setdefault(g, [0, 0]); grp[g][0] += 1; grp[g][1] += v.numel()
    for g, (n, m) in grp.items(): print("  ", g, n, m)
    print("total", sum(v.numel() for v in st.values()), [ (k, v.dtype) for k, v in list(st.items())[:2]])
