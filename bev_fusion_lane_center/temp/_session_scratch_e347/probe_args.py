import torch, os, sys
R = os.environ["BFLC_ROOT"]
for r in ("e307_n_s0", "e332_ma_s0", "e312_p_s0", "e341_p_s0"):
    ck = torch.load(os.path.join(R, "runs", r, "model.pt"), map_location="cpu")
    a = ck["args"]
    print(r, {k: a[k] for k in a if k in ("ctx", "d", "dirx", "no_paint", "no_zone", "raster", "sigma", "far_w", "joint", "maps_args", "scales", "e2e", "name", "steps", "lr", "bs")}, len(ck["state"]))
