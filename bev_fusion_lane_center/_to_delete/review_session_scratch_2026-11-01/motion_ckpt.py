import sys, torch
torch.set_num_threads(2)
for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    print("==", path, "step", ck["step"])
    for src in ("state", "ema"):
        sd = ck[src]
        for k, v in sd.items():
            if "motion" in k:
                print(f"  {src:5s} {k:45s} shape {tuple(v.shape)} absmax {v.abs().max().item():.3e} mean|.| {v.abs().mean().item():.3e}")
    # optimizer state for motion params: find via order is hard; print loss.task.s
    for k, v in ck["state"].items():
        if k.startswith("loss."):
            print("  ", k, v.tolist() if v.numel() < 8 else v.shape)
