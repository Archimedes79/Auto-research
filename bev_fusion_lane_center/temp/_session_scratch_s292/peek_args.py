import os, torch
ck = torch.load(os.environ["BFLC_ROOT"] + "/runs/e273_p_s0/model.pt", map_location="cpu")
print({k: v for k, v in ck["args"].items()}); print(ck.get("steps"))
