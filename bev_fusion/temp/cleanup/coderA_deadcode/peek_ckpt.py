import sys
import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(type(ck), list(ck.keys()) if isinstance(ck, dict) else "")
for k, v in ck.items():
    if isinstance(v, dict):
        print(k, "dict", len(v), list(v.keys())[:4])
    else:
        print(k, type(v), (v if isinstance(v, (int, float, str)) else ""))
