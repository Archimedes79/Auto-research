"""One-off: look into models/pretrained/bevfusion/camera-only-seg.pth WITHOUT executing anything from it: (1) the classes its pickle refers
to (pickletools, no unpickling), (2) torch.load(weights_only=True), (3) key groups and shapes of the state dict.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python models/inspect_bevfusion_ckpt.py"""
import collections
import pickletools
import zipfile

import torch

P = "models/pretrained/bevfusion/camera-only-seg.pth"
z = zipfile.ZipFile(P)
names = z.namelist()
print(len(names), "files in the archive; first:", names[:3])
pkl = [n for n in names if n.endswith("data.pkl")][0]
strings, globs = [], set()
for op, arg, _ in pickletools.genops(z.read(pkl)):
    if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
        strings.append(arg)
    if op.name == "GLOBAL":
        globs.add(arg.replace(" ", "."))
    if op.name == "STACK_GLOBAL" and len(strings) >= 2:
        globs.add(f"{strings[-2]}.{strings[-1]}")
print("classes referred to by the pickle:", sorted(globs))
try:
    ck = torch.load(P, map_location="cpu", weights_only=True)
    print("weights_only load OK; top-level keys:", list(ck.keys()))
except Exception as e:                                                # noqa: BLE001
    print("weights_only load FAILED:", str(e)[:400])
    raise SystemExit(1)
sd = ck.get("state_dict", ck)
print(len(sd), "tensors,", sum(v.numel() for v in sd.values()) / 1e6, "M values")
groups = collections.OrderedDict()
for k, v in sd.items():
    g = ".".join(k.split(".")[:3])
    n, c = groups.get(g, (0, 0))
    groups[g] = (n + 1, c + v.numel())
for g, (n, c) in groups.items():
    print(f"  {g:60s} {n:4d} tensors {c / 1e6:7.2f} M")
for k in list(sd)[:6]:
    print(k, tuple(sd[k].shape))
for k, v in sd.items():
    if any(s in k for s in ("depthnet", "dtransform", "vtransform")) and ("weight" in k) and v.ndim == 4:
        print(k, tuple(v.shape))
