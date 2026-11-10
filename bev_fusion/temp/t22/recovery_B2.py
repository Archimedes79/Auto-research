# scratch: share of the frozen stand-in's loss that the fine-tune recovered, read from metrics.json (not typed)
import json, os
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, p + k + "/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool): o[p + k] = v
    return o
def load(r): return flat(json.load(open(os.path.join(r, "eval", "metrics.json")))["metrics"])
C = load("runs/t21_pair4_s0")
runs = {"standin_cur": load("runs/t22_standin_pair4"), "B": load("runs/t22_B_s0"), "standin_all": load("runs/t22_standin_all_pair4"), "B2": load("runs/t22_B2_s0")}
rows = [("lane divider IoU@1", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1"),
        ("drivable IoU@0", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"),
        ("mAP fusion area (detf)", "detf/dt+0.00/mean/all/mAP") ]
ks = [k for k in C if k.endswith("mean/all/mAP") and k.startswith("detf/dt+0.00")]
print(ks[:3])
for name, k in rows:
    if k not in C:
        print("missing", k); continue
    c = C[k]
    print(f"{name:26s} control {c:.4f}", end="")
    for n, m in runs.items(): print(f"  {n} {m[k]:.4f}", end="")
    print()
    for a, b in (("standin_cur", "B"), ("standin_all", "B2")):
        loss = runs[a][k] - c; rec = runs[b][k] - runs[a][k]
        print(f"    {a}->{b}: loss vs control {loss:+.4f}, fine-tune change {rec:+.4f}, recovered {100*rec/-loss:.0f} %")
