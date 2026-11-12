import sys, os
for p in ("training/train", "training/grid", "training/data", "evaluation", "training"):
    sys.path.insert(0, os.path.join("/mnt/c/Auto-research/bev_fusion_lane_center", p))
import torch, qgraph as QG, lane_gt_data as LD
m = QG.QGraph().cuda()
sd = torch.load("/mnt/c/Auto-research/bev_fusion_lane_center/runs/e273_p_s0_p2/model.pt", map_location="cpu")["state"]
m.load_state_dict({k: v for k, v in sd.items() if k.startswith("pyr.")}, strict=False)
ds, dl = LD.loader("train", bs=4, workers=0, shuffle=True)
b = next(iter(dl))
with torch.no_grad():
    out = m(b, upto="links", frozen=("pyr", "nodes", "links"), gt=True)
L = out["L"]
n_gt = int((b["lk"][..., 0] >= 0).sum())
print("GT links", n_gt, "pairs", len(L["pb"]), "labelled (one-to-many)", int(L["label"].sum()), "gt_missing", L["gt_missing"], "long pairs", int(L["is_long"].sum()))
loss, info = QG.qgraph_loss(m, b, 4)
print(round(float(loss), 3), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()})
