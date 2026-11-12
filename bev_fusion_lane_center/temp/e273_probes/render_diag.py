import sys, os
for p in ("tools/viz", "training/train", "training/grid", "training/data", "evaluation", "training"):
    sys.path.insert(0, os.path.join("/mnt/c/Auto-research/bev_fusion_lane_center", p))
import torch, qgraph_view as QV, qpredict as QP, qgates as G
run, scene, idx = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3:]]
torch.set_num_threads(1)
fn = QP.make_pred_fn(run, "exact", "cpu", keep_maps=True)
_, name, toks = QV.resolve_scene(scene)
out = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e273_view/diag"
os.makedirs(out, exist_ok=True)
for i in idx:
    t = toks[i]
    p = G.component_sections(fn(t))
    QV.render_frame(t, p, f"{out}/{run}_{name}_{i}.png", cams=False, g=QV.load_gt(t))
    print(f"{out}/{run}_{name}_{i}.png")
