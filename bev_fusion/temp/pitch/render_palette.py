import sys

import numpy as np

sys.path[:0] = [".", "training"]
import evaluation.bev_gtcheck as G
from evalsuite import Page
from evalsuite.render.figure import render_page
from evaluation import bev_views as V
from lift_pd import Grid
from losses import NUSC_GROUPS, layer_spec

grid, spec = Grid(), layer_spec(False, True)
groups = [g for g, _ in NUSC_GROUPS]
k = V.layer_index(spec)
frames = G.pick(int(sys.argv[2]) if len(sys.argv) > 2 else 6, G.resolve_sig(None), 3)
name, s = frames[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
g0 = s[0.0]
prob = {n: np.asarray(g0[k[n][2]], np.float32) for n, (i, kind, key) in k.items() if kind in ("area", "object") and k[n][2] in g0}
dist = {n: np.asarray(g0[n + "_dist"], np.float32) for n in ("lane_divider", "road_divider") if n + "_dist" in g0}
obs = V.observed_area(g0)
gtb = V.gt_boxes(g0, groups)
dets = [dict(b, score=0.9, cx=b["cx"] + 0.4) for b in gtb[::2]]
panels = [V.bev(grid, "rgb", "ground truth now", V.composite(prob, dist, observed=obs), V.box_shapes(gtb, V.GT_BOX_COLOR, observed=obs)),
          V.bev(grid, "rgb", "prediction now (amber boxes >= 0.3, dashed = GT)", V.composite(prob, dist, observed=obs),
                V.box_shapes(gtb, V.GT_OVERLAY, "--", 0.7) + V.box_shapes(dets, V.DET_COLOR))]
render_page(Page(title="palette check " + name, panels=panels, ncols=2, panel_size=(7.5, 7.0)), "temp/pitch/palette_check.png", dpi=110)
print("written")
