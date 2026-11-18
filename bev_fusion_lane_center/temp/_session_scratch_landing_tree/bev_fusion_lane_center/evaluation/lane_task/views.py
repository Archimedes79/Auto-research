"""The pictures of a lane frame as a Page (no plotting code): ground truth, prediction, the errors on the lines and on the image,
the error budget. Error colours as everywhere: grey right, RED false, BLUE missed, cream not evaluated."""
from __future__ import annotations

from typing import List

import numpy as np

from Evaluation.contracts import Page, Panel, Prediction, Sample, Shape, View, ViewContext
from Evaluation.render.figure import ERROR_LEGEND, FN_COLOR, FP_COLOR, SKIP_COLOR, TP_COLOR
from .geometry import Dense
from .measures import nn_dist
from .raster import edt
from .task import LaneConfig, lane_data, lane_probe


def status_shapes(dense: Dense, ok: np.ndarray, good, bad, lw_good=1.2, lw_bad=2.6) -> List[Shape]:
    """lines coloured by a per-point status: contiguous runs of the same status on one polyline become one Shape"""
    out: List[Shape] = []
    for ln in np.unique(dense.line):
        idx = np.flatnonzero(dense.line == ln)
        idx = idx[np.argsort(dense.s[idx])]
        st = ok[idx]
        a = 0
        for b in range(1, len(idx) + 1):
            if b == len(idx) or st[b] != st[a]:
                seg = dense.xy[idx[max(a - 1, 0):min(b + 1, len(idx))]]       # one point of overlap keeps the line closed
                if len(seg) >= 2:
                    out.append(Shape("line", seg, good if st[a] else bad, lw=lw_good if st[a] else lw_bad))
                a = b
    return out


def graph_shapes(g, color, lw=1.4, vertices=True, arrows=True, ds=15.0) -> List[Shape]:
    out: List[Shape] = []
    for p in g.lines:
        if len(p) >= 2:
            out.append(Shape("line", p, color, lw=lw))
    if vertices:
        pts = [p for p in g.lines if len(p)]
        if pts:
            out.append(Shape("points", np.concatenate(pts), "k", lw=0.25))
    if arrows and g.directed:
        d = g.dense(ds)
        for i in range(len(d)):
            if d.s[i] > 1.0:
                t = d.tan[i] * 1.6
                out.append(Shape("arrow", [d.xy[i, 0], d.xy[i, 1], d.xy[i, 0] + t[0], d.xy[i, 1] + t[1]], color, lw=1.4))
    return out


class LaneOverviewView(View):
    name = "overview"
    description = "ground truth, prediction (dots = vertices), errors on the lines and on the image, error budget of the frame"

    def __init__(self, cfg: LaneConfig):
        self.cfg = cfg

    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:
        c, d = self.cfg, lane_data(self.cfg, sample, pred)
        ext = c.grid.extent
        xl, yl = (ext[0], ext[1]), (ext[2], ext[3])
        tau = c.tau
        dg, dp = d.gt.dense(c.ds, c.box), d.pred.dense(c.ds, c.box)
        gt_ok = nn_dist(dg.xy, dp.xy) <= tau if len(dg) else np.zeros(0, bool)
        pr_ok = nn_dist(dp.xy, dg.xy) <= tau if len(dp) else np.zeros(0, bool)
        panels: List[Panel] = []
        cam = sample.get("camera")
        if cam is not None:
            panels.append(Panel("image", "camera", cam, note="what the network sees"))
        panels.append(Panel("raster", "ground truth", d.mask.astype(np.float32), extent=ext, cmap="Greys", vmin=0, vmax=1, xlim=xl, ylim=yl,
                            shapes=graph_shapes(d.gt, "tab:green", 1.2, vertices=False, arrows=d.gt.directed),
                            note=f"{len(d.gt)} lanes, {d.gt.length():.0f} m (from the {d.gt_from})"))
        panels.append(Panel("raster", "prediction", d.prob, extent=ext, cmap="magma", vmin=0, vmax=1, xlim=xl, ylim=yl,
                            shapes=graph_shapes(d.pred, "tab:cyan", 1.2), colorbar=False,
                            note=f"{len(d.pred)} pieces, {d.pred.length():.0f} m, {d.pred.vertices()} vertices (graph from the {d.pred_from})"))
        panels.append(Panel("raster", f"errors on the lines (tau {tau:g} m)", np.zeros(c.grid.shape, np.float32), extent=ext, cmap="Greys", vmin=0, vmax=1,
                            xlim=xl, ylim=yl,
                            shapes=status_shapes(dg, gt_ok, TP_COLOR, FN_COLOR) + status_shapes(dp, pr_ok, "0.55", FP_COLOR),
                            note="blue = ground truth without prediction, red = prediction without ground truth, grey = right"))
        # the errors on the image: relaxed (cells within tau of the other set)
        pred_m = d.prob >= c.thr
        dg_img, dp_img = edt(d.mask, c.grid), edt(pred_m, c.grid)
        rgb = np.ones(d.mask.shape + (3,), np.float32)
        v = np.ones(d.mask.shape, bool) if d.valid is None else d.valid
        rgb[d.mask & (dp_img <= tau)] = TP_COLOR
        rgb[pred_m & (dg_img <= tau)] = TP_COLOR
        rgb[pred_m & (dg_img > tau)] = FP_COLOR
        rgb[d.mask & (dp_img > tau)] = FN_COLOR
        rgb[~v] = SKIP_COLOR
        panels.append(Panel("rgb", f"errors on the image (tolerant, {tau:g} m)", rgb, extent=ext, xlim=xl, ylim=yl,
                            note="cells of the thresholded picture; the sparse errors above are read from the same prediction"))
        pr = lane_probe(c, d)
        bud = {k.split(".", 1)[1] + (" (miss)" if k.startswith("cause.fn") else " (false)"): v for k, v in pr.items()
               if k.startswith("cause.") and v > 0}
        if bud:
            panels.append(Panel("bars", "error budget: metres per cause", dict(sorted(bud.items(), key=lambda kv: kv[1]))))
        tags = [f"{k[4:]:14s} {v:.3g}" for k, v in pr.items() if k.startswith("tag.")]
        flags = [f"{f.status}: {f.name}" for f in ctx.flags if f.status != "ok"]
        panels.append(Panel("text", "situation of this frame (tags) and flags", "\n".join(tags + ([""] + flags if flags else []))))
        return Page(title=f"lane centre lines, tolerance {tau:g} m", panels=panels, ncols=3, legend=ERROR_LEGEND, panel_size=(4.3, 4.3))
