"""Lane centre lines for the evalsuite: evaluate a dense head's picture first, then the lane graph read from it (or a graph head's
output), with the common metrics, and say where the errors come from. numpy + scipy (+ matplotlib for the views); no torch.

    from evalsuite.lanes import LaneConfig, Grid, LaneTask, LaneGraph
    cfg = LaneConfig(Grid(u0=-50, v0=-50, res=0.5, nu=200, nv=200))
    task = LaneTask(cfg)            # metrics img / geo / topo, view 'overview', checks, probe tags and causes

An adapter only has to deliver (Sample.reference, Prediction.outputs) as described in `task.py`. Files: geometry (LaneGraph, resample,
simplify), raster (Grid, rasterize, thin, skeleton_graph), measures (additive counts + formulas), causes (error budget), task, views.
"""
from .geometry import LaneGraph, resample, simplify
from .raster import Grid, rasterize, skeleton_graph, thin
from .task import LaneConfig, LaneData, LaneImageMetric, LaneSparseMetric, LaneTask, LaneTopoMetric, lane_data
