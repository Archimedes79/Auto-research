"""The lane-centre-line Task of the Evaluation (`../../../autoresearch/Evaluation`): evaluate a dense head's picture first, then the lane graph
read from it (or a graph head's output), with the common metrics, and say where the errors come from. numpy + scipy (+ matplotlib for the
views); no torch. Lane-specific, so it lives in the lane project (moved out of `autoresearch/Evaluation/lanes/` on 2026-10-31, decision D24).

    from lane_task import LaneConfig, Grid, LaneTask, LaneGraph
    cfg = LaneConfig(Grid(u0=-50, v0=-50, res=0.5, nu=200, nv=200))
    task = LaneTask(cfg)            # metrics img / geo / topo, view 'overview', checks, probe tags and causes

An adapter only has to deliver (Sample.reference, Prediction.outputs) as described in `task.py`. Files: geometry (LaneGraph, resample,
simplify), raster (Grid, rasterize, thin, skeleton_graph), measures (additive counts + formulas), causes (error budget), task, views,
synth (the synthetic lane world: test bed and porting template). Tests: `../tests/test_lane_task.py`.
"""
import os as _os
import sys as _sys

_AR = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, _os.pardir, _os.pardir, "autoresearch"))
if _AR not in _sys.path:                       # the Evaluation core (problem-independent) sits in the repository's autoresearch/
    _sys.path.append(_AR)

from .geometry import LaneGraph, resample, simplify  # noqa: E402
from .raster import Grid, rasterize, skeleton_graph, thin  # noqa: E402
from .task import LaneConfig, LaneData, LaneImageMetric, LaneSparseMetric, LaneTask, LaneTopoMetric, lane_data  # noqa: E402
