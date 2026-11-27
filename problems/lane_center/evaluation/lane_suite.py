"""The lane grid and configuration of the scorer (`qgraph_eval.py` reads `config()` and `B123`): ego frame, u = x to the right, v = y forward, 200 x 200 cells of
0.5 m, evaluated |x|, |y| < 48 m like the project's metrics. The old lane suite around it (the dense and registry subjects on the general evaluation suite, its command
line and selftest) is in the tag v1.
"""
import bflc_paths as BP  # project paths from the location of this checkout
from lane_task import Grid, LaneConfig  # the lane Task, evaluation/lane_task/

B123 = BP.B123
GRID = Grid(u0=-50.0, v0=-50.0, res=0.5, nu=200, nv=200)
BOX = (-48.0, 48.0, -48.0, 48.0)


def config() -> LaneConfig:
    return LaneConfig(GRID, taus=(0.5, 1.0, 1.5), ds=0.5, box=BOX, thr=0.5, clip=5.0, far_m=40.0, origin=(0.0, 0.0))
