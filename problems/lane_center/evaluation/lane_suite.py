"""The lane grid and configuration of the scorer (`qgraph_eval.py` reads `config()` and `B123`; `lane_task.measures` reads the settings of `LaneConfig`): ego frame, u = x to
the right, v = y forward, 200 x 200 cells of 0.5 m, evaluated |x|, |y| < 48 m like the project's metrics. In the evaluator hash.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from lane_center.training import bflc_paths as BP  # project paths from the location of this checkout
from lane_center.evaluation.lane_task import measures as M
from lane_center.evaluation.lane_task.raster import Grid


@dataclass
class LaneConfig:
    grid: Grid
    taus: Tuple[float, ...] = (0.5, 1.0, 1.5)       # tolerances in metres (the middle one is the main one)
    ds: float = 0.5                                 # spacing of the dense points of a graph
    box: Optional[Tuple[float, float, float, float]] = None   # sparse region (u0, u1, v0, v1); default = the grid
    thr: float = 0.5                                # threshold of the probability image
    sweep: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # ascending
    clip: float = 5.0                               # chamfer distances are clipped at this many metres
    found_frac: float = 0.8                         # a ground-truth lane is found when this share of it is within tau
    skeleton_min_len: float = 2.0
    skeleton_tol: float = 0.3
    render_width: float = 0.0                       # width of a predicted graph drawn as an image (0 = one cell)
    topo: Dict[str, float] = None                   # TOPO settings: ds, seed_step, L, snap, r, j_tol
    near_factor: float = 3.0                        # 'displaced' = within near_factor * tau
    far_m: float = 40.0                             # range from the origin beyond which a lane is 'far'
    origin: Tuple[float, float] = (0.0, 0.0)        # the ego position in the frame of the coordinates

    def __post_init__(self):
        if self.topo is None:
            self.topo = dict(ds=1.0, seed_step=5.0, L=30.0, snap=2.0, r=2.0, j_tol=3.0)
        if self.box is None:
            self.box = self.grid.extent

    @property
    def tau(self) -> float:
        return self.taus[min(1, len(self.taus) - 1)]

    @property
    def tk(self) -> str:
        return M.tkey(self.tau)


B123 = BP.B123
GRID = Grid(u0=-50.0, v0=-50.0, res=0.5, nu=200, nv=200)
BOX = (-48.0, 48.0, -48.0, 48.0)


def config() -> LaneConfig:
    return LaneConfig(GRID, taus=(0.5, 1.0, 1.5), ds=0.5, box=BOX, thr=0.5, clip=5.0, far_m=40.0, origin=(0.0, 0.0))
