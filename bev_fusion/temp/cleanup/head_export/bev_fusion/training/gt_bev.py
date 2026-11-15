"""Compatibility shim: the GT generator moved into the standalone suite `bev_fusion/autogt/` (2026-10-30).

Everything that used to live here is re-exported unchanged (NuScenesGT, Raster, AREAS / LINES / OBJECTS, cell_centres,
occlusion, pose_matrix, category_group, radar_coverage, ...); new code should `import autogt` directly. Self-test of the whole
suite: `python -m autogt selftest` (from the bev_fusion folder); `python training/gt_bev.py` runs it as well.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
while not os.path.isdir(os.path.join(_ROOT, "autogt")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)      # a run snapshot (runs/<name>/code/) sits three levels below the bev_fusion folder
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from autogt.layers import AREAS, DIST_MAX, LINES, OBJECTS, SHIFT, Raster, cell_centres, inside_quad as _inside, occlusion  # noqa: E402,F401
from autogt.nuscenes_adapter import category_group, pose_matrix  # noqa: E402,F401
from autogt.nuscenes_gt import RADAR_FOV, NuScenesGT, radar_coverage  # noqa: E402,F401

if __name__ == "__main__":
    from autogt.selftest import run
    run()
