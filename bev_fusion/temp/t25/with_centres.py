"""scratch: run a bev_fusion script with lift_pd.Grid.centres() added (GtGenerator v13's cell_centres calls grid.centres(); the model's Grid has only xy_centres,
so every path through GtGenerator with that Grid fails at HEAD, independent of the t25 change). Usage: with_centres.py <script> [args]"""
import os
import runpy
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
import lift_pd  # noqa: E402


def centres(self):
    xs = self.x_min + (np.arange(self.nx) + 0.5) * self.res
    ys = self.y_min + (np.arange(self.ny) + 0.5) * self.res
    return np.meshgrid(xs, ys, indexing="ij")


lift_pd.Grid.centres = centres
script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
