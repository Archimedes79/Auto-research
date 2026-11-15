"""The BEV grid every GtGenerator layer is defined on: x forward, y left, the ego frame at the frame time; arrays are [ix, iy]."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    x_min: float = 0.0
    x_max: float = 60.0
    y_min: float = -26.0
    y_max: float = 26.0
    res: float = 0.5                          # metres per cell (0.5 or 0.25)

    @property
    def nx(self):
        return int(round((self.x_max - self.x_min) / self.res))

    @property
    def ny(self):
        return int(round((self.y_max - self.y_min) / self.res))

    def centres(self):
        """(gx, gy): cell-centre coordinates, both (nx, ny)."""
        xs = self.x_min + (np.arange(self.nx) + 0.5) * self.res
        ys = self.y_min + (np.arange(self.ny) + 0.5) * self.res
        return np.meshgrid(xs, ys, indexing="ij")
