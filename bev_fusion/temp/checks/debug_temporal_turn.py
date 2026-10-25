"""One-off: per-frame memory-read quality of training/temporal_fusion.py while turning (corr, coverage)."""
import sys

sys.path.insert(0, "training")
from lift_pd import Grid  # noqa: E402
from temporal_fusion import TemporalFusion, _drive  # noqa: E402

grid = Grid(x_min=0.0, x_max=24.0, y_min=-8.0, y_max=8.0, res=0.5)
for v, om, n in [(8.0, 0.5, 36), (8.0, 0.25, 36), (10.0, 0.0, 36)]:
    mod = TemporalFusion(2, 8, grid, mem_cells=(128, 96)).eval()
    log = _drive(mod, grid, v, om, n)
    print(f"v={v} omega={om}: frame corr/coverage:", " ".join(f"{i}:{c:.2f}/{cv:.2f}" for i, (c, cv, _) in enumerate(log) if i % 3 == 2))
