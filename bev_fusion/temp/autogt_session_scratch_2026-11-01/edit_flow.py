import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
# imports and GridSpec
s = s.replace("import math\nfrom dataclasses import dataclass, field\n\nimport numpy as np\n",
              "import math\nfrom dataclasses import dataclass, field\n\nimport numpy as np\n\nfrom .grid import GridSpec\nfrom .layers import inside_quad as _inside_quad\n", 1)
a = s.index("@dataclass\nclass GridSpec:")
b = s.index("@dataclass\nclass Config:")
s = s[:a] + s[b:]
# Track.extra
s = s.replace("    stationary: bool = None           # None = decide from the track\n",
              "    stationary: bool = None           # None = decide from the track\n    extra: dict = field(default_factory=dict)   # free adapter data (e.g. annotation tokens, 3D positions)\n", 1)
# public names
s = s.replace("_ego_velocity(", "ego_velocity(").replace("def _is_stationary(", "def is_stationary(").replace("_is_stationary(", "is_stationary(")
s = s.replace("def ego_velocity(ego, t, h):", "def ego_velocity(ego, t, h=0.09):")
# the old private quad test is replaced by layers.inside_quad
a = s.index("def _inside_quad(P, quad):")
b = s.index("def radar_points_in_box(")
s = s[:a] + s[b:]
# ego flow field as a public function
s = s.replace("    fg.flow_ego = np.stack([-vx + om * gy, -vy - om * gx]).astype(np.float32)          # -(v + omega x r)",
              "    fg.flow_ego = ego_flow_field(grid, fg.ego)")
s = s.replace("def _rasterise(fg, grid, pose):", '''def ego_flow_field(grid, ego_v):
    """apparent velocity (2, nx, ny) m/s of static world points in the ego frame: -(v + omega x r), ego_v = (vx, vy, omega)."""
    vx, vy, om = ego_v
    gx, gy = grid.centres()
    return np.stack([-vx + om * gy, -vy - om * gx]).astype(np.float32)


def _rasterise(fg, grid, pose):''', 1)
s = s.replace("Self-test (synthetic world with known truth): python autogt/flow.py",
              "Self-test (synthetic world with known truth): python -m autogt.flow  (all tests of the suite: python -m autogt selftest)")
open(p, "w", encoding="utf-8").write(s)
print("ok")
