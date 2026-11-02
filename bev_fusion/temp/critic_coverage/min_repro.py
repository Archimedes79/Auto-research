import numpy as np
from autogt import coverage as C
fp = np.array([[3.0, 1.2], [3.0, 3.7], [-5.0, 3.7], [-5.0, 1.2]])          # a truck beside the camera: 8 m long, 2.5 m wide, front 3 m ahead
prism = (fp, 0.0, 3.5)
cam = (0.0, 0.0, 1.5)
pts = np.array([[20.0, 8.1, 0.0], [20.0, 9.0, 0.0], [20.0, 8.0, 0.0]])
print("with the pre-filter   :", C.blocked(pts, cam, [prism]).tolist())
orig = C._wedge; C._wedge = lambda c, p, f: np.ones(len(p), bool)
print("without the pre-filter:", C.blocked(pts, cam, [prism]).tolist())
C._wedge = orig
from autogt.layers import occlusion
print("old 2D rule           :", occlusion(pts[:, 0:1], pts[:, 1:2], np.array(cam[:2]), [fp]).ravel().tolist())
