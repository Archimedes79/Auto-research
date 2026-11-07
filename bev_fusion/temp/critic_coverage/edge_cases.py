import numpy as np
from autogt import coverage as C
box = np.array([[10.0, -1], [12, -1], [12, 1], [10, 1]])
def b(pts, cam, z=(0.0, 1.5), fp=box):
    return C.blocked(np.array(pts, float), np.array(cam, float), [(fp, z[0], z[1])]).tolist()
cam = (0, 0, 1.6)
print("through the box, z inside        ", b([(20, 0, 0)], cam), "want True")
print("over the box (z const 1.6>1.5)   ", b([(20, 0, 1.6)], cam), "want False")
print("end inside footprint             ", b([(11, 0, 0)], cam), "want False")
print("end on near edge x=10            ", b([(10, 0, 0)], cam), "want False")
print("end just behind far edge         ", b([(12.001, 0, 0)], cam), "want True")
print("camera inside the box            ", b([(20, 0, 0)], (11, 0, 0.5)), "want True")
print("box behind the camera            ", b([(20, 0, 0)], cam, fp=box - [20, 0]), "want False")
print("segment along edge y=1 (cam y=1) ", b([(20, 1, 0)], (0, 1, 1.6)), "ambiguous (old rule: hit)")
print("segment parallel outside y=1.001 ", b([(20, 1.001, 0)], (0, 1.001, 1.6)), "want False")
print("floating box z 2..3, ground pt   ", b([(20, 0, 0)], cam, z=(2.0, 3.0)), "want False")
print("floating box z 2..3, point z 3.5 ", b([(20, 0, 3.5)], cam, z=(2.0, 3.0)), "want True (z at x=10: 2.55)")
print("box below ground z -2..-1        ", b([(20, 0, 0)], cam, z=(-2.0, -1.0)), "want False")
print("CW and CCW corner order          ", b([(20, 0, 0)], cam, fp=box[::-1]), b([(20, 0, 0)], cam, fp=box), "want True True")
print("grazing corner (10,-1)           ", b([(20, -2, 0)], cam, z=(0, 1.5)), "line from (0,0) to (20,-2): y=-0.1x -> at x=10 y=-1 -> corner; ambiguous")
print("z-test: z range exactly touches  ", b([(20, 0, 0)], cam, z=(0.0, 0.64)), "z at far edge x=12 is 0.64 -> touching, want True (>=)")
# soft vs old for a tall box with the camera at various places
gx, gy = np.meshgrid(np.arange(0.25, 60, 0.5), np.arange(-25.75, 26, 0.5), indexing="ij")
from autogt.layers import occlusion
for camxy in [(1.5, 0.0), (5.0, 0.5), (11.0, 0.0), (11.7, 0.9)]:
    old = occlusion(gx, gy, np.array(camxy), [box]); new = C.hidden_cells(gx, gy, (camxy[0], camxy[1], 1.6), [(box, -100, 100)])
    print("cam", camxy, "old rule vs soft(inf height): differing cells", int((old != new).sum()), "old", int(old.sum()), "soft", int(new.sum()))
