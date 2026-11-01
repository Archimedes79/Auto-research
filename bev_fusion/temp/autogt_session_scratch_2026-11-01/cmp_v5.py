"""scratch: the keyframes the v5 worktree wrote (temp/gt_v5_out) equal the stored v5 GT (data/autogt/15efbde2d1)"""
import glob, os, sys
import numpy as np
sys.path.insert(0, ".")
from autogt.cli import load_keyframe
a_dir, b_dir = "temp/gt_v5_out/15efbde2d1", "data/autogt/15efbde2d1"
for f in sorted(glob.glob(a_dir + "/*.npz")):
    t = os.path.basename(f)[:-4]
    a, b = load_keyframe(a_dir, t), load_keyframe(b_dir, t)
    assert all(np.array_equal(a[k], b[k]) for k in a if k != "meta"), t
    print("identical", t)
