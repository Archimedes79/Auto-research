"""scratch: vectorised single-linkage groups of ends (the O(n^2) python loop of jgt._groups) in numpy, float32 like the original; tested against jgt._groups on real keypoints and random sets."""
import os, sys, time, pickle
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jgt as JG

ARM_ANG, ARM_ALONG, ARM_ACROSS = JG.ARM_ANG, JG.ARM_ALONG, JG.ARM_ACROSS


def groups_vec(E):
    """E (n, 4) float32 -> labels (n,) int, the same labels as `jgt._groups` (a component is numbered by the order of its first member)"""
    n = len(E)
    P, T = E[:, :2], E[:, 2:]
    ti, tj = T[:, None, :], T[None, :, :]
    cos = ti[..., 0] * tj[..., 0] + ti[..., 1] * tj[..., 1]
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    s = ti + tj
    sn = s + 1e-9
    t = s / np.sqrt(sn[..., 0] * sn[..., 0] + sn[..., 1] * sn[..., 1])[..., None]
    d = P[None, :, :] - P[:, None, :]                                    # d[i, j] = E[j, :2] - E[i, :2]
    along = np.abs(d[..., 0] * t[..., 0] + d[..., 1] * t[..., 1])
    across = np.abs(d[..., 0] * (-t[..., 1]) + d[..., 1] * t[..., 0])
    A = (ang <= ARM_ANG) & (along <= ARM_ALONG) & (across <= ARM_ACROSS)
    A |= A.T
    A[np.arange(n), np.arange(n)] = True
    R = A.astype(np.float32)
    for _ in range(5):                                                   # path length up to 32
        R = ((R @ R) > 0).astype(np.float32)
    first = R.argmax(1)                                                  # the smallest index of the component
    return np.unique(first, return_inverse=True)[1].astype(int)


def make_real():
    out = []
    for run in ("e312_p_s0", "e312_p_s1"):
        fr = pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))
        out.append(fr)
    return out


if __name__ == "__main__":
    fr = pickle.load(open(os.path.join(ROOT, "runs", "e312_p_s0", "ends_val.pkl"), "rb"))
    print(type(fr), len(fr))
    f0 = fr[0] if isinstance(fr, list) else fr[list(fr)[0]]
    print(type(f0), list(f0.keys()) if isinstance(f0, dict) else "")
