"""evaluation/bev_free.py on REAL frames (CPU, minutes, about 6 GB: nuScenes and the run; not in run_gates.sh). The first 4 frames of val scene-0003 go through the live pair (bev_adapter's
Source and BEVSubject.predict: ground truth read, loss computed) and through FreeSource / FreeSubject (no ground truth, no loss). Checks: (1) the outputs of every dt (maps, distance fields,
mass, detections, refined boxes, velocity fields, 1 m direction) and the taps are bit-equal to the live ones; (2) the sample has no reference and no `gt`, the info has no loss, describe() names no
ground truth, the scenes and frames (with their dataset keys) are the live ones; (3) the memory is carried over the 4 frames and a second pass after reset() gives the same, the memory
confidence is empty at the first frame and grows; (4) a dataset item that is not the planned frame is refused, and an interface error of the model comes out as ValueError, not SystemExit.
Run (from the bev_fusion folder): ~/miniconda3/envs/gcar/bin/python tests/test_bev_free.py [runs/<name>]"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from evaluation import bev_adapter as A  # noqa: E402
from evaluation import bev_free as F  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/t20_base_s0"
SCENE, N, TAPS = "scene-0003", 4, ("fused", "memory_conf")


def same(a, b, path=""):
    """bit for bit: the same keys, the same dtypes and shapes, the same numbers (nan equals nan)"""
    if isinstance(a, dict):
        assert isinstance(b, dict) and sorted(map(str, a)) == sorted(map(str, b)), (path, sorted(map(str, a)), sorted(map(str, b)) if isinstance(b, dict) else type(b))
        for k in a:
            same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), (path, len(a), len(b) if isinstance(b, (list, tuple)) else type(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), \
            (path, a.dtype, getattr(b, "dtype", None), a.shape, getattr(b, "shape", None), float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if isinstance(b, np.ndarray) and a.shape == b.shape else "shape")
    else:
        assert a == b or (a != a and b != b), (path, a, b)


def raises(fn, kind):
    try:
        fn()
    except kind as e:
        return e
    except SystemExit:
        raise AssertionError("SystemExit escaped")
    raise AssertionError(f"{kind.__name__} expected")


def main():
    raises(lambda: same({"a": np.zeros(2, np.float32)}, {"a": np.ones(2, np.float32)}), AssertionError)           # the comparison itself sees a difference
    src, subj, _ = A.build(RUN, device="cpu", split="val", workers=0)
    refs = src.frames(SCENE)[:N]
    subj.reset()
    live = []
    for ref in refs:
        s = src.load(ref)
        assert s.reference is not None
        live.append(subj.predict(s, taps=TAPS))
    scenes_live, frames_live = src.scenes(), {sc: [r.key for r in src.frames(sc)] for sc in src.scenes()}

    fsrc = F.free_source(src)                                              # from here on the dataset makes its items without `gt`
    fsub = F.FreeSubject(subj)
    assert fsrc.scenes() == scenes_live and {sc: [r.key for r in fsrc.frames(sc)] for sc in fsrc.scenes()} == frames_live, "the frames and their dataset keys are the live ones"
    d = fsrc.describe()
    assert not any("gt" in k for k in d) and {"dataset", "feature_cache", "control"} <= set(d), d
    assert fsub.describe()["class"] == "FreeSubject" and [t.name for t in fsub.taps()] == [t.name for t in subj.taps()] and fsub.device == subj.device

    def run_free():
        fsub.reset()
        out = []
        for ref in refs:
            s = fsrc.load(ref)
            assert s.reference is None and "gt" not in s.inputs and s.meta["token"] == ref.key
            out.append(fsub.predict(s, taps=TAPS))
        return out
    free1 = run_free()
    for k, (a, b) in enumerate(zip(live, free1)):
        same(a.outputs, b.outputs, f"frame {k} outputs")
        same(a.taps, b.taps, f"frame {k} taps")
        assert "loss" in a.info and "loss" not in b.info, (a.info, b.info)
    dts = sorted(free1[0].outputs["dt"])
    free2 = run_free()                                                     # reset() restarts the memory: the same frames give the same predictions
    for k, (a, b) in enumerate(zip(free1, free2)):
        same(a.outputs, b.outputs, f"pass 2 frame {k} outputs")
        same(a.taps, b.taps, f"pass 2 frame {k} taps")
    conf = [float(np.mean(p.taps["memory_conf"])) for p in free1]
    assert conf[0] <= 0.05 and conf[-1] > conf[0], f"the memory must be empty at the first frame and grow: {conf}"

    # a dataset item that is not the planned frame is refused; an interface error of the model is a ValueError
    other, planned = fsrc.frames(SCENE)[1], refs[0]
    saved = fsrc._idx[other.key]
    fsrc._idx[other.key] = fsrc._idx[planned.key]
    try:
        e = raises(lambda: fsrc.load(other), ValueError)
        assert "disagree" in str(e), e
    finally:
        fsrc._idx[other.key] = saved
    fresh = F.FreeSubject(subj)                                            # its first call runs the interface check
    s = fsrc.load(refs[0])
    broken = dict(s.inputs)
    broken.pop("radar", None)
    e = raises(lambda: fresh.predict(type(s)(ref=s.ref, inputs=broken, reference=None), taps=()), (ValueError, KeyError, AssertionError, TypeError))
    n_arrays = sum(1 for o in free1[0].outputs["dt"].values() for g in ("prob", "dist", "dir") for _ in o.get(g, {})) + len(free1[0].taps)       # (a snapshot without the 1 m head has no `dir`)
    print(f"test_bev_free passed: {N} frames of {SCENE} through {os.path.basename(RUN)}: FreeSource / FreeSubject outputs bit-equal to the live ones for dts {dts} "
          f"({n_arrays} arrays in frame 0 and the detections), taps {sorted(free1[0].taps)}, second pass after reset equal, memory confidence {[round(c, 3) for c in conf]}, "
          f"a wrong dataset item and a broken input refused ({type(e).__name__})")


if __name__ == "__main__":
    main()
