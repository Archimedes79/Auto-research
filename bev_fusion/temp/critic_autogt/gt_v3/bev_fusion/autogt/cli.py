"""Command line of the autoGT suite (from the bev_fusion folder):

  python -m autogt selftest
  python -m autogt build  --version v1.0-mini --out DIR [--dataroot ~/data/nuscenes] [--scenes scene-0061 ...] [--res 0.5]
                          [--stride 1] [--workers 1] [--no-flow] [--no-radar]
      writes one compressed file per keyframe to DIR/<signature>/<sample token>.npz (float16 layers + meta) and an index.json;
      resumable (existing files are skipped); the signature changes with every edit of the generator, so GT of an older
      generator is never mixed in
  python -m autogt check  [--version v1.0-mini] [--scenes ...] [--stride 1] [--holdout]
      automatic consistency checks of the GT over keyframes (see autogt/checks.py); --holdout adds the leave-one-radar-out test
  python -m autogt view   OUT_DIR [scene:keyframe ...]      sample pictures (camera, semantic BEV, objects + velocity, flow)
  python -m autogt sheets OUT_DIR                            overview sheets of the sample pictures
  python -m autogt video  SCENE OUT.mp4 [seconds]            driving through the GT at 12 Hz
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from .grid import GridSpec

_STATE = {}


def _flat(g):
    """GT dict -> npz arrays (float16 for float32 layers) + meta as an object array."""
    arrs = {k: (v.astype(np.float16) if v.dtype == np.float32 else v) for k, v in g.items() if isinstance(v, np.ndarray)}
    arrs["meta"] = np.array(g["meta"], dtype=object)
    return arrs


def load_keyframe(sig_dir, token):
    """a keyframe written by `build` -> the GT dict (float32 layers + meta)."""
    z = np.load(os.path.join(sig_dir, token + ".npz"), allow_pickle=True)
    out = {k: (z[k].astype(np.float32) if z[k].dtype == np.float16 else z[k]) for k in z.files if k != "meta"}
    out["meta"] = z["meta"].item()
    return out


def _work(args):
    token, sig_dir = args
    gt = _STATE["gt"]
    path = os.path.join(sig_dir, token + ".npz")
    if os.path.exists(path):
        return 0
    np.savez_compressed(path + ".tmp.npz", **_flat(gt.keyframe(token)))
    os.replace(path + ".tmp.npz", path)
    return 1


def _init(dataroot, version, res, ss, options):
    from .suite import AutoGT
    _STATE["gt"] = AutoGT(dataroot, version, GridSpec(res=res), ss, **options)


def build(dataroot, version, out, scenes=None, stride=1, workers=1, res=0.5, ss=4, verbose=True, **options):
    """see the module docstring; returns the number of keyframes written (existing ones are skipped)."""
    from .suite import AutoGT
    gt = AutoGT(dataroot, version, GridSpec(res=res), ss, **options)
    sig_dir = os.path.join(out, gt.signature())
    os.makedirs(sig_dir, exist_ok=True)
    names = scenes or gt.scenes()
    tokens = [t for s in names for t in gt.keyframes(s)[::stride]]
    t0 = time.time()
    if workers <= 1:
        _STATE["gt"] = gt
        done = 0
        for i, t in enumerate(tokens):
            done += _work((t, sig_dir))
            if verbose and i % 50 == 0:
                print(f"{i}/{len(tokens)} keyframes, {time.time() - t0:.0f} s", flush=True)
    else:
        with mp.get_context("spawn").Pool(workers, _init, (dataroot, version, res, ss, options)) as pool:
            done = 0
            for i, r in enumerate(pool.imap_unordered(_work, [(t, sig_dir) for t in tokens], chunksize=4)):
                done += r
                if verbose and i % 200 == 0:
                    print(f"{i}/{len(tokens)} keyframes, {time.time() - t0:.0f} s", flush=True)
    idx = {"signature": gt.signature(), "version": version, "res": res, "ss": ss, "options": {k: repr(v) for k, v in options.items()},
           "scenes": names, "stride": stride, "keyframes": len(tokens)}
    json.dump(idx, open(os.path.join(sig_dir, "index.json"), "w"), indent=1)
    if verbose:
        print(f"{done} written, {len(tokens) - done} already there: {sig_dir}")
    return done


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m autogt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    b = sub.add_parser("build")
    b.add_argument("--dataroot", default="~/data/nuscenes")
    b.add_argument("--version", default="v1.0-mini")
    b.add_argument("--out", required=True)
    b.add_argument("--scenes", nargs="*")
    b.add_argument("--res", type=float, default=0.5)
    b.add_argument("--stride", type=int, default=1)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--no-flow", action="store_true")
    b.add_argument("--no-radar", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--dataroot", default="~/data/nuscenes")
    c.add_argument("--version", default="v1.0-mini")
    c.add_argument("--scenes", nargs="*")
    c.add_argument("--stride", type=int, default=1)
    c.add_argument("--holdout", action="store_true")
    v = sub.add_parser("view")
    v.add_argument("out")
    v.add_argument("picks", nargs="*")
    s = sub.add_parser("sheets")
    s.add_argument("out")
    d = sub.add_parser("video")
    d.add_argument("scene")
    d.add_argument("mp4")
    d.add_argument("seconds", nargs="?", type=float, default=0.0)
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        from .selftest import run
        run()
    elif a.cmd == "build":
        build(os.path.expanduser(a.dataroot), a.version, a.out, a.scenes, a.stride, a.workers, a.res,
              use_flow=not a.no_flow, use_radar=not a.no_radar)
    elif a.cmd == "check":
        from .checks import run
        run(os.path.expanduser(a.dataroot), a.version, a.scenes, a.stride, a.holdout)
    elif a.cmd == "view":
        from .view import run
        run(a.out, [x.split(":") for x in a.picks] or None)
    elif a.cmd == "sheets":
        from .view import sheets
        sheets(a.out)
    elif a.cmd == "video":
        from .video import run
        run(a.scene, a.mp4, a.seconds)


if __name__ == "__main__":
    main()
