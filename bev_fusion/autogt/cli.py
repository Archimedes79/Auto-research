"""Command line of the autoGT suite (from the bev_fusion folder):

  python -m autogt selftest
  python -m autogt build  --version v1.0-mini --out DIR [--dataroot ~/data/nuscenes] [--scenes scene-0061 ...] [--res 0.5]
                          [--stride 1] [--workers 1] [--no-flow] [--no-radar]
      writes one compressed file per keyframe to DIR/<signature>/<sample token>.npz (float16 layers + meta) and an index.json;
      resumable (existing files are skipped); the signature changes with every edit of the generator, so GT of an older
      generator is never mixed in
  python -m autogt check  [--version v1.0-mini] [--scenes ...] [--stride 1] [--holdout]
      automatic consistency checks of the GT over keyframes (see autogt/checks.py); --holdout adds the leave-one-radar-out test
  python -m autogt view   OUT_DIR [scene:keyframe | TOKEN ...] [--total]   sample pictures (camera, semantic BEV, objects + velocity,
      motion relative to the static world; --total shows the total apparent flow instead); scene:keyframe = nuScenes mini,
      generated now; TOKEN = a stored keyframe of mini or trainval (needs the stored GT of the current generator)
  python -m autogt sheets OUT_DIR                            overview sheets of the sample pictures
  python -m autogt video  SCENE OUT.mp4 [seconds]            driving through the GT at 12 Hz
  python -m autogt report [--version v1.0-mini|v1.0-trainval] [--out REPORT.pdf] [--data DIR | --no-data] [--scenes ..] [--stride N]
                          [--samples N] [--holdout] [--workers N] [--light]
      GT evaluation of all sequences in one pass: store the GT (as `build`), run every check, write ONE PDF report (summary, table
      per sequence, legend, sample pictures drawn straight into the PDF - no picture files exist, nothing to clean up but the report)
  python -m autogt clean  [--videos] [--pictures] [--reports] [--old-outputs] [--stale-caches DIR | --all-caches DIR] [--older-than DAYS] [--move | --delete]
      dry run by default; --move puts the items into _to_delete/, --delete removes them (see autogt/clean.py)
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


def write_index(sig_dir, gen, version, names, stride, n_keyframes, options):
    """index.json of a cache folder (marks it as an autogt cache for `clean`, records what it holds)."""
    idx = {"signature": gen.signature(), "version": version, "res": gen.grid.res, "ss": gen.ss,
           "options": {k: repr(v) for k, v in options.items()}, "scenes": list(names), "stride": stride, "keyframes": n_keyframes}
    json.dump(idx, open(os.path.join(sig_dir, "index.json"), "w"), indent=1)


def build(dataroot, version, out, scenes=None, stride=1, workers=1, res=0.5, ss=4, verbose=True, **options):
    """see the module docstring; returns the number of keyframes written (existing ones are skipped)."""
    from .suite import AutoGT
    gt = AutoGT(dataroot, version, GridSpec(res=res), ss, **options)
    sig_dir = os.path.join(out, gt.signature())
    idx_path = os.path.join(sig_dir, "index.json")
    if os.path.exists(idx_path):               # the signature does not contain res / ss: existing files would be reused silently
        held = json.load(open(idx_path))
        if abs(held.get("res", res) - res) > 1e-9 or held.get("ss", ss) != ss:
            raise SystemExit(f"{sig_dir} holds res {held.get('res')} / ss {held.get('ss')} (asked: res {res} / ss {ss}): "
                             "build into another output folder")
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
    write_index(sig_dir, gt, version, names, stride, len(tokens), options)
    if verbose:
        print(f"{done} written, {len(tokens) - done} already there: {sig_dir}")
    return done


def _boxes_option(a):
    """the generator option of --boxes-at (nothing for the default, so the default options stay empty in the records)."""
    return {} if a.boxes_at == "camera" else {"boxes_at": a.boxes_at}


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
    b.add_argument("--boxes-at", choices=("camera", "annotation"), default="camera",
                   help="time of the moving boxes: the camera time (default) or the annotation time")
    c = sub.add_parser("check")
    c.add_argument("--dataroot", default="~/data/nuscenes")
    c.add_argument("--version", default="v1.0-mini")
    c.add_argument("--scenes", nargs="*")
    c.add_argument("--stride", type=int, default=1)
    c.add_argument("--holdout", action="store_true")
    v = sub.add_parser("view")
    v.add_argument("out")
    v.add_argument("picks", nargs="*", help="scene-0061:8 (mini, generated now) or a sample token (stored GT, mini or trainval)")
    v.add_argument("--total", action="store_true", help="4th panel: total apparent flow (ego motion included) instead of motion relative to the static world")
    s = sub.add_parser("sheets")
    s.add_argument("out")
    r = sub.add_parser("report", help="evaluate the GT of all sequences, store it, write one PDF report with samples")
    r.add_argument("--dataroot", default=None, help="default ~/data/nuscenes (mini) or ~/data/nuscenes_full (trainval)")
    r.add_argument("--version", default="v1.0-mini")
    r.add_argument("--workers", type=int, default=1, help="processes (fork, one sequence per task); a finished sequence is saved, reruns resume")
    r.add_argument("--light", action="store_true", help="memory-light devkit (automatic for trainval)")
    r.add_argument("--out", default=None, help="report PDF (default autogt/reports/autogt_report_<version>_<date>.pdf)")
    r.add_argument("--data", default=None, help="cache directory (default bev_fusion/data/autogt); --no-data: do not store")
    r.add_argument("--no-data", action="store_true")
    r.add_argument("--scenes", nargs="*")
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--samples", type=int, default=10, help="sample pictures in the report (evenly spread over the sequences)")
    r.add_argument("--holdout", action="store_true", help="add the leave-one-radar-out test of the radar correction")
    r.add_argument("--boxes-at", choices=("camera", "annotation"), default="camera",
                   help="time of the moving boxes: the camera time (default) or the annotation time")
    cl = sub.add_parser("clean", help="list / move / delete generated videos, pictures, old outputs, stale caches")
    cl.add_argument("--videos", action="store_true")
    cl.add_argument("--pictures", action="store_true")
    cl.add_argument("--reports", action="store_true")
    cl.add_argument("--old-outputs", action="store_true")
    cl.add_argument("--all-caches", metavar="DIR")
    cl.add_argument("--stale-caches", metavar="DIR")
    cl.add_argument("--older-than", type=float, metavar="DAYS")
    cl.add_argument("--move", action="store_true")
    cl.add_argument("--delete", action="store_true")
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
              use_flow=not a.no_flow, use_radar=not a.no_radar, **_boxes_option(a))
    elif a.cmd == "check":
        from .checks import run
        run(os.path.expanduser(a.dataroot), a.version, a.scenes, a.stride, a.holdout)
    elif a.cmd == "view":
        from .view import run
        run(a.out, [x.split(":") for x in a.picks] or None, total=a.total)
    elif a.cmd == "sheets":
        from .view import sheets
        sheets(a.out)
    elif a.cmd == "report":
        import datetime
        from .report import report
        out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autogt", "reports",
                                    f"autogt_report_{a.version}_{datetime.date.today().isoformat()}.pdf")
        data = None if a.no_data else (a.data or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "autogt"))
        root = a.dataroot or ("~/data/nuscenes_full" if "trainval" in a.version else "~/data/nuscenes")
        report(os.path.expanduser(root), a.version, out, data, a.scenes, a.stride, a.holdout, a.samples, a.workers,
               a.light or "trainval" in a.version, **_boxes_option(a))
    elif a.cmd == "clean":
        from .clean import clean
        clean(delete=a.delete, move=a.move, videos=a.videos, pictures=a.pictures, reports=a.reports, old_outputs=a.old_outputs,
              stale_caches=a.stale_caches, all_caches=a.all_caches, older_than=a.older_than)
    elif a.cmd == "video":
        from .video import run
        run(a.scene, a.mp4, a.seconds)


if __name__ == "__main__":
    main()
