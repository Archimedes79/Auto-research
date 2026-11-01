"""Command line of the autoGT suite (from the repository root):

  python -m autogt selftest
  python -m autogt generate DATAROOT [--out DIR] [--version V] [--scenes scene-0061 ...] [--stride 1] [--res 0.5] [--workers 1]
                          [--no-flow] [--no-radar] [--boxes-at camera|annotation]
      THE entry point: reads nuScenes from DATAROOT (e.g. ~/data/nuscenes_full) and writes one compressed file per keyframe to
      DIR/<signature>/<sample token>.npz (float16 layers + meta) and an index.json; DIR default data/autogt (where
      autogt.load reads); memory-light (groups of 60 sequences), resumable (existing files are skipped); the signature changes
      with every edit of the generator, so GT of an older generator is never mixed in
  python -m autogt build  --version v1.0-mini --out DIR [--dataroot ~/data/nuscenes] ...   the low-level writer behind it (full devkit)
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
import types

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


def _init(dataroot, version, res, ss, options, scenes=None):
    """a worker's generator; scenes = names: the memory-light devkit of just these sequences (`generate`), else the full devkit"""
    from .suite import AutoGT
    _STATE["gt"] = AutoGT(dataroot, version, GridSpec(res=res), ss, light=scenes is not None, scenes=scenes, **options)


def write_index(sig_dir, gen, version, names, stride, n_keyframes, options, complete=True):
    """index.json of a cache folder (marks it as an autogt cache for `clean`, records what it holds). `build` writes it with
    complete=False before the first keyframe (so an interrupted build still says which resolution it holds) and again at the end."""
    names = list(names)
    try:
        with open(os.path.join(sig_dir, "index.json")) as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = {}
    if (old.get("signature") == gen.signature() and abs(old.get("res", gen.grid.res) - gen.grid.res) < 1e-9 and old.get("ss", gen.ss) == gen.ss
            and not set(old.get("scenes", [])) <= set(names)):
        # mini is a subset of trainval and both share the folder of their signature: a smaller run (the mini report after the trainval
        # one) must not shrink the record of what the folder holds
        if set(old["scenes"]) >= set(names):
            version = old["version"]
        names = sorted(set(old["scenes"]) | set(names))
        n_keyframes = len([f for f in os.listdir(sig_dir) if f.endswith(".npz") and ".tmp" not in f])
    from .versions import version_of
    gt_version = version_of(gen.signature())                       # None: a signature without a row in versions.csv (unreleased): no key
    idx = {"signature": gen.signature(), **({"gt_version": gt_version} if gt_version else {}), "version": version, "res": gen.grid.res,
           "ss": gen.ss, "options": {k: repr(v) for k, v in options.items()}, "scenes": names, "stride": stride, "keyframes": n_keyframes,
           "complete": complete}
    tmp = os.path.join(sig_dir, "index.json.tmp")
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=1)
    os.replace(tmp, os.path.join(sig_dir, "index.json"))


def check_resume(sig_dir, grid, ss):
    """The folder is named by the generator signature, which contains neither the resolution nor the supersampling: continuing a
    build of another one would reuse its keyframes silently. The index says what the folder holds. A folder of an interrupted build
    from before the provisional index has none: one stored keyframe is compared with the grid instead."""
    idx_path = os.path.join(sig_dir, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            held = json.load(f)
        if abs(held.get("res", grid.res) - grid.res) > 1e-9 or held.get("ss", ss) != ss:
            raise SystemExit(f"{sig_dir} holds res {held.get('res')} / ss {held.get('ss')} (asked: res {grid.res} / ss {ss}): "
                             "build into another output folder")
        return
    stored = [f for f in os.listdir(sig_dir) if f.endswith(".npz") and ".tmp" not in f] if os.path.isdir(sig_dir) else []
    if stored:
        with np.load(os.path.join(sig_dir, stored[0]), allow_pickle=True) as z:
            shape = tuple(z["drivable_area"].shape) if "drivable_area" in z.files else None
        if shape is not None and shape != (grid.nx, grid.ny):
            raise SystemExit(f"{sig_dir} has no index.json (an interrupted build) and its keyframes have the cell grid {shape}, not "
                             f"{(grid.nx, grid.ny)} (res {grid.res}): build into another output folder")


def _write_tokens(gt, tokens, sig_dir, workers, init_args, verbose):
    """write the keyframes `tokens` of the generator `gt` into sig_dir (existing files are skipped); -> the number written.
    workers > 1: a pool of spawned processes, each builds its own generator from init_args (see _init)."""
    t0 = time.time()
    done = 0
    if workers <= 1:
        _STATE["gt"] = gt
        for i, t in enumerate(tokens):
            done += _work((t, sig_dir))
            if verbose and i % 50 == 0:
                print(f"{i}/{len(tokens)} keyframes, {time.time() - t0:.0f} s", flush=True)
    else:
        with mp.get_context("spawn").Pool(workers, _init, init_args) as pool:
            for i, r in enumerate(pool.imap_unordered(_work, [(t, sig_dir) for t in tokens], chunksize=4)):
                done += r
                if verbose and i % 200 == 0:
                    print(f"{i}/{len(tokens)} keyframes, {time.time() - t0:.0f} s", flush=True)
    return done


def build(dataroot, version, out, scenes=None, stride=1, workers=1, res=0.5, ss=4, verbose=True, **options):
    """the low-level writer (one full devkit, `out` required; `generate` is the entry point): returns the number of keyframes written
    (existing ones are skipped)."""
    from .suite import AutoGT
    gt = AutoGT(dataroot, version, GridSpec(res=res), ss, **options)
    sig_dir = os.path.join(out, gt.signature())
    check_resume(sig_dir, gt.grid, ss)
    os.makedirs(sig_dir, exist_ok=True)
    names = scenes or gt.scenes()
    tokens = [t for s in names for t in gt.keyframes(s)[::stride]]
    write_index(sig_dir, gt, version, names, stride, len(tokens), options, complete=False)       # before the first keyframe, also on a resume
    done = _write_tokens(gt, tokens, sig_dir, workers, (dataroot, version, res, ss, options), verbose)
    write_index(sig_dir, gt, version, names, stride, len(tokens), options)
    if verbose:
        print(f"{done} written, {len(tokens) - done} already there: {sig_dir}")
    return done


GROUP_SCENES = 60            # sequences per devkit load: the memory-light devkit of 60 sequences needs 0.4 GB (the full trainval one 6.4 GB)


def detect_version(dataroot):
    """the nuScenes version inside `dataroot`: the one `v1.0-*` folder that holds a scene.json"""
    root = os.path.expanduser(dataroot)
    found = sorted(d for d in os.listdir(root) if d.startswith("v1.0-") and os.path.exists(os.path.join(root, d, "scene.json"))) \
        if os.path.isdir(root) else []
    if len(found) != 1:
        raise ValueError(f"{root}: " + (f"several nuScenes versions {found}, pass version=" if found else "no v1.0-*/scene.json: not a nuScenes folder"))
    return found[0]


def generate(dataroot, out=None, version=None, scenes=None, stride=1, res=0.5, ss=4, workers=1, verbose=True, **options):
    """THE entry point of the suite: read nuScenes from `dataroot` and write the GT of every keyframe.

    dataroot   a nuScenes folder (it holds v1.0-mini/ or v1.0-trainval/, maps/, samples/, sweeps/); ~ is expanded
    out        where the GT goes: OUT/<generator signature>/<sample token>.npz + index.json; default data/autogt, the
               place `autogt.load(token)` reads from (not in git)
    version    v1.0-mini / v1.0-trainval; default: the one found in dataroot (an error if there are several)
    scenes     names such as ["scene-0061"]; default all sequences of the version
    stride     every n-th keyframe (1 = all); res: cell size in m (0.5), ss: supersampling (4)
    workers    processes (1 = this process); options: stationary_fix, use_flow, use_radar, boxes_at, ... (see AutoGT)
    -> {"out": the folder written, "signature", "gt_version" (the key of versions.csv, D29), "version", "scenes", "keyframes": wanted,
        "written": new files}

    Memory-light and resumable: the sequences are read in groups of GROUP_SCENES with the light devkit (the GT is identical to the
    full devkit's), files that exist are skipped, so an interrupted run continues where it stopped. A folder of another resolution
    or supersampling is never continued. The signature changes with every edit of the generator, so GT of an older generator is
    never mixed in. Needs no GPU; trainval takes 5 - 8 hours in one process."""
    import gc
    from .suite import AutoGT, DATA, json_tables, scene_tokens
    dataroot = os.path.expanduser(dataroot)
    version = version or detect_version(dataroot)
    out = os.path.expanduser(out) if out else DATA
    known = [s["name"] for s in json_tables(dataroot, version)[0]]
    names = list(scenes) if scenes else known
    if not names or [n for n in names if n not in known]:
        raise ValueError(f"unknown scene(s) {[n for n in names if n not in known] or 'none given'} in {version} of {dataroot}")
    tokens = {n: scene_tokens(n, version, dataroot)[::stride] for n in names}                       # json tables only, no devkit
    n_total = sum(len(t) for t in tokens.values())
    sig_dir, ident, done = None, None, 0
    for g0 in range(0, len(names), GROUP_SCENES):
        grp = names[g0:g0 + GROUP_SCENES]
        gen = AutoGT(dataroot, version, GridSpec(res=res), ss, light=True, scenes=grp, **options)
        if sig_dir is None:
            sig = gen.signature()
            sig_dir = os.path.join(out, sig)
            check_resume(sig_dir, gen.grid, ss)
            os.makedirs(sig_dir, exist_ok=True)
            write_index(sig_dir, gen, version, names, stride, n_total, options, complete=False)    # before the first keyframe, also on a resume
            ident = types.SimpleNamespace(signature=lambda: sig, grid=gen.grid, ss=ss)               # what write_index needs, without the devkit
        done += _write_tokens(gen, [t for n in grp for t in tokens[n]], sig_dir, workers, (dataroot, version, res, ss, options, grp), verbose)
        del gen
        gc.collect()
    write_index(sig_dir, ident, version, names, stride, n_total, options)
    from .versions import version_of
    gt_version = version_of(ident.signature()) or "unreleased"
    if verbose:
        print(f"{done} keyframes written, {n_total - done} already there: {sig_dir}  (GT version {gt_version})\n"
              f"read one with autogt.load(<sample token>{'' if out == DATA else ', data=' + repr(out)})")
    return {"out": sig_dir, "signature": ident.signature(), "gt_version": gt_version, "version": version, "scenes": len(names),
            "keyframes": n_total, "written": done}


def _print_versions():
    """the table of versions.csv: key, signature, date, CURRENT for this source, stored when its folder exists in data/autogt"""
    from . import versions
    from .suite import DATA, current_signature
    cur, rows = current_signature(), versions.table()
    for r in rows:
        print(f"{r['version']:3s} {r['signature']}  {r['date']}  {'CURRENT' if r['signature'] == cur else '       '}  "
              f"{'stored    ' if os.path.isdir(os.path.join(DATA, r['signature'])) else 'not stored'}  {r['tag']:11s} {r['what_changed'][:80]}")
    if cur not in {r["signature"] for r in rows}:
        print(f"this source has the signature {cur}: no row in autogt/versions.csv (unreleased); add a row, commit, tag autogt/v<N>")


def _boxes_option(a):
    """the generator option of --boxes-at (nothing for the default, so the default options stay empty in the records)."""
    return {} if a.boxes_at == "camera" else {"boxes_at": a.boxes_at}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m autogt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("versions", help="the GT versions (autogt/versions.csv, D29): the current one marked, which GT is stored")
    ge = sub.add_parser("generate", help="read nuScenes from DATAROOT and write the GT of every keyframe (the entry point)")
    ge.add_argument("dataroot", help="a nuScenes folder, e.g. ~/data/nuscenes_full (it holds v1.0-trainval/, maps/, samples/, sweeps/)")
    ge.add_argument("--out", default=None, help="folder of the stored GT (default data/autogt); the files go to OUT/<signature>/")
    ge.add_argument("--version", default=None, help="v1.0-mini | v1.0-trainval (default: the one found in DATAROOT)")
    ge.add_argument("--scenes", nargs="*", help="names such as scene-0061 (default: all)")
    ge.add_argument("--res", type=float, default=0.5)
    ge.add_argument("--stride", type=int, default=1, help="every n-th keyframe")
    ge.add_argument("--workers", type=int, default=1)
    ge.add_argument("--no-flow", action="store_true")
    ge.add_argument("--no-radar", action="store_true")
    ge.add_argument("--boxes-at", choices=("camera", "annotation"), default="camera",
                    help="time of the moving boxes: the camera time (default) or the annotation time")
    b = sub.add_parser("build", help="the low-level writer of generate (one full devkit, --out required)")
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
    r.add_argument("--out", default=None, help="report PDF (default data/autogt/reports/autogt_report_<version>_<date>.pdf)")
    r.add_argument("--data", default=None, help="cache directory (default data/autogt); --no-data: do not store")
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
    elif a.cmd == "versions":
        _print_versions()
    elif a.cmd == "generate":
        generate(a.dataroot, a.out, a.version, a.scenes, a.stride, a.res, workers=a.workers,
                 use_flow=not a.no_flow, use_radar=not a.no_radar, **_boxes_option(a))
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
        out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "autogt", "reports",
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
