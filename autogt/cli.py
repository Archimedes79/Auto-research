"""Command line of the autoGT suite (from the repository root: `python -m autogt COMMAND ...`; in WSL: `wsl bash autogt/run_nice.sh -m autogt COMMAND ...`).

Every command is ONE call of a function of the suite (the map below; `python -m autogt COMMAND --help` has the options), so what the command line does is what Python does:

  make GT (all resumable; --out is always the ROOT that holds the stores, default data/autogt; the lane stores are below ROOT/lane; they print the GT key, see autogt/gt_api.py)
    generate DATAROOT      pipeline.generate      the front GT + both lane GTs + the area masks of the BEV Fusion store (everything)
    generate-bev DATAROOT  gt_api.generate_bev_gt   BEV Fusion: the front GT + the lane GT of its rig + the packed intersection masks
    generate-lane DATAROOT gt_api.generate_lane_gt  the lane GT of one system (--rig lane_center | bev_fusion | all)
    area-masks             lane_raster.build_mask_cache  only the packed masks of an existing BEV Fusion lane store
    audit-lane SYSTEM      store_audit.audit_lane   the audit of a stored lane GT before its release (exit 1 on a problem)
    build                  pipeline.build           the low-level writer behind `report` (one full devkit, --out required)
  look at GT
    key SYSTEM             gt_api.gt_key            the key of what is stored (exit 1 when incomplete), --json for the dict
    have                   sets.stored_for          is the front GT of this generator there for these sequences? (exit 1 when not: the check before a training)
    data DATAROOT          sets.check_raw           are the raw nuScenes files on disk? (exit 1 when not)
    sets / versions        sets.status / versions   the scene sets and how much is stored / the GT versions, lane versions and releases
    area-mask TOKEN        lane_raster.load_mask    one area mask: the number of cells, --png to draw it
    view / sheets / video / lanegraph / masks     pictures of keyframes, of the lane graph, of the coverage masks; videos
  check and clean
    check                  checks.run               consistency checks of the GT of mini        report   report.report   the checks over all sequences, one PDF
    digest                 checks.digests           a digest per frame of mini (two generators that make the same GT give the same file)
    clean                  clean.clean              list / move generated pictures, videos, reports and old caches (--delete only for what one command rebuilds)
    selftest               selftest.run             all tests of the suite (5 - 10 min)

Exit codes: 0 done, 1 the thing asked about is not there / not complete (have, data, key, digest --compare), 2 wrong arguments (a message, no traceback; AUTOGT_TRACEBACK=1 shows it).
"""
import argparse
import json
import os
import sys

# the storage and the writer moved to store.py / pipeline.py; the old names stay importable from here (bev_fusion tests and other projects use them)
from .pipeline import GROUP_SCENES, _group_task, _init, _same_generator, _work, _write_tokens, build, generate  # noqa: F401
from .store import _environment, _flat, canonical_of, check_resume, detect_version, load_keyframe, run_manifest, save_keyframe, write_index  # noqa: F401


# ---------------------------------------------------------------------------------------------------------------- the tables the commands print
def _print_versions():
    """the table of versions.csv: key, signature, date, CURRENT for this source, stored when its folder exists in data/autogt"""
    from . import lane_gt, versions
    from .suite import DATA, code_signature
    cur, rows = code_signature(), versions.table()
    for r in rows:
        same = f" (the GT of {r['same_gt_as']})" if r.get("same_gt_as") else ""
        print(f"{r['version']:3s} {r['signature']}  {r['date']}  {'CURRENT' if r['signature'] == cur else '       '}  "
              f"{'stored    ' if os.path.isdir(os.path.join(DATA, versions.canonical(r['signature']))) else 'not stored'}  {r['tag']:11s} "
              f"{r['what_changed'][:70]}{same}")
    if cur not in {r["signature"] for r in rows}:
        print(f"this source has the signature {cur}: no row in autogt/versions.csv (unreleased); add a row, commit, tag autogt/v<N>")
    here = versions.current_release()
    print("lane GT (own versions, autogt/lane_versions.csv):")
    for r in lane_gt.table():
        same = f" (the GT of {r['same_gt_as']})" if r["same_gt_as"] else ""
        print(f"  {r['version']:3s} {r['rig']:11s} {r['signature']}  {r['date']}  {'CURRENT' if r['signature'] == lane_gt.code_signature(rig=r['rig']) else '       '}  "
              f"{'stored    ' if os.path.isdir(os.path.join(lane_gt.DATA, lane_gt.store_of(r['signature']))) else 'not stored'}  {r['what_changed'][:60]}{same}")
    print("releases (the suite as one number, autogt/releases.csv):")
    for r in versions.releases():
        print(f"  {r['release']:5s} front {r['front']:4s} + lane {r['lane']:3s}  {r['date']}  {'CURRENT' if r['release'] == here else '       '}  {r['what_changed'][:80]}")
    if here is None:
        print(f"this source (front {versions.version_of(cur)}, lane {lane_gt.version_of(lane_gt.code_signature())}) has no row in autogt/releases.csv: add one")


def _print_sets():
    """each scene set: its sequences, keyframes, and how much of it the store of the current generator holds"""
    from . import sets
    for name in sets.names():
        try:
            s = sets.status(name)
        except ValueError as e:
            print(f"{name:6s} {len(sets.scenes(name)):4d} scenes  not readable here: {e}")
            continue
        print(f"{name:6s} {len(sets.scenes(name)):4d} scenes  {s['total']:6d} keyframes ({s['version']})  stored for this generator: "
              f"{s['have']}/{s['total']}{'  COMPLETE' if s['complete'] else ''}")


# ---------------------------------------------------------------------------------------------------------------- argument helpers
def _scene_names(a):
    """the sequence names asked for on the command line: --scenes, --scenes-file (one name per line, # comments) or None (--set / all)"""
    if getattr(a, "scenes_file", None):
        with open(a.scenes_file, encoding="utf-8") as f:
            return [x for x in (ln.split("#")[0].strip() for ln in f) if x]
    return getattr(a, "scenes", None)


def _options(a):
    """the generator options of the command line (flow and radar on unless switched off; boxes_at only when it is not the default)"""
    return {"use_flow": not a.no_flow, "use_radar": not a.no_radar, **({} if a.boxes_at == "camera" else {"boxes_at": a.boxes_at})}


SELFTESTS = {"cli": "cli_selftest", "gt_api": "gt_api_selftest", "generator": "generator_selftest", "lane_graph": "lane_graph_selftest", "lane_tiers": "lane_tiers_selftest",
             "clean": "clean", "rig_masks": "rig_masks"}           # the parts that can run alone: module names


def _sets():
    from . import sets
    return sets.names()


def _which_scenes(p, set_flag=True):
    """--set | --scenes | --scenes-file: one of them (argparse refuses two)"""
    g = p.add_mutually_exclusive_group()
    if set_flag:
        g.add_argument("--set", dest="set_name", default=None, choices=_sets(), help="a scene set: small (the 10 sequences of mini) or full (all 850 of trainval)")
    g.add_argument("--scenes", nargs="+", help="names such as scene-0061 (default: all)")
    g.add_argument("--scenes-file", help="a file of sequence names, one per line")


def _generator_options(p):
    """the options of the generator that `generate`, `build` and `report` share"""
    p.add_argument("--no-flow", action="store_true")
    p.add_argument("--no-radar", action="store_true")
    p.add_argument("--boxes-at", choices=("camera", "annotation"), default="camera", help="time of the moving boxes: the camera time (default) or the annotation time")


def _parser():
    p = argparse.ArgumentParser(prog="python -m autogt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    registry = {}

    def command(name, func, help, **kw):
        sp = sub.add_parser(name, help=help, description=help, **kw)
        sp.set_defaults(func=func)
        registry[name] = func
        return sp

    def dataroot(sp):
        sp.add_argument("dataroot", help="a nuScenes folder, e.g. ~/data/nuscenes_full (it holds v1.0-trainval/, maps/, samples/, sweeps/)")

    # ---- make GT
    g = command("generate", _cmd_generate, "everything: the front GT, the lane GT of both systems and the area masks (pipeline.generate)")
    dataroot(g)
    g.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt): the front GT goes to OUT/<signature>/, the lane GT to OUT/lane/")
    g.add_argument("--version", default=None, help="v1.0-mini | v1.0-trainval (default: the one found in DATAROOT)")
    _which_scenes(g)
    g.add_argument("--stride", type=int, default=1, help="every n-th keyframe (the lane GT is made for every keyframe: with a stride only the front GT)")
    g.add_argument("--res", type=float, default=0.5)
    g.add_argument("--workers", type=int, default=1, help="processes")
    g.add_argument("--no-lane", action="store_true", help="only the front GT")
    _generator_options(g)
    b = command("generate-bev", _cmd_generate_bev, "BEV Fusion: the front GT + the lane GT of the rig bev_fusion + the packed intersection masks; prints the GT key (gt_api.generate_bev_gt)")
    dataroot(b)
    b.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt)")
    b.add_argument("--version", default=None)
    _which_scenes(b)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--border-m", dest="borders", type=float, nargs="+", default=[0.0, 0.5], help="the borders in metres of the intersection masks (default 0 and 0.5)")
    ln = command("generate-lane", _cmd_generate_lane, "the lane GT of one system or both; prints the key(s) (gt_api.generate_lane_gt)")
    dataroot(ln)
    ln.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt): the lane GT goes to OUT/lane/")
    ln.add_argument("--version", default=None)
    _which_scenes(ln)
    ln.add_argument("--workers", type=int, default=1)
    ln.add_argument("--rig", default="all", choices=["all", "lane_center", "bev_fusion"], help="the system: Spurmitte (lane_center), BEV Fusion (bev_fusion), or both")
    au = command("audit-lane", _cmd_audit_lane, "audit a stored lane GT before it is released: every keyframe there, every file loads and holds what the version promises; exit 1 on a problem (store_audit.audit_lane)")
    au.add_argument("system", choices=["bev_fusion", "lane_center"])
    au.add_argument("--dataset", default="v1.0-trainval", help="v1.0-mini | v1.0-trainval")
    au.add_argument("--dataroot", default=None, help="a nuScenes folder (default ~/data/nuscenes_full or ~/data/nuscenes)")
    au.add_argument("--scenes", nargs="+", help="only the keyframes of these sequences")
    au.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt)")
    au.add_argument("--limit", type=int, default=None, help="check only the first N files")
    am = command("area-masks", _cmd_area_masks, "the packed area masks of an existing BEV Fusion lane store (lane_raster.build_mask_cache)")
    am.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt)")
    am.add_argument("--border-m", dest="borders", type=float, nargs="+", default=[0.0, 0.5])
    am.add_argument("--workers", type=int, default=1, help="processes (34k frames: about 11 min in one)")
    am.add_argument("--force", action="store_true", help="make them again although they are complete")
    bu = command("build", _cmd_build, "the low-level writer of generate (one full devkit, --out required; pipeline.build)")
    bu.add_argument("--dataroot", default="~/data/nuscenes")
    bu.add_argument("--version", default="v1.0-mini")
    bu.add_argument("--out", required=True)
    bu.add_argument("--scenes", nargs="+")
    bu.add_argument("--stride", type=int, default=1)
    bu.add_argument("--res", type=float, default=0.5)
    bu.add_argument("--workers", type=int, default=1)
    _generator_options(bu)
    # ---- look at GT
    k = command("key", _cmd_key, "the GT key of what is stored for a system: data + code signature of every part; exit 1 when incomplete (gt_api.gt_key)")
    k.add_argument("system", choices=["bev_fusion", "lane_center"])
    k.add_argument("--dataset", default="v1.0-trainval", help="v1.0-mini | v1.0-trainval")
    _which_scenes(k)
    k.add_argument("--dataroot", default=None, help="a nuScenes folder: complete is then judged by every keyframe of its json tables")
    k.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt)")
    k.add_argument("--json", action="store_true", help="print the key as json")
    h = command("have", _cmd_have, "is the stored front GT of the current generator there for these sequences? exit 1 when not (the check before a training; sets.stored_for)")
    _which_scenes(h)
    h.add_argument("--version", default=None, help="v1.0-mini | v1.0-trainval (default: the first that holds all the sequences)")
    h.add_argument("--dataroot", default=None)
    h.add_argument("--stride", type=int, default=1)
    h.add_argument("--data", default=None, help="the ROOT of the stores (default data/autogt)")
    d = command("data", _cmd_data, "are the raw nuScenes files the generator reads on disk (tables, maps, CAM_FRONT, radars)? exit 1 when not (sets.check_raw)")
    dataroot(d)
    d.add_argument("--version", default=None)
    _which_scenes(d)
    command("sets", lambda a: _print_sets(), "the scene sets small / full and how much of each is stored")
    command("versions", lambda a: _print_versions(), "the GT versions, the lane versions and the releases; the current one marked, which is stored")
    mk = command("area-mask", _cmd_area_mask, "one area mask of a keyframe of the BEV Fusion lane store: its size, --png to draw it (lane_raster.load_mask / area_mask)")
    mk.add_argument("token", help="the nuScenes sample token of a keyframe")
    mk.add_argument("--kind", default="intersection", choices=["intersection", "road"])
    mk.add_argument("--border-m", type=float, default=0.0)
    mk.add_argument("--png", default=None, help="write the mask as a picture (x forward = up, y left = left)")
    v = command("view", _cmd_view, "sample pictures of keyframes (camera, semantic BEV, objects + velocity, motion)")
    v.add_argument("out")
    v.add_argument("picks", nargs="*", help="scene-0061:8 (mini, generated now) or a sample token (stored GT, mini or trainval)")
    v.add_argument("--total", action="store_true", help="4th panel: total apparent flow (ego motion included) instead of motion relative to the static world")
    command("sheets", lambda a: _cmd_sheets(a), "overview sheets of the sample pictures").add_argument("out")
    command("masks", lambda a: _cmd_masks(a), "the two coverage masks (BEV fusion: front; Spurmitte: surround) as one picture").add_argument("out")
    lg = command("lanegraph", _cmd_lanegraph, "picture of the lane graph of a keyframe: the six cameras and the top view")
    lg.add_argument("out", help="the png to write")
    lg.add_argument("pick", help="scene-0061:8 (the scene's 9th keyframe)")
    lg.add_argument("--version", default="v1.0-mini")
    lg.add_argument("--dataroot", default=None)
    vd = command("video", _cmd_video, "drive through the GT of one sequence at 12 Hz (H.264); mini by default, any trainval sequence with --version")
    vd.add_argument("scene")
    vd.add_argument("mp4")
    vd.add_argument("seconds", nargs="?", type=float, default=0.0)
    vd.add_argument("--version", default="v1.0-mini", help="v1.0-mini | v1.0-trainval")
    vd.add_argument("--dataroot", default=None)
    # ---- check and clean
    c = command("check", _cmd_check, "automatic consistency checks of the GT over keyframes (checks.run)")
    c.add_argument("--dataroot", default="~/data/nuscenes")
    c.add_argument("--version", default="v1.0-mini")
    c.add_argument("--scenes", nargs="+")
    c.add_argument("--stride", type=int, default=1)
    c.add_argument("--holdout", action="store_true", help="add the leave-one-radar-out test of the radar correction")
    r = command("report", _cmd_report, "evaluate the GT of all sequences, store it, write ONE PDF report with samples (report.report; made at 0.5 m)")
    r.add_argument("--dataroot", default=None)
    r.add_argument("--version", default="v1.0-mini")
    r.add_argument("--light", action="store_true", help="memory-light devkit (automatic for trainval)")
    r.add_argument("--out", default=None, help="report PDF (default data/autogt/reports/autogt_report_<version>_<date>.pdf)")
    r.add_argument("--data", default=None, help="the ROOT of the stores (default data/autogt); --no-data: do not store")
    r.add_argument("--no-data", action="store_true")
    r.add_argument("--scenes", nargs="+")
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--samples", type=int, default=10, help="sample pictures in the report (evenly spread over the sequences)")
    r.add_argument("--holdout", action="store_true", help="add the leave-one-radar-out test (not with the light devkit of trainval)")
    _generator_options(r)
    dg = command("digest", _cmd_digest, "a digest per frame of nuScenes mini: two generators that make the same GT give the same file (checks.digests)")
    dg.add_argument("out", nargs="?", help="write the digests to this json file")
    dg.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two digest files: exit 1 when any frame differs")
    dg.add_argument("--stride", type=int, default=2)
    cl = command("clean", _cmd_clean, "list / move generated videos, pictures, reports, old outputs and caches; --delete only for what one command rebuilds (clean.clean)")
    for flag in ("--videos", "--pictures", "--reports", "--old-outputs"):
        cl.add_argument(flag, action="store_true")
    act = cl.add_mutually_exclusive_group()
    act.add_argument("--move", action="store_true")
    act.add_argument("--delete", action="store_true")
    cl.add_argument("--all-caches", metavar="DIR")
    cl.add_argument("--stale-caches", metavar="DIR")
    cl.add_argument("--include-released", action="store_true", help="also list the released stores: --move only")
    cl.add_argument("--older-than", type=float, metavar="DAYS")
    st = command("selftest", _cmd_selftest, "all tests of the suite (5 - 10 min), or only some of the file-level ones with --only")
    st.add_argument("--only", nargs="+", choices=sorted(SELFTESTS), help="run only these parts (each prints its time): " + ", ".join(sorted(SELFTESTS)))
    p.registry = registry
    return p


def commands():
    """{command name: the function that runs it} (a test pins that every command is documented in the module doc and maps onto a function)"""
    return _parser().registry


# ---------------------------------------------------------------------------------------------------------------- the commands: each is one call
def _cmd_selftest(a):
    if not a.only:
        from .selftest import run
        return run()
    import importlib
    import time
    for name in dict.fromkeys(a.only):                                 # no duplicates, the order asked for
        t0 = time.time()
        mod = importlib.import_module(f".{SELFTESTS[name]}", __package__)
        fn = getattr(mod, "run", None) or mod.selftest
        msg = fn(verbose=False) if "verbose" in fn.__code__.co_varnames[:fn.__code__.co_argcount] else fn()
        print(f"{name}: ok ({time.time() - t0:.0f} s)" + (f" {str(msg)[:150]}" if msg else ""))


def _cmd_generate(a):
    r = generate(a.dataroot, a.out, a.version, _scene_names(a), a.stride, a.res, workers=a.workers, set_name=a.set_name, lane=not a.no_lane, **_options(a))
    if r["lane"] is not None:
        from .gt_api import key_of
        name = a.set_name or ("custom" if _scene_names(a) else None)
        for system in ("bev_fusion", "lane_center"):
            print(key_of(system, r["version"], name, a.out, dataroot=a.dataroot, scenes=_scene_names(a), options=_options(a)))
    if r["masks_error"]:
        raise SystemExit(1)


def _cmd_generate_bev(a):
    from .gt_api import generate_bev_gt
    print(generate_bev_gt(a.dataroot, set_name=a.set_name, scenes=_scene_names(a), out=a.out, version=a.version, workers=a.workers, borders=tuple(a.borders)))


def _cmd_generate_lane(a):
    from .gt_api import generate_lane_gt
    keys = generate_lane_gt(a.dataroot, a.rig, a.set_name, _scene_names(a), a.out, a.version, a.workers)
    for key in (keys.values() if isinstance(keys, dict) else [keys]):
        print(key)


def _cmd_audit_lane(a):
    from . import lane_gt
    from .store_audit import audit_lane
    from .suite import DATA, default_root
    store = os.path.join(os.path.expanduser(a.out) if a.out else DATA, "lane", lane_gt.signature(rig=a.system))
    if not os.path.isdir(store):
        raise ValueError(f"no lane store {store}")
    r = audit_lane(store, a.system, a.dataroot or default_root(a.dataset), a.dataset, a.scenes, a.limit)
    if not r["ok"]:
        raise SystemExit(1)


def _cmd_area_masks(a):
    from . import lane_gt, lane_raster
    from .suite import DATA
    root = os.path.expanduser(a.out) if a.out else DATA
    store = os.path.join(root, "lane", lane_gt.signature(rig="bev_fusion"))
    if not os.path.isdir(store):
        raise ValueError(f"no BEV Fusion lane store {store}: make it with `generate-bev` or `generate-lane --rig bev_fusion`")
    r = lane_raster.build_mask_cache(store, borders=tuple(a.borders), force=a.force, workers=a.workers)
    print(f"{r['frames']} frames, {len(r['made'])} file(s) made, {len(r['files']) - len(r['made'])} already complete: {os.path.join(store, 'masks')}")


def _cmd_build(a):
    build(os.path.expanduser(a.dataroot), a.version, os.path.expanduser(a.out), _scene_names(a), a.stride, a.workers, a.res, **_options(a))


def _cmd_key(a):
    from .gt_api import gt_key
    scenes = _scene_names(a)
    key = gt_key(a.system, a.dataset, a.set_name or (None if scenes else "full"), a.out, a.dataroot, scenes)
    print(json.dumps(key.as_dict(), indent=1) if a.json else key)
    if not key.complete:
        raise SystemExit(1)


def _cmd_have(a):
    from . import current_version, sets
    want = sets.scenes(a.set_name) if a.set_name else _scene_names(a)
    if not want:
        raise ValueError("give --set, --scenes or --scenes-file")
    r = sets.stored_for(want, a.data, a.version, a.dataroot, a.stride)
    print(f"{r['have']}/{r['total']} keyframes of {r['scenes']} sequences ({r['version']}) are stored for GT version "
          f"{current_version()}; missing {len(r['missing'])}")
    if r["missing"]:
        raise SystemExit(1)


def _cmd_data(a):
    from . import sets
    r = sets.check_raw(a.dataroot, a.version, a.set_name, _scene_names(a))
    print(sets.report_raw(r))
    if r["missing"] or r["maps"]:
        raise SystemExit(1)


def _cmd_area_mask(a):
    import numpy as np
    from . import lane_gt, lane_raster
    g = lane_gt.load(a.token, rig="bev_fusion")
    m = lane_raster.area_mask(g, a.kind, a.border_m)
    print(f"{a.kind} mask of {a.token}, border {a.border_m:g} m: {int(m.sum())} of {m.size} cells ({m.shape[0]} x {m.shape[1]}, x forward 0 .. 60 m, y left -26 .. 26 m)")
    if a.png:
        import cv2
        img = np.flip(m, (0, 1)).astype(np.uint8) * 255                                  # x forward = up, y left = left, like the pictures of the GT
        if not cv2.imwrite(a.png, cv2.resize(img, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)):
            raise ValueError(f"could not write {a.png}")
        print("->", a.png)


def _cmd_view(a):
    from .view import run
    run(a.out, [x.split(":") for x in a.picks] or None, total=a.total)


def _cmd_sheets(a):
    from .view import sheets
    sheets(a.out)


def _cmd_masks(a):
    from .rig_masks import picture
    print(picture(a.out))


def _cmd_lanegraph(a):
    import cv2
    from . import lane_graph_view as V
    from .lane_gt import LaneGT
    from .light_nusc import light_nuscenes
    from .suite import AutoGT, default_root
    root = os.path.expanduser(a.dataroot or default_root(a.version))
    name, sep, k = a.pick.partition(":")
    if not sep or not k.isdigit():
        raise ValueError(f"pick: scene-0061:8 (the scene and the number of the keyframe), got {a.pick!r}")
    nusc = light_nuscenes(root, a.version, [name], cameras=V.CAMERAS, radars=False)
    gt = AutoGT(root, a.version, nusc=nusc)
    tok = gt.keyframes(name)[int(k)]
    g = LaneGT(root, a.version, nusc=nusc).graph(tok)
    cv2.imwrite(a.out, V.mosaic(gt, tok, g, title=f"{name}:{k} {g['location']}  " + V.legend_text(g)))
    print(f"{a.out}: {V.legend_text(g)}")


def _cmd_video(a):
    from .suite import default_root
    from .video import run
    run(a.scene, a.mp4, a.seconds, dataroot=a.dataroot or default_root(a.version), version=a.version)


def _cmd_check(a):
    from .checks import run
    run(os.path.expanduser(a.dataroot), a.version, a.scenes, a.stride, a.holdout)


def _cmd_report(a):
    import datetime
    from .report import report
    from .suite import DATA, default_root
    out = a.out or os.path.join(DATA, "reports", f"autogt_report_{a.version}_{datetime.date.today().isoformat()}.pdf")
    light = a.light or "trainval" in a.version
    if a.holdout and light:
        print("note: the leave-one-radar-out test needs the full devkit: skipped with the light devkit (trainval)")
    report(os.path.expanduser(a.dataroot or default_root(a.version)), a.version, out, None if a.no_data else os.path.expanduser(a.data or DATA), a.scenes, a.stride, a.holdout, a.samples,
           a.workers, light, **_options(a))


def _cmd_digest(a):
    if a.compare:
        x, y = (json.load(open(p)) for p in a.compare)
        bad = [k for k in x["digests"] if x["digests"][k] != y["digests"].get(k)] + [k for k in y["digests"] if k not in x["digests"]]
        print(f"{len(x['digests'])} frames, signatures {x['signature']} and {y['signature']}: {len(bad)} differ {bad[:3]}")
        if bad:
            raise SystemExit(1)
        return
    if not a.out:
        raise ValueError("give the file to write, or --compare A B")
    from .checks import digests
    r = digests("~/data/nuscenes", "v1.0-mini", a.stride)
    with open(a.out, "w") as f:
        json.dump(r, f)
    print(f"{len(r['digests'])} digests of signature {r['signature']} -> {a.out}")


def _cmd_clean(a):
    from .clean import clean
    clean(delete=a.delete, move=a.move, videos=a.videos, pictures=a.pictures, reports=a.reports, old_outputs=a.old_outputs,
          stale_caches=a.stale_caches, all_caches=a.all_caches, older_than=a.older_than, include_released=a.include_released)


def main(argv=None):
    a = _parser().parse_args(argv)
    try:
        a.func(a)
    except (ValueError, OSError) as e:                                   # a wrong argument or a missing file: the message and exit 2, as argparse does (AUTOGT_TRACEBACK=1 shows the traceback)
        if os.environ.get("AUTOGT_TRACEBACK"):
            raise
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
