"""Command line of the GtGenerator (from the repository root: `python -m autogt COMMAND ...`; in WSL: `wsl bash autogt/run_nice.sh -m autogt COMMAND ...`).

The command line reaches the suite ONLY through the class `GtGenerator` (autogt/api.py holds its methods): every command is ONE method call, so what the command line does is what Python does
(`python -m autogt COMMAND --help` has the options of a command; the same call in Python is `GtGenerator.<method>(...)`).

  make GT (all resumable; --out is always the ROOT that holds the stores, default data/autogt; the lane stores are below ROOT/lane; they print the GT key, see autogt/gt_api.py)
    generate DATAROOT      GtGenerator.generate          the front GT + both lane GTs + the area masks of the BEV Fusion store (everything)
    generate-bev DATAROOT  GtGenerator.generate_bev_gt   BEV Fusion: the front GT + the lane GT of its rig + the packed intersection masks
    generate-lane DATAROOT GtGenerator.generate_lane_gt  the lane GT of one system (--rig lane_center | bev_fusion | all)
    area-masks             GtGenerator.area_masks        only the packed masks of an existing BEV Fusion lane store
    build                  GtGenerator.build             the low-level writer behind `report` (one full devkit, --out required)
  look at GT
    key SYSTEM             GtGenerator.gt_key            the key of what is stored (exit 1 when incomplete), --json for the dict
    have                   GtGenerator.have              is the front GT of this generator there for these sequences? (exit 1 when not: the check before a training)
    data DATAROOT          GtGenerator.check_data        are the raw nuScenes files on disk? (exit 1 when not)
    sets / versions        GtGenerator.sets / versions   the scene sets and how much is stored / the GT versions, lane versions and releases
    area-mask TOKEN        GtGenerator.area_mask         one area mask: the number of cells, --png to draw it
    view / sheets / video / lanegraph / masks   GtGenerator.view / sheets / video / lanegraph / masks   pictures of keyframes, of the lane graph, of the coverage masks; videos
  check and clean
    audit-lane SYSTEM      GtGenerator.audit_lane        the audit of a stored lane GT before its release (exit 1 on a problem)
    check                  GtGenerator.check             consistency checks of the GT of mini        report   GtGenerator.report   the checks over all sequences, one PDF
    digest                 GtGenerator.digest            a digest per frame of mini (two generators that make the same GT give the same file)
    clean                  GtGenerator.clean             list / move generated pictures, videos, reports and old caches (--delete only for what one command rebuilds)
    selftest               GtGenerator.selftest          all tests of the suite (5 - 10 min), or parts with --only

Exit codes: 0 done, 1 the thing asked about is not there / not complete (have, data, key, audit-lane, digest --compare), 2 wrong arguments (a message, no traceback; AUTOGT_TRACEBACK=1 shows it).
"""
import argparse
import json
import os
import sys

# the storage and the writer moved to store.py / pipeline.py; the old names stay importable from here (bev_fusion tests and other projects use them)
from .pipeline import GROUP_SCENES, _group_task, _init, _same_generator, _work, _write_tokens, build, generate  # noqa: F401
from .store import _environment, _flat, canonical_of, check_resume, detect_version, load_keyframe, run_manifest, save_keyframe, write_index  # noqa: F401


# ---------------------------------------------------------------------------------------------------------------- argument helpers
from .api import SELFTESTS  # noqa: E402  (the test parts `selftest --only` can run)


def _which_scenes(p, set_flag=True):
    """--set | --scenes | --scenes-file: one of them (argparse refuses two)"""
    g = p.add_mutually_exclusive_group()
    if set_flag:
        g.add_argument("--set", dest="set_name", default=None, help="a scene set: small (the 10 sequences of mini) or full (all 850 of trainval)")
    g.add_argument("--scenes", nargs="+", help="names such as scene-0061 (default: all)")
    g.add_argument("--scenes-file", help="a file of sequence names, one per line")


def _generator_options(p):
    """the options of the generator that `generate`, `build` and `report` share"""
    p.add_argument("--no-flow", dest="use_flow", action="store_false")
    p.add_argument("--no-radar", dest="use_radar", action="store_false")
    p.add_argument("--boxes-at", choices=("camera", "annotation"), default="camera", help="time of the moving boxes: the camera time (default) or the annotation time")


def _parser():
    p = argparse.ArgumentParser(prog="python -m autogt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    registry = {}

    def command(name, method, help, **kw):
        """a command = one method of GtGenerator (autogt/api.py); the options of the parser are the arguments of the method, by name"""
        sp = sub.add_parser(name, help=help, description=help, **kw)
        sp.set_defaults(method=method)
        registry[name] = method
        return sp

    def dataroot(sp):
        sp.add_argument("dataroot", help="a nuScenes folder, e.g. ~/data/nuscenes_full (it holds v1.0-trainval/, maps/, samples/, sweeps/)")

    # ---- make GT
    g = command("generate", "generate", "everything: the front GT, the lane GT of both systems and the area masks (pipeline.generate)")
    dataroot(g)
    g.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt): the front GT goes to OUT/<signature>/, the lane GT to OUT/lane/")
    g.add_argument("--version", default=None, help="v1.0-mini | v1.0-trainval (default: the one found in DATAROOT)")
    _which_scenes(g)
    g.add_argument("--stride", type=int, default=1, help="every n-th keyframe (the lane GT is made for every keyframe: with a stride only the front GT)")
    g.add_argument("--res", type=float, default=0.5)
    g.add_argument("--workers", type=int, default=1, help="processes")
    g.add_argument("--no-lane", dest="lane", action="store_false", help="only the front GT")
    _generator_options(g)
    b = command("generate-bev", "generate_bev_gt", "BEV Fusion: the front GT + the lane GT of the rig bev_fusion + the packed intersection masks; prints the GT key (gt_api.generate_bev_gt)")
    dataroot(b)
    b.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt)")
    b.add_argument("--version", default=None)
    _which_scenes(b)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--border-m", dest="borders", type=float, nargs="+", default=[0.0, 0.5], help="the borders in metres of the intersection masks (default 0 and 0.5)")
    ln = command("generate-lane", "generate_lane_gt", "the lane GT of one system or both; prints the key(s) (gt_api.generate_lane_gt)")
    dataroot(ln)
    ln.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt): the lane GT goes to OUT/lane/")
    ln.add_argument("--version", default=None)
    _which_scenes(ln)
    ln.add_argument("--workers", type=int, default=1)
    ln.add_argument("--rig", default="all", choices=["all", "lane_center", "bev_fusion"], help="the system: Spurmitte (lane_center), BEV Fusion (bev_fusion), or both")
    au = command("audit-lane", "audit_lane", "audit a stored lane GT before it is released: every keyframe there, every file loads and holds what the version promises; exit 1 on a problem (store_audit.audit_lane)")
    au.add_argument("system", choices=["bev_fusion", "lane_center"])
    au.add_argument("--dataset", default="v1.0-trainval", help="v1.0-mini | v1.0-trainval")
    au.add_argument("--dataroot", default=None, help="a nuScenes folder (default ~/data/nuscenes_full or ~/data/nuscenes)")
    au.add_argument("--scenes", nargs="+", help="only the keyframes of these sequences")
    au.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt)")
    au.add_argument("--limit", type=int, default=None, help="check only the first N files")
    am = command("area-masks", "area_masks", "the packed area masks of an existing BEV Fusion lane store (lane_raster.build_mask_cache)")
    am.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt)")
    am.add_argument("--border-m", dest="borders", type=float, nargs="+", default=[0.0, 0.5])
    am.add_argument("--workers", type=int, default=1, help="processes (34k frames: about 11 min in one)")
    am.add_argument("--force", action="store_true", help="make them again although they are complete")
    bu = command("build", "build", "the low-level writer of generate (one full devkit, --out required; pipeline.build)")
    bu.add_argument("--dataroot", default="~/data/nuscenes")
    bu.add_argument("--version", default="v1.0-mini")
    bu.add_argument("--out", required=True)
    bu.add_argument("--scenes", nargs="+")
    bu.add_argument("--stride", type=int, default=1)
    bu.add_argument("--res", type=float, default=0.5)
    bu.add_argument("--workers", type=int, default=1)
    _generator_options(bu)
    # ---- look at GT
    k = command("key", "gt_key", "the GT key of what is stored for a system: data + code signature of every part; exit 1 when incomplete (gt_api.gt_key)")
    k.add_argument("system", choices=["bev_fusion", "lane_center"])
    k.add_argument("--dataset", default="v1.0-trainval", help="v1.0-mini | v1.0-trainval")
    _which_scenes(k)
    k.add_argument("--dataroot", default=None, help="a nuScenes folder: complete is then judged by every keyframe of its json tables")
    k.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt)")
    k.add_argument("--border-m", dest="borders", type=float, nargs="+", default=[0.0, 0.5], help="the mask borders in metres the masks part must hold (default 0 and 0.5)")
    k.add_argument("--fast", dest="deep", action="store_false", help="judge by file names only: a zero-byte file is not looked for (saves about 30 s on the full set)")
    k.add_argument("--json", action="store_true", help="print the key as json")
    h = command("have", "have", "is the stored front GT of the current generator there for these sequences? exit 1 when not (the check before a training; sets.stored_for)")
    _which_scenes(h)
    h.add_argument("--version", default=None, help="v1.0-mini | v1.0-trainval (default: the first that holds all the sequences)")
    h.add_argument("--dataroot", default=None)
    h.add_argument("--stride", type=int, default=1)
    h.add_argument("--data", default=None, help="the ROOT of the stores (default data/autogt)")
    d = command("data", "check_data", "are the raw nuScenes files the generator reads on disk (tables, maps, CAM_FRONT, radars)? exit 1 when not (sets.check_raw)")
    dataroot(d)
    d.add_argument("--version", default=None)
    _which_scenes(d)
    command("sets", "sets", "the scene sets small / full and how much of each is stored")
    command("versions", "versions", "the GT versions, the lane versions and the releases; the current one marked, which is stored")
    mk = command("area-mask", "area_mask", "one area mask of a keyframe of the BEV Fusion lane store: its size, --png to draw it (lane_raster.load_mask / area_mask)")
    mk.add_argument("token", help="the nuScenes sample token of a keyframe")
    mk.add_argument("--kind", default="intersection", choices=["intersection", "road"])
    mk.add_argument("--border-m", type=float, default=0.0)
    mk.add_argument("--out", default=None, help="the ROOT of the stores (default data/autogt)")
    mk.add_argument("--png", default=None, help="write the mask as a picture (x forward = up, y left = left)")
    v = command("view", "view", "sample pictures of keyframes (camera, semantic BEV, objects + velocity, motion)")
    v.add_argument("out")
    v.add_argument("picks", nargs="*", help="scene-0061:8 (mini, generated now) or a sample token (stored GT, mini or trainval)")
    v.add_argument("--total", action="store_true", help="4th panel: total apparent flow (ego motion included) instead of motion relative to the static world")
    command("sheets", "sheets", "overview sheets of the sample pictures").add_argument("out")
    command("masks", "masks", "the two coverage masks (BEV fusion: front; Spurmitte: surround) as one picture").add_argument("out")
    lg = command("lanegraph", "lanegraph", "picture of the lane graph of a keyframe: the six cameras and the top view")
    lg.add_argument("out", help="the png to write")
    lg.add_argument("pick", help="scene-0061:8 (the scene's 9th keyframe)")
    lg.add_argument("--version", default="v1.0-mini")
    lg.add_argument("--dataroot", default=None)
    vd = command("video", "video", "drive through the GT of one sequence at 12 Hz (H.264); mini by default, any trainval sequence with --version")
    vd.add_argument("scene")
    vd.add_argument("mp4")
    vd.add_argument("seconds", nargs="?", type=float, default=0.0)
    vd.add_argument("--version", default="v1.0-mini", help="v1.0-mini | v1.0-trainval")
    vd.add_argument("--dataroot", default=None)
    # ---- check and clean
    c = command("check", "check", "automatic consistency checks of the GT over keyframes (checks.run)")
    c.add_argument("--dataroot", default="~/data/nuscenes")
    c.add_argument("--version", default="v1.0-mini")
    c.add_argument("--scenes", nargs="+")
    c.add_argument("--stride", type=int, default=1)
    c.add_argument("--holdout", action="store_true", help="add the leave-one-radar-out test of the radar correction")
    r = command("report", "report", "evaluate the GT of all sequences, store it, write ONE PDF report with samples (report.report; made at 0.5 m)")
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
    dg = command("digest", "digest", "a digest per frame of nuScenes mini: two generators that make the same GT give the same file (checks.digests)")
    dg.add_argument("out", nargs="?", help="write the digests to this json file")
    dg.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two digest files: exit 1 when any frame differs")
    dg.add_argument("--stride", type=int, default=2)
    cl = command("clean", "clean", "list / move generated videos, pictures, reports, old outputs and caches; --delete only for what one command rebuilds (clean.clean)")
    for flag in ("--videos", "--pictures", "--reports", "--old-outputs"):
        cl.add_argument(flag, action="store_true")
    act = cl.add_mutually_exclusive_group()
    act.add_argument("--move", action="store_true")
    act.add_argument("--delete", action="store_true")
    cl.add_argument("--all-caches", metavar="DIR")
    cl.add_argument("--stale-caches", metavar="DIR")
    cl.add_argument("--include-released", action="store_true", help="also list the released stores: --move only")
    cl.add_argument("--older-than", type=float, metavar="DAYS")
    st = command("selftest", "selftest", "all tests of the suite (5 - 10 min), or only some of the file-level ones with --only")
    st.add_argument("--only", nargs="+", choices=sorted(SELFTESTS), help="run only these parts (each prints its time): " + ", ".join(sorted(SELFTESTS)))
    p.registry = registry
    return p


def commands():
    """{command name: the method of GtGenerator that runs it} (a test pins that every command is documented in the module doc and maps onto a method)"""
    return _parser().registry


# ---------------------------------------------------------------------------------------------------------------- one call, one exit code
_NOT_ARGUMENTS = ("cmd", "method", "json", "scenes_file")


def _kwargs(a):
    """the arguments of the method: every option that is given (the others keep the default of the method); --scenes-file is read here (one name per line, # comments)"""
    kw = {k: v for k, v in vars(a).items() if k not in _NOT_ARGUMENTS and v is not None}
    if "borders" in kw:
        kw["borders"] = tuple(kw["borders"])
    if getattr(a, "scenes_file", None):
        with open(a.scenes_file, encoding="utf-8") as f:
            kw["scenes"] = [x for x in (ln.split("#")[0].strip() for ln in f) if x]
    return kw


def _emit(r, as_json=False):
    """print what the method returned and give the exit code: a GtKey (or a dict of them, or the `keys` of a dict) as its line; `text` of a dict; 1 when `ok` is False
    (a dict without `ok`: a key that is not complete)"""
    if hasattr(r, "as_dict"):
        keys = [r]
    elif isinstance(r, dict) and r and all(hasattr(v, "as_dict") for v in r.values()):
        keys = list(r.values())
    else:
        keys = list(r.get("keys", {}).values()) if isinstance(r, dict) else []
    for k in keys:
        print(json.dumps(k.as_dict(), indent=1) if as_json else k)
    if isinstance(r, dict) and r.get("text"):
        print(r["text"])
    if isinstance(r, dict) and "ok" in r:
        return 0 if r["ok"] else 1
    return 1 if any(not k.complete for k in keys) else 0


def main(argv=None):
    a = _parser().parse_args(argv)
    from .suite import GtGenerator                                              # the class interface: the only way the command line reaches the suite
    try:
        code = _emit(getattr(GtGenerator, a.method)(**_kwargs(a)), getattr(a, "json", False))
    except (ValueError, OSError) as e:                                   # a wrong argument or a missing file: the message and exit 2, as argparse does (AUTOGT_TRACEBACK=1 shows the traceback)
        if os.environ.get("AUTOGT_TRACEBACK"):
            raise
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    if code:
        raise SystemExit(code)
