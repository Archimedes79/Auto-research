p = "autogt/cli.py"
s = open(p, encoding="utf-8").read()
a = s.index("def _boxes_option(a):")
new = '''def _options(a):
    """the generator options of the command line (flow and radar on unless switched off; boxes_at only when it is not the default)"""
    return {"use_flow": not a.no_flow, "use_radar": not a.no_radar, **({} if a.boxes_at == "camera" else {"boxes_at": a.boxes_at})}


def _generator_args():
    """the options that `generate`, `build` and `report` share (a parent parser)"""
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--res", type=float, default=0.5)
    g.add_argument("--stride", type=int, default=1, help="every n-th keyframe")
    g.add_argument("--workers", type=int, default=1, help="processes")
    g.add_argument("--no-flow", action="store_true")
    g.add_argument("--no-radar", action="store_true")
    g.add_argument("--boxes-at", choices=("camera", "annotation"), default="camera",
                   help="time of the moving boxes: the camera time (default) or the annotation time")
    return g


def _parser():
    p = argparse.ArgumentParser(prog="python -m autogt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    gen = _generator_args()
    sub.add_parser("selftest")
    sub.add_parser("versions", help="the GT versions (autogt/versions.csv, D29): the current one marked, which GT is stored")
    sub.add_parser("sets", help="the scene sets small / full (autogt/sets/): sequences, keyframes, how much of each is stored")
    da = sub.add_parser("data", help="check that the raw nuScenes files the generator reads are on disk (tables, maps, CAM_FRONT, radars)")
    da.add_argument("dataroot", help="a nuScenes folder, e.g. ~/data/nuscenes_full")
    da.add_argument("--version", default=None)
    da.add_argument("--set", dest="set_name", default=None, help="check only the sequences of a set: small | full")
    da.add_argument("--scenes", nargs="*")
    ge = sub.add_parser("generate", parents=[gen], help="read nuScenes from DATAROOT and write the GT of every keyframe (the entry point)")
    ge.add_argument("dataroot", help="a nuScenes folder, e.g. ~/data/nuscenes_full (it holds v1.0-trainval/, maps/, samples/, sweeps/)")
    ge.add_argument("--out", default=None, help="folder of the stored GT (default data/autogt); the files go to OUT/<signature>/")
    ge.add_argument("--version", default=None, help="v1.0-mini | v1.0-trainval (default: the one found in DATAROOT)")
    ge.add_argument("--set", dest="set_name", default=None, help="a scene set: small (the 10 sequences of mini) or full (all of trainval)")
    ge.add_argument("--scenes", nargs="*", help="names such as scene-0061 (default: all)")
    b = sub.add_parser("build", parents=[gen], help="the low-level writer of generate (one full devkit, --out required)")
    b.add_argument("--dataroot", default="~/data/nuscenes")
    b.add_argument("--version", default="v1.0-mini")
    b.add_argument("--out", required=True)
    b.add_argument("--scenes", nargs="*")
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
    sub.add_parser("sheets").add_argument("out")
    r = sub.add_parser("report", parents=[gen], help="evaluate the GT of all sequences, store it, write one PDF report with samples")
    r.add_argument("--dataroot", default=None, help="default ~/data/nuscenes (mini) or ~/data/nuscenes_full (trainval)")
    r.add_argument("--version", default="v1.0-mini")
    r.add_argument("--light", action="store_true", help="memory-light devkit (automatic for trainval)")
    r.add_argument("--out", default=None, help="report PDF (default data/autogt/reports/autogt_report_<version>_<date>.pdf)")
    r.add_argument("--data", default=None, help="cache directory (default data/autogt); --no-data: do not store")
    r.add_argument("--no-data", action="store_true")
    r.add_argument("--scenes", nargs="*")
    r.add_argument("--samples", type=int, default=10, help="sample pictures in the report (evenly spread over the sequences)")
    r.add_argument("--holdout", action="store_true", help="add the leave-one-radar-out test of the radar correction")
    cl = sub.add_parser("clean", help="list / move / delete generated videos, pictures, old outputs, stale caches")
    for flag in ("--videos", "--pictures", "--reports", "--old-outputs", "--move", "--delete"):
        cl.add_argument(flag, action="store_true")
    cl.add_argument("--all-caches", metavar="DIR")
    cl.add_argument("--stale-caches", metavar="DIR")
    cl.add_argument("--include-released", action="store_true", help="also list the caches of released GT versions (versions.csv): --move only")
    cl.add_argument("--older-than", type=float, metavar="DAYS")
    d = sub.add_parser("video", help="drive through the GT of one sequence at 12 Hz (H.264); mini by default, any trainval sequence with --version")
    d.add_argument("scene")
    d.add_argument("mp4")
    d.add_argument("seconds", nargs="?", type=float, default=0.0)
    d.add_argument("--version", default="v1.0-mini", help="v1.0-mini | v1.0-trainval")
    d.add_argument("--dataroot", default=None, help="default ~/data/nuscenes (mini) or ~/data/nuscenes_full (trainval)")
    return p


def _cmd_selftest(a):
    from .selftest import run
    run()


def _cmd_data(a):
    from . import rawdata
    r = rawdata.check(a.dataroot, a.version, a.set_name, a.scenes)
    print(rawdata.report(r))
    if r["missing"] or r["maps"]:
        raise SystemExit(1)


def _cmd_generate(a):
    generate(a.dataroot, a.out, a.version, a.scenes, a.stride, a.res, workers=a.workers, set_name=a.set_name, **_options(a))


def _cmd_build(a):
    build(os.path.expanduser(a.dataroot), a.version, a.out, a.scenes, a.stride, a.workers, a.res, **_options(a))


def _cmd_check(a):
    from .checks import run
    run(os.path.expanduser(a.dataroot), a.version, a.scenes, a.stride, a.holdout)


def _cmd_view(a):
    from .view import run
    run(a.out, [x.split(":") for x in a.picks] or None, total=a.total)


def _cmd_sheets(a):
    from .view import sheets
    sheets(a.out)


def _cmd_report(a):
    import datetime
    from .report import report
    from .suite import DATA, default_root
    out = a.out or os.path.join(DATA, "reports", f"autogt_report_{a.version}_{datetime.date.today().isoformat()}.pdf")
    data = None if a.no_data else (a.data or DATA)
    report(os.path.expanduser(a.dataroot or default_root(a.version)), a.version, out, data, a.scenes, a.stride, a.holdout, a.samples,
           a.workers, a.light or "trainval" in a.version, **_options(a))


def _cmd_clean(a):
    from .clean import clean
    clean(delete=a.delete, move=a.move, videos=a.videos, pictures=a.pictures, reports=a.reports, old_outputs=a.old_outputs,
          stale_caches=a.stale_caches, all_caches=a.all_caches, older_than=a.older_than, include_released=a.include_released)


def _cmd_video(a):
    from .suite import default_root
    from .video import run
    run(a.scene, a.mp4, a.seconds, dataroot=a.dataroot or default_root(a.version), version=a.version)


COMMANDS = {"selftest": _cmd_selftest, "versions": lambda a: _print_versions(), "sets": lambda a: _print_sets(), "data": _cmd_data,
            "generate": _cmd_generate, "build": _cmd_build, "check": _cmd_check, "view": _cmd_view, "sheets": _cmd_sheets,
            "report": _cmd_report, "clean": _cmd_clean, "video": _cmd_video}
CHECKS_ARGUMENTS = ("data", "generate", "clean")           # these commands raise ValueError for a wrong argument: the message, no traceback


def main(argv=None):
    a = _parser().parse_args(argv)
    try:
        COMMANDS[a.cmd](a)
    except ValueError as e:
        if a.cmd not in CHECKS_ARGUMENTS:
            raise
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()
'''
s = s[:a] + new
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("cli rewritten", len(s.splitlines()), "lines")
