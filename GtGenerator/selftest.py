"""Self-test of GtGenerator: python -m GtGenerator selftest [--only NAME ...] [--all] [--root FOLDER]

CASES is the table of the tests: (part, case, function, needs). A NAME of --only is a part (all its cases) or a case. `needs`:
    None     no data: synthetic worlds, made-up maps, the synthetic fixture of a nuScenes folder (fixture.py), the tables in git
    "mini"   nuScenes mini at --root (default ~/data/nuscenes: the json tables and the map expansion); the devkit is loaded once per process
             (front_selftest.devkit) and shared by the cases that need it
    "asked"  only when its part or case is named: the scan of the folders of other sessions (a half-edited file there must not fail the gate)
Without mini the cases that need it are listed as SKIPPED and the run passes; `--all` (the gate of a signature step and of a release) fails
instead. A case function takes `root` when it needs mini and nothing otherwise; what it returns (a str or None) is printed after its name.
"""
import importlib
import inspect
import os
import time

CASES = [
    # part         case                 module:function                         needs
    ("flow",       "flow",              "flow_selftest:run",                    None),
    ("front",      "ego_velocity",      "front_selftest:ego_velocity",          None),
    ("front",      "raster",            "front_selftest:raster",                None),
    ("front",      "coverage_rules",    "front_selftest:coverage_rules",        None),
    ("front",      "motion_arrows",     "front_selftest:motion_arrows",         None),
    ("versions",   "tables",            "versions_selftest:tables",             None),
    ("store",      "index",             "store_selftest:index",                 None),
    ("store",      "sets_manifest",     "store_selftest:sets_manifest",         None),
    ("clean",      "clean",             "clean:selftest",                       None),
    ("lane_graph", "lane_graph_toy",    "lane_graph_selftest:toy",              None),
    ("lane_toy",   "lane_toy",          "lane_tiers:selftest",                  None),
    ("lane_tiers", "lane_tiers",        "lane_tiers_selftest:run",              None),
    ("gt_api",     "gt_api_toy",        "gt_api_selftest:toy",                  None),
    ("generator",  "generator_toy",     "generator_selftest:toy",               None),
    ("cli",        "cli_toy",           "cli_selftest:toy",                     None),
    ("rig_masks",  "rig_masks_toy",     "rig_masks:selftest_toy",               None),
    ("interface",  "interface_rules",   "interface_selftest:package",           None),
    ("interface",  "interface_tree",    "interface_selftest:tree",              "asked"),
    ("front",      "real_keyframes",    "front_selftest:real_keyframes",        "mini"),
    ("front",      "velocity_layers",   "front_selftest:velocity_layers",       "mini"),
    ("front",      "coverage",          "front_selftest:coverage",              "mini"),
    ("versions",   "pins",              "versions_selftest:pins",               "mini"),
    ("versions",   "agents_doc",        "versions_selftest:agents_doc",         "mini"),
    ("store",      "generate",          "store_selftest:generate",              "mini"),
    ("store",      "set_runs",          "store_selftest:set_runs",              "mini"),
    ("store",      "resume",            "store_selftest:resume",                "mini"),
    ("store",      "raw_check",         "store_selftest:raw_check",             "mini"),
    ("store",      "release",           "store_selftest:release",               "mini"),
    ("front",      "light_devkit",      "front_selftest:light_devkit",          "mini"),
    ("front",      "report",            "front_selftest:report",                "mini"),
    ("front",      "light_cache",       "front_selftest:light_cache",           "mini"),
    ("front",      "video",             "front_selftest:video",                 "mini"),
    ("lane_graph", "lane_graph",        "lane_graph_selftest:mini",             "mini"),
    ("gt_api",     "gt_api",            "gt_api_selftest:mini",                 "mini"),
    ("generator",  "generator",         "generator_selftest:mini",              "mini"),
    ("cli",        "cli",               "cli_selftest:mini",                    "mini"),
    ("rig_masks",  "rig_masks",         "rig_masks:selftest",                   "mini"),
]
NAMES = sorted({x for part, case, _, _ in CASES for x in (part, case)})       # what --only takes


def mini_root(root=None):
    """the nuScenes mini folder (default ~/data/nuscenes) when it holds the json tables and the map expansion, else None"""
    root = os.path.expanduser(root or "~/data/nuscenes")
    have = os.path.exists(os.path.join(root, "v1.0-mini", "scene.json")) and os.path.isdir(os.path.join(root, "maps", "expansion"))
    return root if have else None


def chosen(only=None):
    """the cases of a run, in table order: all but the "asked" ones, or those of the parts and cases named in `only`"""
    if not only:
        return [c for c in CASES if c[3] != "asked"]
    unknown = sorted(set(only) - set(NAMES))
    if unknown:
        raise ValueError(f"unknown test part or case {unknown}: {', '.join(NAMES)}")
    return [c for c in CASES if c[0] in only or c[1] in only]


def run(only=None, root=None, all_=False, verbose=True):
    """run the chosen cases (an AssertionError at the first that fails) -> {"ok", "text": the summary, "lines": one per case, "ran", "skipped"};
    `verbose` prints the line of a case as it ends. With `all_`, a case that needs mini where mini is not there makes the run fail before
    anything runs: ok False"""
    cases, mini = chosen(only), mini_root(root)
    where_ = os.path.expanduser(root or "~/data/nuscenes")
    skipped = [case for _, case, _, needs in cases if needs == "mini" and mini is None]
    if skipped and all_:
        return {"ok": False, "ran": 0, "skipped": skipped, "lines": [],
                "text": f"--all: no nuScenes mini at {where_} (the json tables, the map expansion), so {len(skipped)} case(s) cannot run: {', '.join(skipped)}"}
    lines, t_all = [], time.time()
    for _, case, where, needs in cases:
        if case in skipped:
            lines.append(f"{case}: SKIPPED (no nuScenes mini at {where_})")
        else:
            module, name = where.split(":")
            fn = getattr(importlib.import_module(f".{module}", __package__), name)
            kwargs = {"verbose": False} if "verbose" in inspect.signature(fn).parameters else {}
            t0 = time.time()
            msg = fn(mini, **kwargs) if needs == "mini" else fn(**kwargs)
            lines.append(f"{case}: ok ({time.time() - t0:.0f} s)" + (f" {msg}" if isinstance(msg, str) and msg else ""))
        if verbose:
            print(lines[-1], flush=True)
    ran = len(cases) - len(skipped)
    summary = f"GtGenerator selftest ok: {ran} case(s) in {time.time() - t_all:.0f} s" + (f", {len(skipped)} SKIPPED without nuScenes mini" if skipped else "")
    return {"ok": True, "text": summary, "lines": lines, "ran": ran, "skipped": skipped}
