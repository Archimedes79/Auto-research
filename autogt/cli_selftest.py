"""Self-test of the command line (cli.py) and the store helpers behind it: every command is ONE method of AutoGT (api.py), which makes ONE call of a function of the suite (a recorder stands in for the
function and the test asserts the call and its arguments), wrong arguments give a message and exit 2 and never a traceback, "not there" gives exit 1, every --help works, the key line printed is the key of the stored GT; and the
consistency of the stores (listing by file name, a rerun that writes nothing keeps the manifest, versions tables).

    python -m autogt.cli_selftest [--root ~/data/nuscenes]      (also run by `python -m autogt selftest`)
"""
import contextlib
import io
import os
import tempfile

from . import cli


class _Patch:
    """set attributes of modules for the length of a with block"""

    def __init__(self, *items):
        self.items, self.old = items, []

    def __enter__(self):
        for obj, name, value in self.items:
            self.old.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        return self

    def __exit__(self, *exc):
        for obj, name, value in reversed(self.old):
            setattr(obj, name, value)


def _recorder(ret=None):
    calls = []

    def fn(*a, **k):
        calls.append((a, k))
        return ret
    return fn, calls


def _run(argv):
    """cli.main(argv) -> (exit code, stdout, stderr)"""
    out, err, code = io.StringIO(), io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cli.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return code, out.getvalue(), err.getvalue()


class _FakeKey:
    """what gt_key returns in the tests of the key command"""

    def __init__(self, complete):
        self.complete = complete

    def as_dict(self):
        return {"id": "abc", "complete": self.complete}

    def __str__(self):
        return f"fake key complete={self.complete}"


def _mapping_checks():
    from . import checks, clean, gt_api, lane_raster, report, rig_masks, sets, store_audit, video, view
    from . import lane_gt
    from . import pipeline  # noqa: F401
    # the map of the module doc is the map of the code: every command is ONE method of AutoGT, named in the doc; no method is without a command, and the command line imports nothing but the class
    from .api import Interface
    from .suite import AutoGT
    expected = {"generate": "generate", "generate-bev": "generate_bev_gt", "generate-lane": "generate_lane_gt", "area-masks": "area_masks", "audit-lane": "audit_lane", "build": "build",
                "key": "gt_key", "have": "have", "data": "check_data", "sets": "sets", "versions": "versions", "area-mask": "area_mask", "view": "view", "sheets": "sheets", "masks": "masks",
                "lanegraph": "lanegraph", "video": "video", "check": "check", "report": "report", "digest": "digest", "clean": "clean", "selftest": "selftest"}
    cmds = cli.commands()
    assert cmds == expected, (set(cmds) ^ set(expected), {k: v for k, v in cmds.items() if expected.get(k) != v})
    assert all(callable(getattr(AutoGT, m)) for m in expected.values()), "every command is a method of AutoGT"
    assert {m for m in vars(Interface) if not m.startswith("_")} == set(expected.values()), "no method of the interface without a command, no command without a method"
    assert all(f"AutoGT.{m}" in cli.__doc__ or m in cli.__doc__ for m in expected.values()), [m for m in expected.values() if m not in cli.__doc__]
    import ast
    tree = ast.parse(open(cli.__file__, encoding="utf-8").read())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 1}
    assert imported <= {"pipeline", "store", "api", "suite"}, f"the command line reaches the suite only through AutoGT (pipeline / store are the old names re-exported): {imported}"
    assert not any(isinstance(n, ast.Import) and any(x.name.startswith("autogt") for x in n.names) for n in ast.walk(tree))
    # every command has a --help that works
    for name in cmds:
        code, out, err = _run([name, "--help"])
        assert code == 0 and out.startswith("usage:") and not err, (name, code, err[:200])

    # generate: all options reach pipeline.generate, the keys of both systems are printed
    gen, calls = _recorder({"lane": {"x": 1}, "version": "v1.0-mini", "masks_error": None, "lane_rigs": {}})
    with _Patch((pipeline, "generate", gen), (gt_api, "key_of", lambda system, *a, **k: f"key of {system}")):
        code, out, err = _run(["generate", "~/r", "--scenes", "scene-0061", "scene-0103", "--stride", "20", "--workers", "2", "--out", "o", "--no-flow", "--boxes-at", "annotation"])
    assert code == 0 and calls == [(("~/r", "o", None, ["scene-0061", "scene-0103"], 20, 0.5), dict(workers=2, set_name=None, lane=True, verbose=True, use_flow=False, use_radar=True, boxes_at="annotation"))], calls
    assert out.splitlines() == ["key of bev_fusion", "key of lane_center"], out
    gen, calls = _recorder({"lane": None, "version": "v1.0-mini", "masks_error": None})
    with _Patch((pipeline, "generate", gen)):
        code, out, _ = _run(["generate", "~/r", "--set", "small", "--no-lane"])
    assert code == 0 and calls[0][0][3] is None and calls[0][1]["set_name"] == "small" and calls[0][1]["lane"] is False and out == "", (calls, out)
    gen, _ = _recorder({"lane": {}, "version": "v1.0-mini", "masks_error": "OSError: disk full"})
    with _Patch((pipeline, "generate", gen), (gt_api, "key_of", lambda *a, **k: "k")):
        assert _run(["generate", "~/r"])[0] == 1, "a failed mask cache is a failed command"
    # generate-bev / generate-lane: the key is printed, the options are passed on
    fn, calls = _recorder(_FakeKey(True))
    with _Patch((gt_api, "generate_bev_gt", fn)):
        code, out, _ = _run(["generate-bev", "~/r", "--set", "full", "--out", "o", "--workers", "3", "--border-m", "0", "0.5", "1"])
    assert code == 0 and out.strip() == "fake key complete=True" and calls == [((), dict(dataroot="~/r", set_name="full", out="o", workers=3, borders=(0.0, 0.5, 1.0)))], calls
    fn, calls = _recorder({"lane_center": _FakeKey(True), "bev_fusion": _FakeKey(True)})
    with _Patch((gt_api, "generate_lane_gt", fn)):
        code, out, _ = _run(["generate-lane", "~/r", "--scenes", "scene-0061"])
    assert code == 0 and len(out.splitlines()) == 2 and calls == [((), dict(dataroot="~/r", rig="all", scenes=["scene-0061"], workers=1))], calls
    fn, calls = _recorder(_FakeKey(True))
    with _Patch((gt_api, "generate_lane_gt", fn)):
        _run(["generate-lane", "~/r", "--rig", "lane_center", "--set", "small", "--out", "o"])
    assert calls == [((), dict(dataroot="~/r", rig="lane_center", set_name="small", out="o", workers=1))], calls
    # area-masks needs an existing store; area-mask draws one
    with tempfile.TemporaryDirectory() as d:
        code, out, err = _run(["area-masks", "--out", d])
        assert code == 2 and err.startswith("error:") and "generate-bev" in err, (code, err)
        os.makedirs(os.path.join(d, "lane", lane_gt.signature(rig="bev_fusion")))
        fn, calls = _recorder({"frames": 5, "made": ["a"], "files": ["a", "b"]})
        with _Patch((lane_raster, "build_mask_cache", fn)):
            code, out, _ = _run(["area-masks", "--out", d, "--border-m", "0.5", "--workers", "2", "--force"])
        assert code == 0 and calls[0][1] == dict(borders=(0.5,), force=True, workers=2) and "5 frames" in out, (calls, out)
    # audit-lane: the store of the system, the arguments, exit 1 on a problem
    with tempfile.TemporaryDirectory() as d:
        code, out, err = _run(["audit-lane", "lane_center", "--out", d])
        assert code == 2 and "no lane store" in err, (code, err)
        os.makedirs(os.path.join(d, "lane", lane_gt.signature(rig="lane_center")))
        for ok, code_want in ((True, 0), (False, 1)):
            fn, calls = _recorder({"ok": ok, "problems": [], "stats": {}, "lines": []})
            with _Patch((store_audit, "audit_lane", fn)):
                code, _, _ = _run(["audit-lane", "lane_center", "--out", d, "--dataset", "v1.0-mini", "--dataroot", "r", "--scenes", "scene-0061", "--limit", "3"])
            assert code == code_want and calls == [((os.path.join(d, "lane", lane_gt.signature(rig="lane_center")), "lane_center", "r", "v1.0-mini", ["scene-0061"], 3), dict(verbose=True))], (code, calls)
    # selftest --only: a part alone, an unknown part is refused
    code, out, err = _run(["selftest", "--only", "clean"])
    assert code == 0 and out.startswith("clean: ok"), (code, out, err[-200:])
    assert _run(["selftest", "--only", "nothing"])[0] == 2
    # build
    fn, calls = _recorder(0)
    with _Patch((pipeline, "build", fn)):
        assert _run(["build", "--out", "o", "--scenes", "scene-0061", "--workers", "2", "--no-radar"])[0] == 0
    assert calls == [((os.path.expanduser("~/data/nuscenes"), "v1.0-mini", "o", ["scene-0061"], 1, 2, 0.5), dict(use_flow=True, use_radar=False))], calls
    # key: the key and its exit code; --json
    for complete, code_want in ((True, 0), (False, 1)):
        fn, calls = _recorder(_FakeKey(complete))
        with _Patch((gt_api, "gt_key", fn)):
            code, out, _ = _run(["key", "bev_fusion", "--set", "small", "--dataset", "v1.0-mini", "--out", "o"])
        assert code == code_want and out.strip() == f"fake key complete={complete}" and calls == [((), dict(system="bev_fusion", dataset="v1.0-mini", set_name="small", out="o"))], (calls, out, code)
    fn, calls = _recorder(_FakeKey(True))
    with _Patch((gt_api, "gt_key", fn)):
        code, out, _ = _run(["key", "lane_center", "--json"])
        assert code == 0 and '"id": "abc"' in out and calls[0] == ((), dict(system="lane_center", dataset="v1.0-trainval")), calls
        _run(["key", "lane_center", "--scenes", "scene-0061"])
        assert calls[1] == ((), dict(system="lane_center", dataset="v1.0-trainval", scenes=["scene-0061"])), calls[1]
    # have / data: exit 1 when something is missing
    fn, calls = _recorder({"have": 3, "total": 5, "scenes": 1, "version": "v1.0-mini", "missing": ["a", "b"]})
    with _Patch((sets, "stored_for", fn)):
        code, out, _ = _run(["have", "--scenes", "scene-0061", "--data", "d"])
    assert code == 1 and "3/5" in out and calls[0][0][1:] == ("d", None, None, 1), (code, calls)
    with _Patch((sets, "stored_for", _recorder({"have": 5, "total": 5, "scenes": 1, "version": "v1.0-mini", "missing": []})[0])):
        assert _run(["have", "--set", "small"])[0] == 0
    code, _, err = _run(["have"])
    assert code == 2 and "--set, --scenes or --scenes-file" in err, (code, err)
    with _Patch((sets, "check_raw", _recorder({"missing": ["x"], "maps": []})[0]), (sets, "report_raw", lambda r: "report")):
        assert _run(["data", "~/r"])[0] == 1
    with _Patch((sets, "check_raw", _recorder({"missing": [], "maps": []})[0]), (sets, "report_raw", lambda r: "report")):
        assert _run(["data", "~/r", "--set", "small"])[0] == 0
    # report: no --res (it was accepted and dropped), the hold-out note for the light devkit
    code, _, err = _run(["report", "--res", "0.25"])
    assert code == 2 and "--res" in err, "report has no --res"
    fn, calls = _recorder(None)
    with _Patch((report, "report", fn)):
        code, out, _ = _run(["report", "--version", "v1.0-trainval", "--holdout", "--no-data", "--scenes", "scene-0001", "--workers", "2"])
    assert code == 0 and "skipped with the light devkit" in out and calls[0][0][2:] == (calls[0][0][2], None, ["scene-0001"], 1, True, 10, 2, True), calls
    # clean: move and delete exclude each other, the arguments arrive
    code, _, err = _run(["clean", "--move", "--delete"])
    assert code == 2 and "not allowed" in err, (code, err)
    fn, calls = _recorder([])
    with _Patch((clean, "clean", fn)):
        _run(["clean", "--videos", "--reports", "--stale-caches", "d", "--older-than", "5", "--move"])
    assert calls == [((), dict(delete=False, move=True, videos=True, pictures=False, reports=True, old_outputs=False, stale_caches="d", all_caches=None, older_than=5.0, include_released=False))], calls
    # the picture and video commands
    fn, calls = _recorder(None)
    with _Patch((view, "run", fn)):
        _run(["view", "o.png", "scene-0061:8", "--total"])
    assert calls == [(("o.png", [["scene-0061", "8"]]), dict(total=True))], calls
    with _Patch((view, "sheets", fn)):
        _run(["sheets", "dir"])
    with _Patch((rig_masks, "picture", _recorder("picture")[0])):
        assert _run(["masks", "m.png"])[1].strip() == "picture"
    fn, calls = _recorder(None)
    with _Patch((video, "run", fn)):
        _run(["video", "scene-0061", "o.mp4", "2.5", "--version", "v1.0-mini", "--dataroot", "r"])
    assert calls == [(("scene-0061", "o.mp4", 2.5), dict(dataroot="r", version="v1.0-mini"))], calls
    fn, calls = _recorder(None)
    with _Patch((checks, "run", fn)):
        _run(["check", "--scenes", "scene-0061", "--stride", "5", "--holdout"])
    assert calls == [((os.path.expanduser("~/data/nuscenes"), "v1.0-mini", ["scene-0061"], 5, True), {})], calls


def _error_checks(root):
    """wrong arguments: a message and exit 2, no traceback; not there: exit 1; the traceback on request"""
    cases = [(["generate", "~/r", "--workers", "0"], "workers"), (["generate", "~/r", "--res", "0"], "res"), (["generate", "~/r", "--stride", "0"], "stride"),
             (["generate", root, "--set", "tiny"], "no scene set"), (["generate", "~/r", "--set", "small", "--scenes", "scene-0061"], "not allowed"),
             (["have", "--set", "small", "--scenes", "x"], "not allowed"), (["generate", root, "--scenes-file", "/nonexistent/file.txt"], "nonexistent"),
             (["lanegraph", "o.png", "scene-0061"], "scene-0061:8"), (["digest"], "--compare"), (["key", "surround"], "invalid choice"),
             (["key", "lane_center", "--dataset", "v1.0-mini", "--dataroot", root, "--scenes", "scene-9999"], "scene-9999"),
             (["generate", root, "--scenes", "scene-9999"], "scene-9999"), (["build", "--out", "o", "--workers", "0"], "workers"), (["digest", "--compare", "/nonexistent/a.json", "/nonexistent/b.json"], "nonexistent")]
    for argv, word in cases:
        code, out, err = _run(argv)
        assert code == 2 and word in err and "Traceback" not in err, (argv, code, err[-300:])
    assert "Traceback" not in _run(["generate", "~/r", "--workers", "0"])[2]
    os.environ["AUTOGT_TRACEBACK"] = "1"
    try:
        try:
            _run(["generate", "~/r", "--workers", "0"])
            raise AssertionError("AUTOGT_TRACEBACK=1 must raise")
        except ValueError as e:
            assert "workers" in str(e)
    finally:
        del os.environ["AUTOGT_TRACEBACK"]
    # not there is exit 1, with the key printed: an empty store folder for mini, the line is the key of the stored GT
    from . import gt_api
    with tempfile.TemporaryDirectory() as d:
        code, out, err = _run(["key", "bev_fusion", "--dataset", "v1.0-mini", "--scenes", "scene-0061", "--out", d])
        assert code == 1 and out.strip() == str(gt_api.gt_key("bev_fusion", "v1.0-mini", None, d, None, ["scene-0061"])) and "INCOMPLETE" in out, (code, out)
        code, out, _ = _run(["key", "lane_center", "--dataset", "v1.0-mini", "--set", "small", "--out", d, "--json"])
        assert code == 1 and '"complete": false' in out, out
        code, out, err = _run(["have", "--scenes", "scene-0061", "--data", d, "--dataroot", root, "--version", "v1.0-mini"])
        assert code == 1 and "missing" in out, (code, out, err)


def _store_checks(root):
    """the store helpers: listing by name, a rerun that writes nothing keeps the manifest, an unknown scene is a ValueError, the versions tables"""
    import json
    from . import pipeline, sets, suite, versions
    from .store import check_arguments, front_keys, names, read_index
    with tempfile.TemporaryDirectory() as d:
        for f in ("a" * 32 + ".npz", "b" * 32 + ".npz", "c" * 32 + ".npz.123.tmp.npz", "index.json", "d" * 32 + ".pkl.gz"):
            open(os.path.join(d, f), "wb").write(b"x")
        os.makedirs(os.path.join(d, "masks"))
        assert front_keys(d) == {"a" * 32, "b" * 32} and names(d, ".pkl.gz") == {"d" * 32} and names(os.path.join(d, "none"), ".npz") == set(), "listing by file name: no tmp files, no folders"
        assert sets.held(d) == front_keys(d)
        assert read_index(d) == {} and read_index(os.path.join(d, "none")) == {}, "a damaged or missing index is an empty dict"
        # missing(): one listing for many tokens, one stat for few: the same answer
        sig = suite.current_signature()
        os.makedirs(os.path.join(d, sig))
        open(os.path.join(d, sig, "a" * 32 + ".npz"), "wb").write(b"x")
        few, many = ["a" * 32, "z" * 32], ["a" * 32] + ["z%031d" % i for i in range(60)]
        assert suite.missing(few, data=d) == ["z" * 32] and suite.missing(many, data=d) == many[1:]
    for kw in (dict(stride=0), dict(res=0), dict(ss=0), dict(workers=0), dict(set_name="small", scenes=["scene-0061"])):
        try:
            check_arguments(**kw)
            raise AssertionError(f"{kw} must be refused")
        except ValueError:
            pass
    for fn, kw in ((suite.scene_tokens, dict(scene="scene-9999", version="v1.0-mini", dataroot=root)),):
        try:
            fn(**kw)
            raise AssertionError("an unknown scene must be a ValueError, not a StopIteration")
        except ValueError as e:
            assert "scene-9999" in str(e)
    # a rerun that writes nothing keeps the manifest of the run that made the store (commit, release, environment)
    with tempfile.TemporaryDirectory() as d:
        kw = dict(scenes=["scene-0061"], stride=40, verbose=False, lane=False)
        r = pipeline.generate(root, d, **kw)
        assert r["written"] >= 1
        path = os.path.join(d, r["signature"], "index.json")
        idx = json.load(open(path))
        idx["commit"], idx["release"] = "abc1234", True
        json.dump(idx, open(path, "w"))
        again = pipeline.generate(root, d, **kw)
        kept = json.load(open(path))
        assert again["written"] == 0 and kept["commit"] == "abc1234" and kept["release"] is True and again["release"] is True, "a rerun that writes nothing leaves the manifest alone"
        pipeline.generate(root, d, scenes=["scene-0103"], stride=40, verbose=False, lane=False)
        assert set(json.load(open(path))["scenes"]) == {"scene-0061", "scene-0103"}, "a rerun that writes something updates it"
        assert json.load(open(path))["complete"] is True, "a store that grew by another run is complete (the provisional index of that run must not stay)"
        # the key of a store written with a stride is never complete, and a key made with generator options reads ITS store
        from . import gt_api
        from .suite import scene_tokens
        assert not gt_api.key_of("bev_fusion", "v1.0-mini", "custom", d, parts=("front",), scenes=["scene-0061"]).complete, "a stride store is not complete"
        pipeline.generate(root, d, scenes=["scene-0553"], stride=40, use_flow=False, verbose=False, lane=False)           # a sequence that only this options store holds
        opt = gt_api.key_of("bev_fusion", "v1.0-mini", "custom", d, parts=("front",), scenes=["scene-0553"], options={"use_flow": False})
        plain = gt_api.key_of("bev_fusion", "v1.0-mini", "custom", d, parts=("front",), scenes=["scene-0553"])
        tok = scene_tokens("scene-0553", "v1.0-mini", root)[0]
        assert opt.part("front").store != plain.part("front").store and opt.id != plain.id, "other generator options, another store and another id"
        assert opt.load(tok)["drivable_area"].shape == (120, 104) and os.path.isdir(opt.part("front").path), "the key loads from the store it names"
        try:
            plain.part("lane")
            raise AssertionError("a part that the key does not list must raise")
        except ValueError:
            pass
    # the completeness the key claims, on synthetic manifests: never a guess
    from .gt_api import _complete
    held = {"t1", "t2"}
    full = {"complete": True, "stride": 1, "scenes": ["scene-0061"], "sets_complete": ["small"]}
    assert _complete(full, held, None, "small", None) and _complete(full, held, None, "custom", ["scene-0061"]) and not _complete(full, held, None, "full", None), "a set by sets_complete, scenes by the index"
    assert not _complete({**full, "stride": 40}, held, None, "small", None) and not _complete({**full, "stride": 40}, held, None, "custom", ["scene-0061"]), "a stride store is never complete"
    assert not _complete({**full, "complete": False}, held, None, "custom", ["scene-0061"]) and not _complete(full, held, None, "custom", ["scene-0553"]) and not _complete({}, held, None, "small", None)
    assert not _complete(full, set(), None, "small", None) and not _complete(full, held, None, None, None), "no keyframes, or nothing asked: incomplete"
    assert _complete({}, {"t1", "t2", "t3"}, {"t1", "t2"}, "full", None) and not _complete({}, {"t1"}, {"t1", "t2"}, "full", None), "against the json tables: every keyframe, whatever the index says"
    # the versions tables: consistent, a cycle or a dangling same_gt_as is an error, not a hang
    versions.check_table()
    real = versions.table
    try:
        versions.table = lambda: [dict(version="v1", signature="a", same_gt_as="v2", tag="", date="", what_changed=""), dict(version="v2", signature="b", same_gt_as="v1", tag="", date="", what_changed="")]
        try:
            versions.canonical("a")
            raise AssertionError("a cycle must raise")
        except ValueError as e:
            assert "cycle" in str(e)
        versions.table = lambda: [dict(version="v1", signature="a", same_gt_as="v9", tag="", date="", what_changed="")]
        try:
            versions.canonical("a")
            raise AssertionError("a dangling same_gt_as must raise")
        except ValueError:
            pass
    finally:
        versions.table = real
    for rows, why in (([dict(version="v1", signature="a", same_gt_as=""), dict(version="v2", signature="a", same_gt_as="")], "one signature twice"),
                      ([dict(version="v1", signature="a", same_gt_as="v2"), dict(version="v2", signature="b", same_gt_as="")], "same_gt_as names a later row")):
        try:
            versions.check_table(rows, release_rows=[], lane_versions=set())
            raise AssertionError(f"check_table must refuse: {why}")
        except ValueError:
            pass
    try:
        versions.check_table([dict(version="v1", signature="a", same_gt_as="")], release_rows=[dict(release="v9", front="v7", lane="l1")], lane_versions={"l1"})
        raise AssertionError("a release naming a missing front version must be refused")
    except ValueError:
        pass
    # the documented module attributes are there after `import autogt`, and the rigs of gt_api are those of lane_gt
    import autogt
    from . import gt_api
    assert autogt.rig_masks.mask_for and set(autogt.lane_gt.RIGS) == set(gt_api.SYSTEMS), "autogt.lane_gt / autogt.rig_masks are there, the systems of gt_api are the rigs of lane_gt"
    try:
        autogt.nothing_here
        raise AssertionError("an unknown attribute must raise AttributeError")
    except AttributeError:
        pass


def _architecture_check():
    """the architecture drawing (architecture.json -> .svg / .md) is in step with the code: every box points at real code, no file is in no box, the generated files are current (the check of
    autoresearch/tools/archdraw.py; skipped when the tool is not there, e.g. in a copy of autogt alone)"""
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    tool = os.path.join(here, "..", "autoresearch", "tools", "archdraw.py")
    if not os.path.exists(tool):
        return False
    r = subprocess.run([sys.executable, tool, "check", os.path.join(here, "architecture.json")], capture_output=True, text=True)
    assert r.returncode == 0, "the architecture drawing is out of step with the code (run `python autoresearch/tools/archdraw.py render autogt/architecture.json`): " + r.stdout[-800:] + r.stderr[-300:]
    return True


def run(root=None, verbose=True):
    root = os.path.expanduser(root or "~/data/nuscenes")
    _mapping_checks()
    _error_checks(root)
    _store_checks(root)
    drawn = _architecture_check()
    msg = "cli selftest ok: every command is one method of AutoGT and one call (generate, generate-bev, generate-lane, audit-lane, area-masks, build, key, have, data, report, clean, selftest --only, pictures), every --help works, wrong arguments exit 2 with a message, not there exits 1, stores and tables" + ("; the architecture drawing is in step with the code" if drawn else "")
    if verbose:
        print(msg)
    return msg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    run(ap.parse_args().root)
