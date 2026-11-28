"""Self-test cases of the front store (store.py, pipeline.py, sets.py): the writer, the index.json of a store folder, the scene sets, resuming;
run by `python -m GtGenerator selftest` (selftest.py holds the table). A case without `root` runs on the synthetic fixture (fixture.py), a case
with `root` needs nuScenes mini.

    index          the index of a folder holds what the folder holds: a smaller or a repeated run neither shrinks the counts nor marks an
                   interrupted run complete; strides that differ are said so
    sets_manifest  the scene sets (small is a part of full), `have`, and the manifest of a store: gt_version / commit / dirty / sets_complete,
                   whether the folder is a release; a long job refuses a generator that changed meanwhile
    generate       `generate` writes and reloads a store keyed by the signature (the light devkit gives the GT of the full one), the readers of
                   the top-level API, the version found in the folder, a rerun writes nothing, wrong arguments are refused
    set_runs       a run of a named set: its index, a no-op rerun leaves it byte for byte, a clean tree stamps an earlier dirty run, another
                   option is another folder
    resume         a rerun after damage writes exactly the damage; two worker processes give the same GT as one
    raw_check      the raw-data check: complete on mini, every missing file of a data root without files is reported
    release        one release, one command: the front GT, the lane GT of both rigs and the masks of the same sequences out of one `generate`
"""
import json
import os
import tempfile

import numpy as np

from .fixture import ToyGenerator, sets_at, tiny_sets, tiny_tables


def _refused(call, why, errors=(ValueError,)):
    try:
        call()
    except errors:
        return
    raise AssertionError(f"must refuse: {why}")


# ---------------------------------------------------------------------------------------------------------------- the synthetic fixture
def index():
    """review of 2026-11-01"""
    from .store import write_index
    from .suite import current_signature
    with tempfile.TemporaryDirectory() as d:
        sd6 = os.path.join(d, current_signature())
        os.makedirs(sd6)
        for i in range(5):
            open(os.path.join(sd6, f"t{i}.npz"), "wb").write(b"x")
        open(os.path.join(sd6, "t9.npz.1.tmp.npz"), "wb").write(b"x")                  # a leftover temporary file is no keyframe
        open(os.path.join(sd6, "t8.npz"), "wb").close()                                # nor is an empty file
        gen6 = ToyGenerator()
        big = [f"scene-{i:04d}" for i in range(19)] + ["scene-0061"]                  # small is a part of big

        def idx(names, stride=1, complete=True):
            write_index(sd6, gen6, "v1.0-trainval" if len(names) > 5 else "v1.0-mini", names, stride, 99, {}, complete=complete)
            return json.load(open(os.path.join(sd6, "index.json")))
        m = idx(big, complete=False)                                                   # an interrupted big run ...
        assert m["complete"] is False and m["keyframes"] == 99, m
        m = idx(["scene-0061"])                                                        # ... then a finished small one
        assert m["complete"] is False and m["scenes"] == sorted(big) and m["version"] == "v1.0-trainval", m
        m = idx(big)                                                                   # the big run resumed and finished
        assert m["complete"] is True and m["keyframes"] == 5 and m["stride"] == 1, m
        m = idx(big, stride=2)                                                         # the same scenes again with another stride
        assert m["keyframes"] == 5 and m["stride"] == "mixed" and m["complete"] is True, m
        m = idx(["scene-0061"], stride=2)                                              # a subset after a full store
        assert m["scenes"] == sorted(big) and m["complete"] is True and m["stride"] == "mixed", m


def sets_manifest():
    """the sets of the repository, then a tiny nuScenes folder (fixture.tiny_tables) whose two sequences are the set `small`; `full` names a
    third sequence that it does not have, as mini does not have trainval"""
    from . import pipeline, sets, versions as V
    from .store import environment, write_index
    from .suite import GtGenerator, current_signature, current_version, scene_tokens
    assert sets.names() == ["full", "small"], sets.names()
    small, full_ = sets.scenes("small"), sets.scenes("full")
    assert len(small) == 10 and len(full_) == 850 == len(set(full_)) and set(small) <= set(full_), (len(small), len(full_))
    sets_text = GtGenerator.sets()["text"]
    assert "small" in sets_text and "full" in sets_text, sets_text
    cs = V.code_state()
    assert set(cs) == {"commit", "dirty"} and (cs["commit"] is None) == (cs["dirty"] is None), cs
    with tempfile.TemporaryDirectory() as d:
        root, data, sets_dir = (os.path.join(d, x) for x in ("nuscenes", "store", "sets"))
        tables = tiny_tables(root)
        tiny_sets(sets_dir, small=list(tables), full=list(tables) + ["scene-0999"])
        sd5 = os.path.join(data, current_signature())
        os.makedirs(sd5)
        with sets_at(sets_dir):
            small_tokens = sets.tokens("small", "v1.0-mini", root)
            assert small_tokens == [t for ts in tables.values() for t in ts], small_tokens
            _refused(lambda: sets.tokens("full", "v1.0-mini", root), "full is not in mini")
            _refused(lambda: sets.scenes("nope"), "unknown set")
            _refused(lambda: pipeline.generate(root, "x", set_name="small", scenes=["scene-0061"]), "set and scenes together")
            for t in small_tokens[1:]:
                open(os.path.join(sd5, t + ".npz"), "wb").write(b"x")
            st = sets.status("small", data=data, version="v1.0-mini", dataroot=root)
            assert st["have"] == len(small_tokens) - 1 and st["total"] == len(small_tokens) and not st["complete"], st
            assert sets.complete_in(sd5, "v1.0-mini", root) == []
            open(os.path.join(sd5, small_tokens[0] + ".npz"), "wb").write(b"x")
            assert sets.status("small", data=data, version="v1.0-mini", dataroot=root)["complete"]
            assert sets.complete_in(sd5, "v1.0-mini", root) == ["small"]                  # full needs trainval: skipped, not an error
            want_ = sets.scenes("small")[:2]                                               # `have`: the check before a training
            every = sets.stored_for(want_, data=data, version="v1.0-mini", dataroot=root)
            assert every["missing"] == [] and every["have"] == every["total"] > 0 and every["scenes"] == 2, every
            gone = scene_tokens(want_[0], "v1.0-mini", root)[0]
            os.remove(os.path.join(sd5, gone + ".npz"))
            gap = sets.stored_for(want_, data=data, version="v1.0-mini", dataroot=root)
            assert gap["missing"] == [gone] and gap["have"] == gap["total"] - 1, gap
            open(os.path.join(sd5, gone + ".npz"), "wb").write(b"x")
            assert sets.stored_for(want_, data=data, version="v1.0-mini", dataroot=root, stride=1000)["total"] == 2, "stride"
            gen5 = ToyGenerator()

            def manifest(run, complete=True):
                write_index(sd5, gen5, "v1.0-mini", ["scene-0061"], 1, 1, {}, complete=complete, run=run)
                return json.load(open(os.path.join(sd5, "index.json")))
            ok = {"set": "full", "commit": "abc1234", "dirty": False, "sets_complete": ["full"]}
            m = manifest(ok)
            assert m["release"] is True and m["gt_version"] == current_version() and m["commit"] == "abc1234" and m["sets_complete"] == ["full"], m
            assert manifest({**ok, "dirty": True})["release"] is False, "a dirty tree is no release"
            assert manifest({**ok, "dirty": None})["release"] is False, "an unknown tree state is no release"
            assert manifest(ok, complete=False)["release"] is False, "an unfinished run is no release"
            assert manifest({**ok, "sets_complete": ["small"]})["sets_complete"] == ["full", "small"], "sets that were complete stay listed"
            m = manifest({**ok, "sets_complete": ["small"]})
            assert m["release"] is True, m                                                # full came from the earlier run on this folder
            m = manifest(None)                                                             # a provisional write (no run manifest) keeps the manifest keys
            assert m["release"] is True and m["commit"] == "abc1234" and m["sets_complete"] == ["full", "small"], m
            assert manifest(None, complete=False)["release"] is False, "the provisional index of a running job is no release"
            # review of 2026-11-02: the sets a run could check are taken from ITS list (a deleted keyframe must remove `full`), a set that this
            # nuScenes version cannot check keeps its old entry
            assert sets.checkable("v1.0-mini", root) == ["small"], sets.checkable("v1.0-mini", root)
            m = manifest({**ok, "sets_complete": ["small"], "sets_checked": ["small"]})
            assert m["sets_complete"] == ["full", "small"] and m["release"] is True, "mini cannot check full: its entry stays"
            m = manifest({**ok, "sets_complete": [], "sets_checked": ["full", "small"]})
            assert m.get("sets_complete", []) == [] and m["release"] is False, "a hole in full takes the set out and the release with it"
            m = manifest({**ok, "sets_complete": ["full", "small"], "sets_checked": ["full", "small"], "environment": environment()})
            assert m["sets_complete"] == ["full", "small"] and m["release"] is True and set(m["environment"]) >= {"numpy", "shapely"}, m
            # a job that runs for hours refuses a generator that changed meanwhile (a pool worker imports the source tree afresh)
            pipeline._same_generator(gen5, gen5.signature())
            try:
                pipeline._same_generator(gen5, "deadbeef00")
                raise AssertionError("a changed generator must be refused")
            except RuntimeError as e:
                assert "changed while this job runs" in str(e)
            os.remove(os.path.join(sd5, "index.json"))
            assert manifest({"set": None, "commit": "abc1234", "dirty": False, "sets_complete": ["small"]})["release"] is False, "no full set"


# ---------------------------------------------------------------------------------------------------------------- nuScenes mini
def generate(root):
    from . import current_signature, current_version, load as load_stored, missing, pipeline, store as store_mod
    from .front_selftest import _same_gt, devkit
    from .grid import GridSpec
    from .store import load_keyframe, write_keyframe
    from .suite import code_signature
    from .versions import canonical
    from .view import plt, stored
    g = GridSpec()
    store_sig = current_signature()
    with tempfile.TemporaryDirectory() as d:
        n = pipeline.generate(root, d, scenes=["scene-0061"], stride=20, verbose=False)["written"]
        sig_dir = os.path.join(d, store_sig)
        files = sorted(os.listdir(sig_dir))
        assert n == len(files) - 1 and "index.json" in files, files
        tok2 = [f for f in files if f.endswith(".npz")][0][:-4]
        g2 = load_keyframe(sig_dir, tok2)
        assert g2["flow"].shape == (2, g.nx, g.ny) and g2["meta"]["sample"]
        full_dir = os.path.join(d, "full")                                              # the same keyframe from the full devkit, through the one writer
        os.makedirs(full_dir)
        assert write_keyframe(full_dir, tok2, lambda: devkit(root).keyframe_gt(tok2, g))
        assert _same_gt(load_keyframe(full_dir, tok2), g2), "light == full: every layer and every meta entry"

        def never():
            raise AssertionError("a stored keyframe is not made again")
        assert not write_keyframe(full_dir, tok2, never)
        # the readers of the top-level API
        assert store_sig == canonical(devkit(root).signature()), "the folder name without the devkit is the generator's"
        assert current_signature(boxes_at="annotation") == code_signature(boxes_at="annotation") != store_sig
        assert np.array_equal(load_stored(tok2, data=d)["flow"], g2["flow"])
        assert missing([tok2, "0" * 32], data=d) == ["0" * 32] and missing([tok2], data=d) == []    # the store covers a token that is there, not another
        fig = stored(tok2, None, data=d)                                               # a picture straight from the stored GT
        assert len(fig.axes) >= 5 and fig.axes[0].images, "stored-GT figure"
        plt.close(fig)
        try:
            load_stored("0" * 32, data=d)
            raise AssertionError("load of an unknown keyframe must fail")
        except FileNotFoundError as e:
            assert store_sig in str(e)
    assert store_mod.detect_version(root) == "v1.0-mini"
    with tempfile.TemporaryDirectory() as d:
        old_group, pipeline.GROUP_SCENES = pipeline.GROUP_SCENES, 1                      # two groups: the loop over devkit loads is exercised
        try:
            r = pipeline.generate(root, d, scenes=["scene-0061", "scene-0103"], stride=20, verbose=False)
            from . import suite as suite_mod
            real_gen, suite_mod.GtGenerator = suite_mod.GtGenerator, None                        # a rerun of finished groups must not load the devkit
            try:
                again = pipeline.generate(root, d, scenes=["scene-0061", "scene-0103"], stride=20, verbose=False)
            finally:
                suite_mod.GtGenerator = real_gen
        finally:
            pipeline.GROUP_SCENES = old_group
        assert r["signature"] == store_sig and r["out"] == os.path.join(d, store_sig), r
        assert r["version"] == "v1.0-mini" and r["scenes"] == 2 and r["keyframes"] == r["written"] == 4 and again["written"] == 0, (r, again)
        held = json.load(open(os.path.join(r["out"], "index.json")))
        assert held["scenes"] == ["scene-0061", "scene-0103"] and held["keyframes"] == 4 and held["complete"] is True, held
        assert held["gt_version"] == r["gt_version"] == current_version(), (held, r)                      # the version key is stamped
        _refused(lambda: pipeline.generate(root, d, verbose=False, scenes=["scene-9999"]), "unknown scene", (ValueError, OSError))
        _refused(lambda: pipeline.generate(root, d, verbose=False, version="v1.0-trainval"), "version", (ValueError, OSError))
        try:
            store_mod.detect_version(d)
            raise AssertionError("a folder that is not a nuScenes folder must be refused")
        except ValueError as e:
            assert "not a nuScenes folder" in str(e)
    return f"generate + reload of {n} keyframes ok"


def set_runs(root):
    from . import pipeline, sets, versions as versions_mod
    from .suite import GtGenerator
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as sets_dir:
        with open(os.path.join(sets_dir, "tiny.txt"), "w") as f:
            f.write("# two sequences\nscene-0061\nscene-0103  # trailing comment\n\n")
        with sets_at(sets_dir):                                                          # a tiny set instead of the real ones
            assert sets.names() == ["tiny"] and sets.scenes("tiny") == ["scene-0061", "scene-0103"]
            r = pipeline.generate(root, d, set_name="tiny", stride=20, verbose=False)
        held = json.load(open(os.path.join(r["out"], "index.json")))
        assert held["set"] == "tiny" and held["scenes"] == ["scene-0061", "scene-0103"] and r["keyframes"] == 4, (held, r)
        assert held["complete"] is True and held["release"] is False and r["release"] is False and "commit" in held and "dirty" in held, held
        # a rerun with nothing to do writes no index at all: it neither flags the store incomplete nor re-stamps its manifest
        calls = []
        real_write = pipeline.write_index
        pipeline.write_index = lambda *a, **k: (calls.append(k.get("complete", True)), real_write(*a, **k))[1]
        real_state = versions_mod.code_state
        # a tree that is not clean: nothing to upgrade (the test must not depend on the state of the checkout)
        versions_mod.code_state = lambda *a, **k: {"commit": "abc1234", "dirty": True}
        try:
            with sets_at(sets_dir):
                r2 = pipeline.generate(root, d, set_name="tiny", stride=20, verbose=False)
        finally:
            pipeline.write_index, versions_mod.code_state = real_write, real_state
        assert r2["written"] == 0 and calls == [] and json.load(open(os.path.join(r["out"], "index.json"))) == held, (
            calls, "a no-op rerun leaves index.json byte for byte as it was")
        # ... but a CLEAN tree turns the index of an earlier, dirty run into a stamped one (the signature is the same, so is the GT; a release also needs the
        # full set): commit and dirty are re-written, a release stays
        versions_mod.code_state = lambda *a, **k: {"commit": "feed123", "dirty": False}
        try:
            with sets_at(sets_dir):
                r3 = pipeline.generate(root, d, set_name="tiny", stride=20, verbose=False)
                up = json.load(open(os.path.join(r["out"], "index.json")))
                assert r3["written"] == 0 and up["commit"] == "feed123" and up["dirty"] is False, up
                assert up["scenes"] == held["scenes"] and up["complete"] is True, up
                up["release"] = True
                json.dump(up, open(os.path.join(r["out"], "index.json"), "w"))
                pipeline.generate(root, d, set_name="tiny", stride=20, verbose=False)
            assert json.load(open(os.path.join(r["out"], "index.json"))) == up, "a release is left alone by a clean tree"
        finally:
            versions_mod.code_state = real_state
        r_opt = pipeline.generate(root, d, scenes=["scene-0061"], stride=40, use_flow=False, verbose=False)      # the folder name without the devkit
        assert r_opt["signature"] == GtGenerator(root, "v1.0-mini", ss=4, light=True, scenes=["scene-0061"], use_flow=False).signature() != r["signature"]


def resume(root):
    from . import pipeline, sets
    from .front_selftest import _same_gt
    from .store import load_keyframe
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as d2:
        kw = dict(scenes=["scene-0061", "scene-0103"], stride=20, verbose=False)
        r = pipeline.generate(root, d, **kw)
        stored = sorted(f for f in os.listdir(r["out"]) if f.endswith(".npz"))
        os.remove(os.path.join(r["out"], stored[0]))                                   # a lost file,
        open(os.path.join(r["out"], stored[1]), "wb").close()                          # an empty one (full disk),
        open(os.path.join(r["out"], stored[2] + ".1.tmp.npz"), "wb").write(b"half")    # a leftover of a killed writer next to a good file
        again = pipeline.generate(root, d, **kw)
        assert again["written"] == 2 and len(sets.held(r["out"])) == 4, again
        assert load_keyframe(r["out"], stored[1][:-4])["drivable_area"].shape == (120, 104)
        two = pipeline.generate(root, d2, workers=2, **kw)                              # the spawn pool: each worker builds its own generator
        assert two["written"] == 4 and two["signature"] == r["signature"], two
        for t in (stored[0][:-4], stored[3][:-4]):
            a_, b_ = load_keyframe(r["out"], t), load_keyframe(two["out"], t)
            assert _same_gt(a_, b_), "workers=2 == workers=1: every layer and every meta entry"
        for bad, why in ((dict(scenes=[]), "no scene is not all scenes"), (dict(scenes=["scene-0061"], stride=0), "stride 0"),
                         (dict(scenes=["scene-0061"], res=0.25), "another resolution in a used folder")):
            _refused(lambda: pipeline.generate(root, d, **{**dict(stride=20, verbose=False), **bad}), why)
        dup = pipeline.generate(root, d, scenes=["scene-0061", "scene-0061"], stride=20, verbose=False)
        assert dup["scenes"] == 1 and dup["keyframes"] == 2, dup                       # a name twice is one sequence


def raw_check(root):
    from . import sets
    ok_ = sets.check_raw(root, "v1.0-mini", scenes=["scene-0061"])
    assert ok_["missing"] == 0 and not ok_["maps"] and ok_["camera"] > 100 and ok_["radar"] > 500, ok_
    # a named set: the parameter `scenes` hid the function (a TypeError before 2026-11-02)
    with tempfile.TemporaryDirectory() as sd:
        with open(os.path.join(sd, "one.txt"), "w") as f:
            f.write("scene-0061\n")
        with sets_at(sd):
            assert sets.check_raw(root, "v1.0-mini", set_name="one") == ok_
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "v1.0-mini"))
        for t in ("scene", "sample", "log", "sample_data"):
            os.symlink(os.path.join(root, "v1.0-mini", t + ".json"), os.path.join(d, "v1.0-mini", t + ".json"))
        bad_ = sets.check_raw(d, scenes=["scene-0061"])                                 # tables only: no image, no radar file, no map
        assert bad_["missing"] == ok_["camera"] + ok_["radar"] and len(bad_["examples"]) == 10 and bad_["maps"], bad_
        assert "missing" in sets.report_raw(bad_) and "all on disk" in sets.report_raw(ok_)


def release(root):
    from . import lane_gt, pipeline, versions as V
    from .suite import code_signature, scene_tokens
    front, lane = V.version_of(code_signature()), lane_gt.version_of(lane_gt.code_signature())
    assert V.current_release() is not None, (f"this source (front {front}, lane {lane}) has no row in GtGenerator/releases.csv: add one (the releases are one "
                                             "running number: the next change of the front or the lane GT is the next number)")
    with tempfile.TemporaryDirectory() as d:
        toks_ = scene_tokens("scene-0061", "v1.0-mini", root)
        both = pipeline.generate(root, d, scenes=["scene-0061"], verbose=False)
        assert both["written"] == len(toks_) and both["lane"] and both["lane"]["written"] == len(toks_), both
        masks_ = os.path.join(d, "lane", both["lane_rigs"]["bev_fusion"]["signature"], "masks", "intersection_b0.5.npz")
        assert both["masks_error"] is None and os.path.exists(masks_), "generate builds the area masks of the BEV Fusion store too"
        assert os.path.isdir(os.path.join(d, "lane", both["lane"]["signature"])) and lane_gt.missing(toks_, data=os.path.join(d, "lane")) == []
        assert set(both["lane_rigs"]) == set(lane_gt.RIGS), "one store per rig of the lane GT"
        assert all(r["written"] == len(toks_) for r in both["lane_rigs"].values()), both["lane_rigs"]
        assert both["lane_rigs"]["bev_fusion"]["signature"] != both["lane"]["signature"]
        assert lane_gt.missing(toks_, data=os.path.join(d, "lane"), rig="bev_fusion") == []
        assert json.load(open(os.path.join(both["out"], "index.json")))["suite_release"] == V.current_release()
        only = pipeline.generate(root, d, scenes=["scene-0061"], verbose=False, lane=False)
        assert only["lane"] is None and only["written"] == 0, only
        strided = pipeline.generate(root, d, scenes=["scene-0061"], stride=20, verbose=False)
        assert strided["lane"] is None, "the lane GT is made for every keyframe: nothing at stride 20"
