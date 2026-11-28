"""Self-test cases of the GT versions (versions.py, D29) and of what pins them; run by `python -m GtGenerator selftest` (selftest.py holds the
table). `tables` needs no data, `pins` and `agents_doc` need nuScenes mini.

    tables      every module is hashed or listed as not shaping the GT, the table is complete and in order, THIS source has a row (an edit of a
                signature file without a new row fails here), the older rows have their git tags, a process hashes the files once, documents
                do not make a tree dirty
    pins        the digests behind every `same_gt_as` (front_digests.json): the generator of today makes the pinned frames, the compare finds a
                changed one
    agents_doc  AGENTS.md names every layer, meta entry and box field of a real keyframe (so it cannot drift from the code)
"""
import ast
import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def tables():
    from . import current_version, suite, versions as V
    from .nuscenes_gt import SIGNATURE_FILES
    todo_, reach_ = ["nuscenes_gt"], set()                        # every module of the package is part of the signature or listed as not shaping the GT
    while todo_:
        mod_ = todo_.pop()
        if mod_ in reach_:
            continue
        reach_.add(mod_)
        for n_ in ast.walk(ast.parse(open(os.path.join(HERE, mod_ + ".py"), encoding="utf-8").read())):
            if isinstance(n_, ast.ImportFrom) and n_.level == 1:
                todo_ += [n_.module] if n_.module else [a_.name for a_ in n_.names]
    assert {m_ + ".py" for m_ in reach_} == set(SIGNATURE_FILES), (
        f"the modules nuscenes_gt.py imports are {sorted(reach_)} but SIGNATURE_FILES is {SIGNATURE_FILES}: a module that shapes the GT must be hashed")
    loose_ = sorted(set(f_ for f_ in os.listdir(HERE) if f_.endswith(".py")) - set(SIGNATURE_FILES) - set(V.NOT_GENERATOR))
    assert not loose_, (
        f"{loose_} is neither in SIGNATURE_FILES (nuscenes_gt.py) nor in versions.NOT_GENERATOR: decide whether it shapes the GT "
        "(then it is hashed and a change needs a new version row) or not (then list it)")
    rows_ = V.table()
    # the front versions and the releases are ONE running sequence: the front rows have gaps (v9, v13)
    nums_ = [int(r["version"][1:]) for r in rows_]
    assert nums_ == sorted(set(nums_)) and nums_[0] == 1 and len({r["signature"] for r in rows_}) == len(rows_), nums_
    code = suite.code_signature()
    assert V.version_of(code) is not None, (f"the generator signature {code} has no row in GtGenerator/versions.csv: `python -m GtGenerator versions --update` "
                                            "derives one from git, "
                                            "commit, tag autogt/v<N> (decision D29)")
    store_sig = V.canonical(code)                                 # the folder of the stored GT: the signature itself, or the older one whose GT this code makes
    assert current_version() == V.version_of(store_sig) and current_version(boxes_at="annotation").startswith(current_version() + "+")
    assert store_sig == V.canonical(store_sig) and V.canonical("deadbeef00") == "deadbeef00", "same_gt_as ends at a signature that is its own store"
    assert all(V.row_of(r["same_gt_as"]) for r in rows_ if r.get("same_gt_as")), "same_gt_as names a version of the table"
    assert V.version_of_key("0.5_4_" + code) == V.version_of(store_sig) == V.version_of_key("0.5_x0_40_y-20_20_4_" + code)
    assert V.version_of_key("0.5_4_deadbeef00") is None and V.version_of_key(code) == V.version_of(store_sig)
    repo_ = os.path.dirname(HERE)                                 # the older rows are final, so their git tag exists (the last row is tagged after its commit)
    if os.path.exists(os.path.join(repo_, ".git")):
        try:
            tags_ = set(subprocess.run(["git", "-C", repo_, "tag", "-l", "autogt/*"], capture_output=True, text=True, timeout=30).stdout.split())
        except (OSError, subprocess.SubprocessError):
            tags_ = set()
        lost_ = [r["tag"] for r in rows_[:-1] if r["tag"] not in tags_]
        assert not tags_ or not lost_, f"versions.csv names git tags that do not exist: {lost_} (git tag -a autogt/vN <commit>)"
        if tags_ and rows_[-1]["tag"] not in tags_:
            print(f"note: the newest version {rows_[-1]['version']} has no git tag {rows_[-1]['tag']} yet: commit, then tag it")
    suite._SIGNATURES.clear()                                     # the key of a process is computed once: a second call does not hash the files again
    first_ = suite.current_signature()
    real_sig = suite.NuScenesGT.signature

    def _no_hashing(self):
        raise AssertionError("current_signature hashed the source files a second time")
    suite.NuScenesGT.signature = _no_hashing
    try:
        assert first_ == store_sig and suite.current_signature() == first_
    finally:
        suite.NuScenesGT.signature = real_sig
    assert all(V._changes_the_code(x) for x in (" M GtGenerator/cli.py", "?? GtGenerator/new_module.py", "R  GtGenerator/a.py -> GtGenerator/b.py"))
    assert not any(V._changes_the_code(x) for x in (" M GtGenerator/README.md", "?? GtGenerator/architecture.svg", "?? GtGenerator/architecture.json",
                                                    " M GtGenerator/REVIEW_2026-11-10.md")), "documents do not make a tree dirty"


def pins(root):
    """a row that says "this code makes the GT of that store" needs its signature in `same_as` of the pin of that store, and the generator of
    today must still make exactly the pinned frames (a recomputed subset: every 40th keyframe of mini and the sweeps of scene-0061)"""
    from . import checks, versions as V
    from .suite import GtGenerator, code_signature
    dg1 = checks.digests(root, "v1.0-mini", stride=40, sweeps=("scene-0061",))
    assert len(dg1["digests"]) >= 11 and dg1["signature"] == code_signature(), dg1["signature"]
    pinned = json.load(open(os.path.join(HERE, "front_digests.json")))
    for row in V.table():
        if row.get("same_gt_as"):
            store_ = V.canonical(row["signature"])
            assert row["signature"] in pinned.get(store_, {}).get("same_as", []), (
                f"{row['version']} claims the GT of {row['same_gt_as']} (store {store_}) but no pinned proof names {row['signature']}: run "
                "`python -m GtGenerator digest` in both source trees, compare, and add it to front_digests.json")
    pin = pinned.get(V.canonical(code_signature()))
    assert pin, (f"the generator of today ({code_signature()}) has no pinned digests: `python -m GtGenerator digest OUT.json` (stride 2) and add them "
                 "to front_digests.json")
    wrong = [k for k, v in dg1["digests"].items() if pin["digests"].get(k) != v]
    assert not wrong, (f"the generator makes other frames than the pinned ones ({len(wrong)} of {len(dg1['digests'])}, e.g. {wrong[:2]}): a changed GT needs a "
                       "new version row, not a changed pin")
    with tempfile.TemporaryDirectory() as d:
        pa, pb = os.path.join(d, "a.json"), os.path.join(d, "b.json")
        json.dump(dg1, open(pa, "w"))
        json.dump(dg1, open(pb, "w"))
        assert GtGenerator.digest(compare=[pa, pb])["ok"], "identical digests"
        json.dump({**dg1, "digests": {**dg1["digests"], next(iter(dg1["digests"])): "0" * 40}}, open(pb, "w"))
        r_ = GtGenerator.digest(compare=[pa, pb])
        assert not r_["ok"] and len(r_["differ"]) == 1, "a changed frame must be found"


def agents_doc(root):
    from .front_selftest import keyframe_61
    _, k = keyframe_61(root)
    guide = open(os.path.join(HERE, "AGENTS.md"), encoding="utf-8").read()
    keys = [x for x in k if x != "meta"] + list(k["meta"]) + list(k["meta"]["boxes"][0]) + ["dt", "steps"]
    missing = [x for x in keys if f"`{x}`" not in guide]
    assert not missing, f"GtGenerator/AGENTS.md does not mention (in backticks): {missing}"
