"""Spec tests of `jtest_eval.py` (E343, the one TEST use of E324 s): the rules that decide what is written, read and reported, on toy data (CPU, no model; no TEST data is read: the gate tests call the entries with a refused gate and every function
behind it replaced by one that fails the test).
The TEST gate on every entry, the write helper (E321's artefacts refused, atomic, written once), the sha256 guard of E321's artefacts (a change and a missing file are detected), dump validation and reuse under the provenance (run, code, device, checkpoint),
the reading rule at its boundaries, the D79 guards against E312, the glue of `score_dumps` (which read-out gets which graph, which pair dump, which ends), the checks of `precheck` / `score_seed` (kill criterion on TEST and VAL tolerance, reproduction of
E312's counts), the cards and results.md of the arms (per-arm D, guards against E312, intervals, asymmetric seeds), `cmd_all` end to end (rehearsal branch, code drift, no overwrite), the stored-card comparison (per-frame counts, `--no-stack`), the code hash.

    wsl bash tools/env/_py14.sh test_jtest_eval.py
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
import jtest_eval as J  # noqa: E402


@contextlib.contextmanager
def patched(*pairs):
    """(object, name, value) triples set for the block and put back after it"""
    old = [(o, n, getattr(o, n)) for o, n, _ in pairs]
    for o, n, v in pairs:
        setattr(o, n, v)
    try:
        yield
    finally:
        for o, n, v in old:
            setattr(o, n, v)


def raises(exc, f, *a, **k):
    try:
        f(*a, **k)
    except exc:
        return True
    return False


def boom(*a, **k):
    raise AssertionError("this function must not be reached")


# ------------------------------------------------------------------------------------------------------------------------------ the TEST gate on every entry
def test_the_test_gate_guards_every_entry():
    """no test id, no launcher: `cmd_all` and the dump commands of `main` refuse TEST before any work. Frames, dumps, hashes, scoring are replaced by functions that fail the test with an AssertionError (a SystemExit is the only pass)"""
    a = argparse.Namespace(split="test", max=0, run_prefix="e324_s_s", seeds=[0, 1], boot=20, no_stack=False, device="cpu", b_dump=None, control_ends_dump=None, control_card=None, test_id="", expect_code_digest=None, expect_code_card=None)
    old_env = os.environ.pop("AR_RUN_ID", None)
    try:
        with patched((J, "frames_of", boom), (J, "dump_b", boom), (J, "dump_ends", boom), (J, "guard_e321", boom), (J, "score_seed", boom), (J, "precheck", boom), (J, "code_sha256", boom)):
            assert raises(SystemExit, J.cmd_all, a), "cmd_all without a test id"
            a.test_id = "E343.t"
            assert raises(SystemExit, J.cmd_all, a), "cmd_all with an id but outside the launcher"
            os.environ["AR_RUN_ID"] = "E343.other"
            assert raises(SystemExit, J.cmd_all, a), "cmd_all under the launcher of another id"
            os.environ.pop("AR_RUN_ID")
            for argv in (["dump-b", "--split", "test", "--seed", "0"], ["dump-ends", "--split", "test", "--seed", "0"], ["dump-b", "--split", "test", "--seed", "0", "--b-run", "runs/e292_c2_s0"],
                         ["all", "--split", "test", "--run-prefix", "e324_s_s"]):
                with patched((sys, "argv", ["jtest_eval.py"] + argv)):
                    assert raises(SystemExit, J.main), argv
    finally:
        os.environ.pop("AR_RUN_ID", None)
        if old_env is not None:
            os.environ["AR_RUN_ID"] = old_env


# ------------------------------------------------------------------------------------------------------------------------------ the write helper and E321's artefacts
def test_write_helper_refuses_e321_and_is_atomic():
    """the guard runs on a COPY of the layout (ROOT = a temporary folder with the two protected folders in it): a broken guard writes into the temporary folder, never into the real artefacts of E321"""
    assert J.PROTECTED == ("temp/e321", "runs/eval/e312_test"), "the protected folders are E321's dumps and cards"
    with tempfile.TemporaryDirectory() as t:
        d = os.path.join(t, "Mixed_Case_Root")                                                        # the real root has capitals too (Auto-research): both sides of the comparison are case-folded
        os.makedirs(d)
        _write_helper_checks(d)


def _write_helper_checks(d):
    with patched((J, "ROOT", d)):
        for rel in J.PROTECTED + ("temp/e343",):
            os.makedirs(os.path.join(d, rel, "test"))
        for rel in ("temp/e321/test/never_written.pkl", "runs/eval/e312_test/never_written.json", "temp/e343/../e321/test/never_written.pkl", "temp/e321/val_check/never_written.json", "temp/e321/new_folder/never_written.txt",
                    "TEMP/E321/test/never_written.pkl", "Temp/e321/never_written.json", "RUNS/EVAL/E312_TEST/never_written.json", "runs/eval/e312_TEST/never_written.txt"):                         # /mnt/c is case-insensitive
            path = os.path.join(d, rel)
            for write in (lambda p: J.save_pickle(p, [1]), lambda p: J.save_json(p, {"a": 1}), lambda p: J.save_text(p, "x")):
                assert raises(PermissionError, write, path), rel
        assert [f for _, _, fs in os.walk(d) for f in fs] == [] and sorted(os.listdir(os.path.join(d, "temp"))) == ["e321", "e343"], "a refused write creates nothing: no file, no temporary file, no folder"
        link = os.path.join(d, "link")
        try:
            os.symlink(os.path.join(d, "temp", "e321"), link)
        except OSError:
            link = None                                                                                # no symlinks on this system
        if link:
            assert J.is_protected(os.path.join(link, "test", "x.pkl")) and raises(PermissionError, J.save_text, os.path.join(link, "x.txt"), "x"), "a symlink into E321's folder is refused by its real path"
        assert not J.is_protected(os.path.join(d, "temp", "e343", "test", "x.pkl")) and not J.is_protected(os.path.join(d, "runs", "eval", "e324s_test", "card_s0.json")) and not J.is_protected(os.path.join(d, "temp", "e3210", "x"))
        assert not J.is_protected(os.path.join(d, "TEMP", "E3210", "x")) and not J.is_protected(os.path.join(d, "runs", "eval", "E312_TEST_x", "x")), "a neighbour with a longer name is not E321's folder"
        p = os.path.join(d, "temp", "e343", "test", "dump.pkl")
        J.save_pickle(p, [1, 2])
        assert pickle.load(open(p, "rb")) == [1, 2]
        try:
            with J.atomic(p) as tmp:                                                                   # a crash in the middle: the old file stays whole, no temporary file is left
                open(tmp, "wb").write(b"trunc")
                raise RuntimeError("crash")
        except RuntimeError:
            pass
        assert pickle.load(open(p, "rb")) == [1, 2] and os.listdir(os.path.dirname(p)) == ["dump.pkl"]
        assert raises(FileExistsError, J.save_json, p, {"a": 1}, overwrite=False) and raises(FileExistsError, J.save_text, p, "x", overwrite=False) and pickle.load(open(p, "rb")) == [1, 2], "a result written once is not overwritten"
        J.save_json(os.path.join(d, "temp", "e343", "test", "new.json"), {"a": 1}, overwrite=False)
        # the allow-list: only temp/e343 and runs/eval/e324s_val, e324s_test, e312_val are written (under the root); an NTFS short name of E321's folders, another folder or a place outside the root is refused, the deny-list or not
        for rel in ("runs/eval/E312_T~1/never.json", "temp/E321~1/test/never.pkl", "runs/eval/e312_test2/never.json", "runs/eval/other/never.json", "temp/other/never.txt", "runs/eval/e324s_train/never.json", "temp/e3430/never.txt", "runs/eval/e312_val2/never.json", "never.txt"):
            assert raises(PermissionError, J.save_text, os.path.join(d, rel), "x"), rel
        assert raises(PermissionError, J.save_text, os.path.join(os.path.dirname(d), "outside_the_root.txt"), "x"), "a path outside the root"
        assert not os.path.exists(os.path.join(d, "runs", "eval", "E312_T~1")) and not os.path.exists(os.path.join(d, "runs", "eval", "other")) and not os.path.exists(os.path.join(os.path.dirname(d), "outside_the_root.txt"))
        for rel in ("runs/eval/e324s_test/card_s0.json", "runs/eval/e324s_val/results.md", "runs/eval/e312_val/card_s0.json", "temp/e343/val_max6/cards_e312/card_s0.json"):
            J.save_text(os.path.join(d, rel), "x")
            assert os.path.exists(os.path.join(d, rel)), rel
        with patched((J, "EXTRA_WRITABLE", (os.path.join(os.path.dirname(d), "extra"),))):
            J.save_text(os.path.join(os.path.dirname(d), "extra", "ok.txt"), "x")                      # the tests' own temporary folders
            assert raises(PermissionError, J.save_text, os.path.join(os.path.dirname(d), "extra2", "no.txt"), "x")


def test_e321_artefacts_unchanged():
    """the sha256 of E321's dumps and cards equal the card's: all 7 files exist (the four pkl files are re-computed here); the dress rehearsal and the TEST run check the same before and after"""
    missing = [r for r in J.E321_SHA256 if not os.path.exists(os.path.join(J.ROOT, r))]
    assert len(J.E321_SHA256) == 7 and not missing, f"E321's artefacts are missing: {missing}"
    assert J.e321_changed() == [], J.e321_changed()
    print(f"  e321 artefacts: {len(J.E321_SHA256)} of {len(J.E321_SHA256)} present, all equal to the card's sha256")


def test_e321_guard_detects_a_change_and_a_missing_file():
    files = {"temp/e321/test/a.pkl": b"one", "runs/eval/e312_test/card_s0.json": b"two"}
    want = {rel: hashlib.sha256(c).hexdigest() for rel, c in files.items()}
    with tempfile.TemporaryDirectory() as d, patched((J, "ROOT", d), (J, "E321_SHA256", want)):
        for rel, c in files.items():
            os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
            open(os.path.join(d, rel), "wb").write(c)
        assert J.e321_changed() == [] and J.guard_e321("test") is None
        open(os.path.join(d, "temp/e321/test/a.pkl"), "wb").write(b"1ne")
        assert J.e321_changed() == ["temp/e321/test/a.pkl"] and raises(SystemExit, J.guard_e321, "after the run"), "a modified artefact stops the run"
        open(os.path.join(d, "temp/e321/test/a.pkl"), "wb").write(b"one")
        os.remove(os.path.join(d, "runs/eval/e312_test/card_s0.json"))
        assert J.e321_changed() == ["runs/eval/e312_test/card_s0.json (missing)"] and raises(SystemExit, J.guard_e321, "before the run"), "a deleted artefact is a change"


# ------------------------------------------------------------------------------------------------------------------------------ dumps: validation, reuse, provenance
def toy_dump(n=None, split="val"):
    rows = J.split_rows(split).tolist()
    return [dict(row=r) for r in rows[: n or None]]


def test_dump_validation_and_reuse():
    full = toy_dump()
    assert J.validate_dump(full, "val", "toy") == full
    assert raises(RuntimeError, J.validate_dump, full[:-1], "val", "truncated"), "a truncated dump is refused"
    assert raises(RuntimeError, J.validate_dump, full + [dict(row=-1)], "val", "too long")
    assert raises(RuntimeError, J.validate_dump, full[::-1], "val", "reversed")
    assert raises(RuntimeError, J.validate_dump, [dict(row=r) for r in J.split_rows("test").tolist()[:873]], "val", "the rows of the other split")
    assert raises(RuntimeError, J.validate_dump, full[:5] + full[6:] + [full[5]], "val", "one row moved")
    assert len(J.validate_dump(full, "val", "toy", n=8)) == 8 and raises(RuntimeError, J.validate_dump, full[:7], "val", "toy", 8), "a smoke run takes the first n frames of a longer dump"
    with tempfile.TemporaryDirectory() as d:
        p, calls = os.path.join(d, "b.pkl"), []
        pickle.dump(full, open(p, "wb"))
        got = J.get_dump(p, lambda: calls.append(1), "val", "B")
        assert len(got) == 873 and not calls, "an existing dump is reused, not made again"
        assert raises(SystemExit, J.get_dump, os.path.join(d, "missing.pkl"), None, "val", "B"), "a dump named by a flag must exist"
        q = os.path.join(d, "made.pkl")
        assert len(J.get_dump(q, lambda: (calls.append(1), pickle.dump(full, open(q, "wb"))), "val", "B")) == 873 and calls == [1], "a missing default dump is made once"
        t = os.path.join(d, "trunc.pkl")
        pickle.dump(full[:800], open(t, "wb"))
        assert raises(RuntimeError, J.get_dump, t, lambda: calls.append(2), "val", "B") and calls == [1], "a truncated dump on disk is refused, not repaired"


def test_a_dump_is_reused_only_under_the_same_provenance():
    full = toy_dump()
    prov = dict(run="r", device="cpu", code_digest="c1", checkpoint_sha256="k1")
    with tempfile.TemporaryDirectory() as d, patched((J, "EXTRA_WRITABLE", (d,))):
        p, calls = os.path.join(d, "ends_r.pkl"), []
        pickle.dump(full, open(p, "wb"))
        J.write_head(p, dict(frames=873), prov)
        assert os.path.exists(os.path.join(d, "ends_r_head.json")) and len(J.get_dump(p, lambda: calls.append(1), "val", "x", expect=prov)) == 873 and not calls
        for bad, why in ((dict(prov, code_digest="c2"), "code"), (dict(prov, device="cuda"), "device"), (dict(prov, checkpoint_sha256="k2"), "checkpoint"), (dict(prov, run="r2"), "run")):
            assert raises(SystemExit, J.get_dump, p, lambda: calls.append(1), "val", "x", expect=bad) and not calls, f"another {why}: refused, nothing made, nothing deleted"
        assert os.path.exists(p)
        q = os.path.join(d, "nohead.pkl")
        pickle.dump(full, open(q, "wb"))
        assert raises(SystemExit, J.get_dump, q, lambda: calls.append(1), "val", "x", expect=prov), "a dump without provenance is refused when it was to be made by this module"
        assert len(J.get_dump(q, None, "val", "x", expect=prov)) == 873, "a dump named by a flag without a sidecar (E321's, the stored VAL dumps) is exempt"
        assert len(J.get_dump(p, None, "val", "x", expect=prov)) == 873 and raises(SystemExit, J.get_dump, p, None, "val", "x", expect=dict(prov, device="cuda")), "a dump named by a flag that THIS module made has a sidecar and is checked like the others"
        full_prov = dict(prov, code_sha256={"a.py": "1", "b.py": "1"})
        J.write_head(p, dict(frames=873), full_prov)
        try:
            J.get_dump(p, None, "val", "x", expect=dict(prov, code_digest="c2", code_sha256={"a.py": "2", "b.py": "1"}))
            raise AssertionError("a changed code file must refuse the reuse")
        except SystemExit as e:
            assert "changed code files: a.py" in str(e) and "b.py" not in str(e), str(e)
        assert raises(RuntimeError, J.head_path, os.path.join(d, "x.out"))
        # the provenance itself: the checkpoint and the code files
        os.makedirs(os.path.join(d, "runs", "r"))
        open(os.path.join(d, "runs", "r", "model.pt"), "wb").write(b"weights")
        with patched((J, "ROOT", d), (J, "code_sha256", lambda entry=None: {"a.py": "1"})):
            p1 = J.provenance("r", "cpu")
            assert p1 == dict(run="r", device="cpu", code_digest=J.code_digest({"a.py": "1"}), checkpoint_sha256=hashlib.sha256(b"weights").hexdigest(), code_sha256={"a.py": "1"}), "the head file names the code files too"
            open(os.path.join(d, "runs", "r", "model.pt"), "wb").write(b"weights2")
            assert J.provenance("r", "cpu")["checkpoint_sha256"] != p1["checkpoint_sha256"] and J.provenance("r", "cuda")["device"] == "cuda"
        assert J.code_digest({"a.py": "1"}) != J.code_digest({"a.py": "2"}) and J.code_digest({"a.py": "1"}) != J.code_digest({"b.py": "1"})


def test_standalone_dumps_never_overwrite_and_need_a_pkl_name():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.pkl")
        open(p, "wb").write(b"x")
        with patched((J, "frames_of", boom), (J, "EXTRA_WRITABLE", (d,))):                             # a refusal comes before any work: reaching the frames fails the test (and never starts a dump)
            assert raises(SystemExit, J.dump_b, "val", "e273_p_s0", p) and raises(SystemExit, J.dump_ends, "val", "e324_s_s0", p) and open(p, "rb").read() == b"x", "an existing dump is not overwritten"
            assert raises(RuntimeError, J.dump_b, "val", "e273_p_s0", os.path.join(d, "x.out")) and raises(RuntimeError, J.dump_ends, "val", "e324_s_s0", os.path.join(d, "y")), "no .pkl: the head file would take the name"
        with patched((J, "frames_of", boom)):
            assert raises(PermissionError, J.dump_b, "val", "e273_p_s0", os.path.join(d, "z.pkl")) and raises(PermissionError, J.dump_ends, "val", "e324_s_s0", os.path.join(d, "z.pkl")), "a place outside the allow-list: refused at the START of the dump, not after it"
        assert sorted(os.listdir(d)) == ["x.pkl"]


def test_code_sha256_follows_the_imports():
    """the closure is looked up in every folder of the project's search path (what bflc_env.sh puts on PYTHONPATH: subfolders such as evaluation/sparse_lanes, training/models), also through imports inside functions of the lane's code (not of the shared library autoresearch/, whose
    lazy imports run only when a function is called); a name in two folders counts for both; the scratch folders, the gate's `evaluate`, the GT package and the library are not followed; a file that changes changes the digest"""
    files = {"proj/evaluation/entry.py": "import helper\nimport qgraph_eval as QE\nimport dup\nimport scratch_mod\nfrom Evaluation import stats as ST\nimport numpy\n\n\ndef f():\n    import lazy\n    import evaluate\n    from GtGenerator import x\n",
             "proj/evaluation/qgraph_eval.py": "import topo_metrics as TM\n\n\ndef g():\n    import inner_lazy\n", "proj/evaluation/sparse_lanes/topo_metrics.py": "", "proj/training/models/inner_lazy.py": "",
             "proj/training/grid/helper.py": "from . import sub\nimport pkg.mod\n", "proj/training/grid/sub.py": "", "proj/training/grid/lazy.py": "def (:\n", "proj/training/grid/pkg/__init__.py": "", "proj/training/grid/pkg/mod.py": "",
             "proj/training/grid/dup.py": "", "proj/evaluation/lane_task/dup.py": "", "proj/temp/scratch_mod.py": "", "proj/training/grid/_session_scratch_/scratch_mod.py": "",
             "proj/evaluation/evaluate.py": "import never\n", "proj/training/grid/never.py": "", "autoresearch/Evaluation/__init__.py": "def __getattr__(name):\n    from . import lazy_api\n", "autoresearch/Evaluation/lazy_api.py": "", "autoresearch/Evaluation/stats.py": "", "GtGenerator/x.py": "", "proj/training/grid/other.py": ""}
    want = ["proj/evaluation/entry.py", "proj/evaluation/qgraph_eval.py", "proj/evaluation/sparse_lanes/topo_metrics.py", "proj/training/models/inner_lazy.py", "proj/training/grid/helper.py", "proj/training/grid/sub.py", "proj/training/grid/lazy.py",
            "proj/training/grid/pkg/__init__.py", "proj/training/grid/pkg/mod.py", "proj/training/grid/dup.py", "proj/evaluation/lane_task/dup.py", "autoresearch/Evaluation/__init__.py", "autoresearch/Evaluation/stats.py"]
    with tempfile.TemporaryDirectory() as t:
        for rel, c in files.items():
            os.makedirs(os.path.dirname(os.path.join(t, rel)), exist_ok=True)
            open(os.path.join(t, rel), "w").write(c)
        entry = os.path.join(t, "proj", "evaluation", "entry.py")
        with patched((J, "ROOT", os.path.join(t, "proj"))):
            got = J.code_sha256(entry=entry)
            assert sorted(got) == sorted(want), sorted(set(got) ^ set(want))      # subfolders and lazy imports followed, both `dup` found, scratch / evaluate / GtGenerator / library not, a broken file hashed
            open(os.path.join(t, "proj/training/grid/sub.py"), "w").write("x = 1\n")
            again = J.code_sha256(entry=entry)
            assert [k for k in got if got[k] != again[k]] == ["proj/training/grid/sub.py"] and J.code_digest(got) != J.code_digest(again)
            open(os.path.join(t, "proj/evaluation/sparse_lanes/topo_metrics.py"), "w").write("x = 2\n")      # the scorer's own helper module, imported by `import topo_metrics as TM` in qgraph_eval: the digest follows it
            third = J.code_sha256(entry=entry)
            assert third["proj/evaluation/sparse_lanes/topo_metrics.py"] != again["proj/evaluation/sparse_lanes/topo_metrics.py"] and J.code_digest(third) != J.code_digest(again)


def test_the_code_digest_covers_every_module_a_run_loads():
    """a subprocess imports what this module imports (the lazy imports of a VAL run included, CUDA hidden) and lists the repository modules it loaded: every one is in the digest of the real tree, and the digest names the helper modules of the scorer
    (evaluation/sparse_lanes/topo_metrics.py ...) that a fixed folder list would have missed; library-only helpers of the suite that nothing imports are not in it"""
    script = (
        "import ast, importlib, json, os, sys\n"
        "import jtest_eval as J\n"
        "tree = ast.parse(open(J.__file__).read())\n"
        "names = sorted({a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and not n.level})\n"
        "for name in names:\n"
        "    if name.split('.')[0] not in J.CODE_SKIP:\n"
        "        importlib.import_module(name)\n"
        "repo = os.path.realpath(os.path.join(J.ROOT, '..'))\n"
        "rel = lambda p: os.path.relpath(os.path.realpath(p), repo).replace(os.sep, '/')\n"
        "print(json.dumps(sorted({rel(m.__file__) for m in list(sys.modules.values()) if os.path.isabs(getattr(m, '__file__', None) or '') and os.path.realpath(m.__file__).startswith(repo + os.sep) and '/temp/' not in rel(m.__file__)})))\n")
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), cwd=os.path.join(J.ROOT, "evaluation"))
    assert r.returncode == 0, r.stderr[-800:]
    loaded = json.loads(r.stdout.strip().splitlines()[-1])
    digest = J.code_sha256(entry=os.path.join(J.ROOT, "evaluation", "jtest_eval.py"))
    assert loaded and not [x for x in loaded if x not in digest], [x for x in loaded if x not in digest]
    assert any(x.endswith("evaluation/sparse_lanes/topo_metrics.py") for x in loaded) and any(x.endswith("evaluation/sparse_lanes/topo_metrics.py") for x in digest), "the scorer's helper modules are loaded and in the digest"
    assert not [x for x in digest if x.endswith(("autoresearch/Evaluation/render/video.py", "autoresearch/Evaluation/api.py", "autoresearch/suitecli.py", "evaluation/evaluate.py"))], "files that nothing imports at import time (and the gate's evaluator) are not in it"


# ------------------------------------------------------------------------------------------------------------------------------ the reading rule
def seeds(D, lo, guards=(True, True), hi=0.03):
    return [dict(D=d, lo=l, hi=hi, guards=g) for d, l, g in zip(D, lo, guards)]


def test_reading_boundaries():
    assert J.reading(seeds((0.0100, 0.0100), (0.0001, 0.0001))) == "FULL RETENTION", "D = 0.0100 exactly is full retention"
    assert J.reading(seeds((0.0100, 0.0099), (0.0001, 0.0001))) == "NOT CONTRADICTED", "D just under 0.0100 on one seed is not full"
    assert J.reading(seeds((0.0200, 0.0200), (0.0001, 0.0))) == "NOT CONTRADICTED", "lo = 0 is not > 0: not full retention"
    assert J.reading(seeds((0.0200, 0.0200), (0.0100, -0.0049))) == "NOT CONTRADICTED"
    assert J.reading(seeds((0.0200, 0.0200), (0.0100, -0.0050))) == "CONTRADICTED", "lo = -0.0050 exactly is not > -0.005"
    assert J.reading(seeds((0.0100, 0.0001), (0.0100, 0.0001))) == "NOT CONTRADICTED" and J.reading(seeds((0.0100, 0.0), (0.0100, 0.0001))) == "CONTRADICTED", "D > 0 is strict"
    assert J.reading(seeds((0.02, 0.02), (0.01, 0.01), guards=(True, False))) == "CONTRADICTED", "a guard miss on one seed is contradicted"
    assert J.reading(seeds((0.02, 0.02), (0.01, 0.01), guards=(False, False))) == "CONTRADICTED"
    assert J.reading(seeds((0.02, float("nan")), (0.01, 0.01))) == "CONTRADICTED", "a nan D fails every comparison"


def test_reading_fallback_without_interval():
    nolo = (None, None)
    assert J.reading(seeds((0.0100, 0.0100), nolo)) == "FULL RETENTION (without interval)"
    assert J.reading(seeds((0.0100, 0.0099), nolo)) == "NOT CONTRADICTED (without interval)"
    assert J.reading(seeds((0.0050, 0.0050), nolo)) == "NOT CONTRADICTED (without interval)" and J.reading(seeds((0.0050, 0.0049), nolo)) == "CONTRADICTED (without interval)"
    assert J.reading(seeds((0.0200, 0.0200), nolo, guards=(True, False))) == "CONTRADICTED (without interval)", "the guards hold in the fallback too"
    # the per-frame counts of E312 reproduce on one seed only: no interval on EITHER seed, the interval of the one seed that has it is not used
    assert J.reading(seeds((0.0200, 0.0075), (-0.02, None))) == "NOT CONTRADICTED (without interval)", "seed 0's interval would contradict, the fallback ignores it"
    assert J.reading(seeds((0.0200, 0.0075), (0.01, float("nan")))) == "NOT CONTRADICTED (without interval)", "a nan interval is no interval"
    assert raises(AssertionError, J.reading, [])


def test_guards_against_e312():
    ref = dict(zip(J.KEYS, [0.5] * len(J.KEYS)))
    ref.update(vert100_dp=19.0)
    arm = lambda **d: dict(ref, **{k: ref[k] + v for k, v in d.items()})  # noqa: E731
    assert J.guards_vs(arm(), ref)["holds"]
    assert J.guards_vs(arm(**{"F1@1.0": -0.0099}), ref)["holds"] and not J.guards_vs(arm(**{"F1@1.0": -0.0101}), ref)["holds"]
    assert J.guards_vs(arm(dup_frac=0.0199), ref)["holds"] and not J.guards_vs(arm(dup_frac=0.0201), ref)["holds"]
    assert J.guards_vs(arm(vert100_dp=1.0), ref)["holds"] and not J.guards_vs(arm(vert100_dp=1.5), ref)["holds"]
    assert J.guards_vs(arm(junction_F1=-0.0299), ref)["holds"] and not J.guards_vs(arm(junction_F1=-0.0301), ref)["holds"]
    g = J.guards_vs(arm(dup_frac=0.015), ref)
    assert g["holds"] and not g["dup_frac_e324_val_bound_ok"], "E324's own VAL bound (0.01) is printed, D79's (0.02) decides"
    assert J.guards_vs(arm(topo_F1=-0.2), ref)["holds"], "the topo F1 gain is the reading's business, not a guard"


# ------------------------------------------------------------------------------------------------------------------------------ comparing numbers and counts
def test_match_and_counts_maxdiff():
    ex, rh = J.EXACT, J.REHEARSAL
    assert J.match(dict(a=1.0, b=dict(c=2.0)), dict(a=1.0, b=dict(c=2.0)), ex)[0]
    ok, worst, where = J.match(dict(a=1.0, b=dict(c=2.0)), dict(a=1.0, b=dict(c=2.5)), ex)
    assert not ok and worst == 0.5 and where == "/b/c"
    assert not J.match(dict(a=1.0), dict(a=1.0, b=2.0), ex)[0], "a number of the stored card that is missing fails"
    assert J.match(dict(n=5774), dict(n=5773), rh)[0] and not J.match(dict(n=5774), dict(n=5773), ex)[0], "integers: relative 5e-4 (E321's 11379 against 11378), exactly under EXACT"
    assert J.match(dict(x=0.6931), dict(x=0.6912), rh)[0] and not J.match(dict(x=0.6931), dict(x=0.6912), ex)[0]
    assert J.match(dict(topo_F1=0.6909), dict(topo_F1=0.6912), rh)[0] and not J.match(dict(topo_F1=0.6931), dict(topo_F1=0.6912), rh)[0], "topo F1 has the tighter tolerance"
    assert not J.match(dict(x=float("nan")), dict(x=1.0), rh)[0]
    c = dict(rec=dict(a=1.0), dp=np.array([1.0, 2.0]), tier=None, lane=dict(m=3))
    assert J.counts_maxdiff(c, dict(rec=dict(a=1.0), dp=np.array([1.0, 2.0]), tier=None, lane=dict(m=3))) == 0.0
    assert abs(J.counts_maxdiff(c, dict(rec=dict(a=1.0), dp=np.array([1.0, 2.0 + 1e-12]), tier=None, lane=dict(m=3))) - 1e-12) < 1e-15
    assert J.counts_maxdiff(c, dict(rec=dict(a=1.0), dp=np.array([1.0, 2.0]), tier=None)) == float("inf"), "another structure"
    assert J.counts_maxdiff([c, c], [c]) == float("inf") and J.counts_maxdiff(np.array([np.nan]), np.array([np.nan])) == 0.0 and J.counts_maxdiff(np.array([np.nan]), np.array([1.0])) == float("inf")


# ------------------------------------------------------------------------------------------------------------------------------ the glue of the scoring: which read-out reads which graph, pair dump and ends
class Frame:
    def __init__(self, row, scene):
        self.row, self.scene, self._lane, self._ref = row, scene, None, None


class Graph:
    def __init__(self, marker):
        self.marker, self.lines, self.edges, self.nodes = marker, [], [], []


def toy_pair_dump(row, off):
    kp = np.array([[off + i, 0.0] for i in range(4)])
    return dict(row=row, kp_pos=kp, kp_dir=np.tile([0.0, 1.0], (4, 1)), att_pos=kp + 100.0, att_dir=np.tile([1.0, 0.0], (4, 1)), logit=np.array([[9.0, -9.0], [-9.0, -9.0]]), allowed=np.ones((2, 2)), scale=None)


def toy_graph_dump(row, marker):
    return dict(row=row, pos=marker, dirv=0, e=0, flags=0, section=0, link_src=0, link_dst=0, link_p=0, link_long=[True], link_a=0, link_b=0)


def test_score_dumps_glue():
    """the variants of E324 s + the stack + the control, on two frames, with the read-out and the scorer replaced by spies: arm w reads the keypoints, p the attached ends with snap_nogeom, k the graph of c2, the control its own pair dump;
    a dump list that two names share (b and ctl) is read out once per frame"""
    import qends_eval as QEV
    import qgraph_eval as QE
    import qgraph_readout as QR
    calls, reads = [], []
    frames = [Frame(10, 0), Frame(11, 0)]
    b = [toy_graph_dump(f.row, "B") for f in frames]
    c2 = [toy_graph_dump(f.row, "C2") for f in frames]
    ends = {"own": {f.row: toy_pair_dump(f.row, 0.0) for f in frames}, "ctl": {f.row: toy_pair_dump(f.row, 50.0) for f in frames}}

    def readout(pred, *a):
        reads.append(pred["marker"])
        g = Graph(pred["marker"])
        g.lines = pred["marker"]                                                                       # the toy graph carries its marker in `lines`: the spy of add_connectors sees which graph the connectors sit on
        return g, dict(kind=[], piece_links=[])

    def add_connectors(lines, edges, nodes, kinds, links, long_mask, pairs, snap=False, snap_nogeom=False, **k):
        assert not k, f"an option the stored cards do not use: {k}"
        calls.append((lines, snap, snap_nogeom, float(pairs[0][0][0]), len(pairs)))
        return dict(pts=[lines]), len(pairs), 1
    variants = [dict(v, man=None) for v in J.variants_of(J.SETUPS["e324_s_s"], True, True)]
    spies = ((QR, "apply", lambda pred, spec: dict(marker=pred["nodes"]["pos"], links=dict(long=pred["links"]["long"]))), (QE, "readout", readout), (QE, "to_raw", lambda g: dict(pts=[g.marker])),
             (QE, "graph_counts", lambda g, f: dict(marker=g.marker if isinstance(g, Graph) else g["pts"][0])), (QEV, "add_connectors", add_connectors))
    with patched(*spies):
        sc = J.score_dumps(frames, {"b": b, "c2": c2, "ctl": b}, ends, variants, log=lambda *a: None)
    want = [("B", True, False, 0.0, 1), ("B", True, True, 0.0, 1), ("B", True, True, 100.0, 1), ("C2", True, False, 0.0, 1), ("B", True, False, 50.0, 1)]      # snapped, snap_nogeom, primary, stack, e312_snapped
    assert calls == want * 2, calls
    assert reads.count("B") == 2 and reads.count("C2") == 2, reads
    assert sc["curves"] == {n: 2 for n in ("snapped", "snap_nogeom", "primary", "stack", "e312_snapped")} == sc["wired"] and sc["man"] == {}
    assert [c["marker"] for c in sc["counts"]["c2_plain"]] == ["C2", "C2"] and [c["marker"] for c in sc["counts"]["plain"]] == ["B", "B"] and [c["marker"] for c in sc["counts"]["stack"]] == ["C2", "C2"]
    calls.clear()                                                                                       # E312's setup: the unsnapped read-out is the only one without snap
    with patched(*spies):
        J.score_dumps(frames, {"b": b}, ends, [dict(v, man=None) for v in J.variants_of(J.SETUPS["e312_p_s"])], log=lambda *a: None)
    assert calls == [("B", False, False, 0.0, 1), ("B", True, False, 0.0, 1)] * 2, calls


def test_pairs_use_attached_ends_only_when_asked():
    e = toy_pair_dump(0, 0.0)
    kp, at = J.pairs_of(e, 0.1)[0], J.pairs_of(e, 0.1, attach=True)[0]
    assert np.allclose(kp[0], [0, 0]) and np.allclose(kp[2], [2, 0]) and np.allclose(at[0], [100, 100]) and np.allclose(at[2], [102, 100]) and len(kp) == 4
    assert len(J.pairs_of(dict(e, scale=np.full((2, 2, 2), 0.3)), 0.1)[0]) == 6 and len(J.pairs_of(e, 0.9999)) == 0


# ------------------------------------------------------------------------------------------------------------------------------ the checks of one seed, the card, results.md
def toy_numbers(counts):
    cs = list(counts)
    x, d, v = float(np.mean([c["x"] for c in cs])), float(np.mean([c.get("d", 0.2) for c in cs])), float(np.mean([c.get("v", 19.0) for c in cs]))
    return {**{k: x for k in J.KEYS}, "dup_frac": d, "vert100_dp": v}


def toy_sc(variants, xs, frames, ds=None):
    """per-frame counts of toy variants: xs[name] a number or a function of the frame, ds[name] the dup_frac (default 0.2)"""
    ds = ds or {}
    val = lambda v, f: v(f) if callable(v) else v  # noqa: E731
    return dict(counts={v["name"]: [dict(x=val(xs[v["name"]], f), d=ds.get(v["name"], 0.2)) for f in frames] for v in variants}, man={v["name"]: [(1, 2, 3, 4)] for v in variants if v["man"]},
                curves={v["name"]: 7 for v in variants if v["ends"]}, wired={v["name"]: 5 for v in variants if v["ends"]})


def toy_card_file(d, head, base, curves=7, wired=5):
    p = os.path.join(d, "control_card.json")
    num = lambda x: dict({k: x for k in J.KEYS}, vert100_dp=19.0, dup_frac=0.2)  # noqa: E731
    json.dump(dict(variants=dict(snapped=dict(numbers=dict(head=num(head), base=num(base)), curves=curves, wired_both_ends=wired))), open(p, "w"))
    return p


PROV = lambda run, device: dict(run=run, device=device, code_digest="c", checkpoint_sha256="k")  # noqa: E731


def run_toy_seed(d, xs, card_base=0.5, ctl_head=0.6, stack=True, split="val", ds=None, card_curves=7, card_wired=5, b_device="cpu", ctl_prov=None, ctl_b_device=None):
    """precheck + score_seed on toy files (all dumps exist, with provenance: none may be made) with the scorer replaced; returns (what score_seed returns, the frames)"""
    import qgraph_eval as QE
    full = toy_dump(split=split)
    frames = [Frame(r["row"], r["row"] % 3) for r in full]
    for name, run in (("predfull_e273_p_s0", "e273_p_s0"), ("ends_e324_s_s0", "e324_s_s0"), ("predfull_e292_c2_s0", "e292_c2_s0")):
        pickle.dump(full, open(os.path.join(d, name + ".pkl"), "wb"))
        json.dump(dict(provenance=PROV(run, b_device if run == "e273_p_s0" else "cpu")), open(os.path.join(d, name + "_head.json"), "w"))
    pickle.dump(full, open(os.path.join(d, "ctl_ends.pkl"), "wb"))
    if os.path.exists(os.path.join(d, "ctl_ends_head.json")):
        os.remove(os.path.join(d, "ctl_ends_head.json"))                                              # a sidecar of an earlier call of this helper in the same folder (scratch of this test only)
    if ctl_prov:
        json.dump(dict(provenance=ctl_prov), open(os.path.join(d, "ctl_ends_head.json"), "w"))
    card = toy_card_file(d, ctl_head, card_base, card_curves, card_wired)
    a = argparse.Namespace(split=split, max=0, device="cpu", run_prefix="e324_s_s", b_dump=None, control_ends_dump=os.path.join(d, "ctl_ends.pkl"), control_card=card)
    defaults = []
    if ctl_b_device:                                                                                   # the control's graph is E312's own B dump (the split's defaults), not the winner's
        pickle.dump(full, open(os.path.join(d, "ctl_b.pkl"), "wb"))
        json.dump(dict(provenance=PROV("e273_p_s0", ctl_b_device)), open(os.path.join(d, "ctl_b_head.json"), "w"))
        a.control_ends_dump = a.control_card = None
        defaults = [(J, "CONTROL_DEFAULTS", {split: dict(ends=os.path.join(d, "ctl_ends.pkl"), b=os.path.join(d, "ctl_b.pkl"), card=card)})]
    with patched(*defaults, (J, "dump_dir", lambda split, n=0: d), (J, "dump_b", boom), (J, "dump_ends", boom), (J, "pyramid_same", lambda *x: True), (J, "provenance", PROV), (QE, "graph_numbers", toy_numbers),
                 (J, "score_dumps", lambda fr, graphs, ends, variants, thr=J.THR, n=0, log=print: toy_sc(variants, xs, fr, ds))):
        pre = J.precheck(a, J.SETUPS["e324_s_s"], 0, stack)
        return J.score_seed(a, J.SETUPS["e324_s_s"], 0, frames, stack, pre), frames


XS = dict(plain=0.5, snapped=0.62, snap_nogeom=0.61, primary=0.60, c2_plain=0.52, stack=0.64, e312_snapped=0.6)


def test_score_seed_control_checks():
    with tempfile.TemporaryDirectory() as d:
        r, _ = run_toy_seed(d, XS)                                                                                       # existing dumps reused (dump_b / dump_ends raise when called)
        c = r["control"]
        assert c["plain_check"]["ok"] and c["repro"]["ok"] and c["repro"]["max_abs_diff"] < 1e-12 and c["counts"] is r["sc"]["counts"]["e312_snapped"], "E312's counts reproduce: they are used"
        assert [v["name"] for v in r["variants"]][-1] == "e312_snapped"
        r, _ = run_toy_seed(d, dict(XS, e312_snapped=0.6 + 1e-7))
        assert not r["control"]["repro"]["ok"] and r["control"]["counts"] is None, "a difference of 1e-7 is no reproduction (1e-9): no per-frame counts, no interval"
        for kw in (dict(card_curves=6), dict(card_wired=4)):
            r, _ = run_toy_seed(d, XS, **kw)
            assert not r["control"]["repro"]["ok"] and r["control"]["counts"] is None and r["control"]["repro"]["max_abs_diff"] < 1e-12, f"the numbers are equal but E312's curves / wired counts differ ({kw}): no reproduction"
        assert raises(SystemExit, run_toy_seed, d, XS, card_base=0.51), "the plain graph differs from the control card's numbers.base by 0.01 (VAL tolerance 3e-3): KILL before any number is printed"
        r, _ = run_toy_seed(d, XS, card_base=0.5 + 5e-4)
        assert r["control"]["plain_check"]["ok"], "a fresh CPU dump within the VAL tolerance passes on VAL"
        assert J.tol_of("test") is J.EXACT and J.tol_of("val") is J.REHEARSAL, "TEST: 1e-9 for the plain-graph check"
        # the TEST branch: B's dump is E321's (not made), the tolerance is 1e-9: 5e-4 is a KILL there, 1e-10 passes
        assert raises(SystemExit, run_toy_seed, d, XS, split="test", card_base=0.5 + 5e-4), "TEST: the plain graph is equal to 1e-9 or no number is used"
        r, _ = run_toy_seed(d, XS, split="test", card_base=0.5 + 1e-10)
        assert r["control"]["plain_check"]["ok"] and r["control"]["repro"]["ok"]
        # provenance of dumps named by a flag or made earlier: B's dump on TEST is not made, a sidecar that says another device refuses it; the control's pair dump likewise; the default dumps need theirs
        assert raises(SystemExit, run_toy_seed, d, XS, split="test", b_device="cuda"), "B's dump (flagged / not made) was made on another device: refused"
        assert raises(SystemExit, run_toy_seed, d, XS, ctl_prov=PROV("e312_p_s0", "cuda")) and raises(SystemExit, run_toy_seed, d, XS, ctl_prov=PROV("e312_p_s1", "cpu")), "the control's pair dump of another device / run: refused"
        r, _ = run_toy_seed(d, XS, ctl_prov=PROV("e312_p_s0", "cpu"))
        assert r["control"]["repro"]["ok"], "the control's pair dump with its own provenance passes"
        assert run_toy_seed(d, XS, ctl_b_device="cpu")[0]["control"]["repro"]["ok"] and raises(SystemExit, run_toy_seed, d, XS, ctl_b_device="cuda"), "the control's own B dump (the split's default) is checked too"
        r, _ = run_toy_seed(d, XS, stack=False)
        assert "stack" not in r["sc"]["counts"] and "c2_plain" not in r["sc"]["counts"] and all(v["graph"] != "c2" for v in r["variants"]), "--no-stack drops arm k only"


def test_precheck_refuses_before_any_dump_is_made():
    ns = lambda **k: argparse.Namespace(**dict(dict(split="val", max=0, device="cpu", run_prefix="e324_s_s", b_dump=None, control_ends_dump=None, control_card=None), **k))  # noqa: E731
    setup = J.SETUPS["e324_s_s"]
    with tempfile.TemporaryDirectory() as d, patched((J, "pyramid_same", lambda *x: True), (J, "dump_dir", lambda split, n=0: d), (J, "provenance", PROV)):
        card = toy_card_file(d, 0.6, 0.5)
        i, ctl = J.precheck(ns(control_card=card), setup, 0, True)
        assert i["run"] == "e324_s_s0" and ctl["variants"]["snapped"]["curves"] == 7, "the control card is read at the start"
        seen = []
        with patched((J, "check_seed", lambda run, seed: seen.append((run, seed)))):
            J.precheck(ns(control_card=card), setup, 1, True)
        assert seen == [("e324_s_s1", 1)], "the seed of the pair run is checked in the precheck"
        assert raises(SystemExit, J.precheck, ns(control_card=os.path.join(d, "missing.json")), setup, 0, True), "a missing control card stops before the dumping"
        assert raises(SystemExit, J.precheck, ns(control_card=card, b_dump=os.path.join(d, "missing_b.pkl")), setup, 0, True), "a flagged B dump that is not there"
        assert raises(SystemExit, J.precheck, ns(control_card=card, control_ends_dump=os.path.join(d, "missing_ends.pkl")), setup, 0, True), "a flagged control pair dump that is not there"
        with patched((J, "dump_dir", lambda split, n=0: d)):
            assert raises(SystemExit, J.precheck, ns(split="test", control_card=card), setup, 0, True), "TEST without --b-dump and without a dump at the default path: B's dump is never made on TEST"
        with patched((J, "pyramid_same", lambda *x: False)):
            assert raises(SystemExit, J.precheck, ns(control_card=card), setup, 0, True), "arm k fails the pyramid check: stop, to be run again with --no-stack"
            assert J.precheck(ns(control_card=card), setup, 0, False)[0]["run"] == "e324_s_s0", "--no-stack does not look at the pyramid"
    with tempfile.TemporaryDirectory() as d, patched((J, "ROOT", d)):
        os.makedirs(os.path.join(d, "runs", "e324_s_s0"))
        json.dump(dict(seed=1), open(os.path.join(d, "runs", "e324_s_s0", "config.json"), "w"))
        assert raises(SystemExit, J.check_seed, "e324_s_s0", 0), "the run of another seed: its keypoints would sit on another seed's graph"
        J.check_seed("e324_s_s0", 1)


OLD_HEAD = dict(ent_recall=0.7, ent_precision=0.7, ext_recall=0.7, ext_precision=0.7, pair_recall=0.4, pair_precision=0.7, counts={}, thr=0.5, frames=833, run="e312_p_s0")      # the `_head.json` that E321's old dump_ends wrote: no provenance key


def test_a_flagged_dump_with_an_old_style_head_file_is_exempt_and_refusals_come_in_the_precheck():
    """E321's pair dump ends_e312_p_s<S>.pkl has a `_head.json` WITHOUT a provenance key: it is not a dump of this module (exempt, validated by its rows and, E321's, by hash); a flagged dump WITH provenance is checked; every refusal, flagged or
    default dump, comes in `precheck` before a dump of any seed is made; the message of a default dump without a sidecar names both causes"""
    full = toy_dump()
    prov = PROV("e312_p_s0", "cpu")
    setup = J.SETUPS["e324_s_s"]
    ns = lambda **k: argparse.Namespace(**dict(dict(split="val", max=0, device="cpu", run_prefix="e324_s_s", b_dump=None, control_ends_dump=None, control_card=None), **k))  # noqa: E731
    with tempfile.TemporaryDirectory() as d, patched((J, "pyramid_same", lambda *x: True), (J, "dump_dir", lambda split, n=0: d), (J, "provenance", PROV), (J, "dump_b", boom), (J, "dump_ends", boom)):
        p, head = os.path.join(d, "ends_e312_p_s0.pkl"), os.path.join(d, "ends_e312_p_s0_head.json")
        pickle.dump(full, open(p, "wb"))
        json.dump(OLD_HEAD, open(head, "w"))
        assert J.sidecar_provenance(p) is None and len(J.get_dump(p, None, "val", "x", expect=prov)) == 873, "an old-style head file (no provenance key) is no provenance: the dump is exempt"
        a = ns(control_ends_dump=p, control_card=toy_card_file(d, 0.6, 0.5))
        assert J.precheck(a, setup, 0, False)[0]["ctl_ends"] == p, "the R1 line of the TEST command: E321's pair dump is not refused (and the winner's B dump, still to be made, is not 'missing')"
        json.dump(dict(OLD_HEAD, provenance=PROV("e312_p_s0", "cuda")), open(head, "w"))
        assert raises(SystemExit, J.precheck, a, setup, 0, False), "a flagged dump that THIS module made, on another device: refused in the precheck"
        json.dump(dict(OLD_HEAD, provenance=PROV("e312_p_s0", "cpu")), open(head, "w"))
        assert J.precheck(a, setup, 0, False)[0]["ctl_ends"] == p
        json.dump("not a dict", open(head, "w"))
        assert J.sidecar_provenance(p) is None, "a head file that is not a mapping is no provenance"
        own = os.path.join(d, "ends_e324_s_s0.pkl")                                                    # a default dump that exists: the precheck needs its provenance (the pickle is there, the sidecar is not)
        pickle.dump(full, open(own, "wb"))
        try:
            J.precheck(a, setup, 0, False)
            raise AssertionError("a default dump without its sidecar must be refused in the precheck")
        except SystemExit as e:
            assert "made by other code OR interrupted between the pickle and its sidecar" in str(e) and "move it away" in str(e), str(e)
        json.dump(dict(provenance=PROV("e324_s_s0", "cuda")), open(os.path.join(d, "ends_e324_s_s0_head.json"), "w"))
        assert raises(SystemExit, J.precheck, a, setup, 0, False), "a default dump made on another device: refused in the precheck"
        json.dump(dict(provenance=PROV("e324_s_s0", "cpu")), open(os.path.join(d, "ends_e324_s_s0_head.json"), "w"))
        assert J.precheck(a, setup, 0, False)[0]["ends"] == own


def test_provenance_is_taken_before_the_model_is_loaded():
    """the provenance of a dump is the code and the checkpoint when the dump STARTS: `provenance` is called before the frames are read and the model is loaded (the frames function stops the call)"""
    seen = []

    class Stop(Exception):
        pass

    def stop(split):
        raise Stop()
    with tempfile.TemporaryDirectory() as d, patched((J, "provenance", lambda run, device: seen.append((run, device)) or PROV(run, device)), (J, "frames_of", stop), (J, "EXTRA_WRITABLE", (d,))):
        assert raises(Stop, J.dump_b, "val", "e273_p_s0", os.path.join(d, "b.pkl"), "cpu") and seen == [("e273_p_s0", "cpu")], seen
        seen.clear()
        assert raises(Stop, J.dump_ends, "val", "e324_s_s0", os.path.join(d, "e.pkl"), "cpu") and seen == [("e324_s_s0", "cpu")], seen


def test_card_arms_and_results_md():
    import qgraph_eval as QE
    setup = J.SETUPS["e324_s_s"]
    with tempfile.TemporaryDirectory() as d, patched((QE, "graph_numbers", toy_numbers)):
        cards = []
        for seed, gain in ((0, 0.016), (1, 0.014)):
            xs = dict(XS, snapped=0.60 + gain, e312_snapped=0.60)
            r, frames = run_toy_seed(d, xs)
            cards.append(J.make_card(seed, f"e324_s_s{seed}", "val", frames, r["sc"], 20, setup, r["variants"], r["control"], True))
        c = cards[0]
        w = c["compare"]["vs_e312"]["w"]
        assert abs(w["difference"]["topo_F1"] - 0.016) < 1e-12 and w["guards"]["holds"] and set(c["compare"]["vs_e312"]) == {"w", "p", "k"} and "k_minus_w" in c["compare"] and "w_minus_p" in c["compare"]
        assert abs(w["paired"]["topo_F1"]["diff"] - 0.016) < 1e-12 and abs(w["paired"]["topo_F1"]["lo"] - 0.016) < 1e-9 and abs(w["paired"]["topo_F1"]["hi"] - 0.016) < 1e-9, "toy: every frame is 0.016 better, so the interval is a point"
        assert abs(c["compare"]["k_minus_w"]["topo_F1"]["diff"] - 0.024) < 1e-12 and abs(c["compare"]["w_minus_p"]["topo_F1"]["diff"] - 0.016) < 1e-12
        assert c["variants"]["stack"]["base"] == "c2_plain" and abs(c["variants"]["stack"]["difference"]["topo_F1"] - 0.12) < 1e-12 and "e312_snapped" not in c["variants"], "k is compared with c2's plain graph; the control is not a variant of the card"
        assert set(c["maneuver"]) == {"plain", "p0.1s", "p0.1an", "c2 plain", "c2 p0.1s"}, "snap_nogeom has no maneuver label"
        assert J.reading([J.reading_inputs(x) for x in cards]) == "FULL RETENTION"
        txt = J.results_md(cards, "val", setup)
        assert "**FULL RETENTION**" in txt and "may become the system of record" in txt and "w - p" in txt and "E2E gap" in txt and "not verified against a stored card" in txt and "snap_nogeom" in txt
        # without E312's per-frame counts: no interval, the fallback label
        cards_nc = []
        for seed, gain in ((0, 0.016), (1, 0.014)):
            r, frames = run_toy_seed(d, dict(XS, snapped=0.60 + gain, e312_snapped=0.6 + 1e-7))
            cards_nc.append(J.make_card(seed, f"e324_s_s{seed}", "val", frames, r["sc"], 20, setup, r["variants"], r["control"], False))
        assert "paired" not in cards_nc[0]["compare"]["vs_e312"]["w"] and J.reading([J.reading_inputs(x) for x in cards_nc]) == "FULL RETENTION (without interval)"
        assert "**FULL RETENTION (without interval)**" in J.results_md(cards_nc, "val", setup)
        assert "not applied" in J.results_md(cards[:1], "val", setup), "the rule needs both seeds"
        assert json.loads(json.dumps(cards[0])), "the card is JSON"


def test_per_arm_values_guards_against_e312_and_intervals():
    """asymmetric seeds (seed 1: D = 0.008), per-arm differences, an arm that misses a guard against E312 only (its own plain graph has the same dup_frac), an interval with lo != hi: every table of results.md shows the right arm and the right seed"""
    import qgraph_eval as QE
    setup = J.SETUPS["e324_s_s"]
    gains = {0: dict(w=0.016, p=0.004, k=0.012), 1: dict(w=0.008, p=0.003, k=0.010)}
    noise = lambda f: 0.02 * ((f.row % 3) - 1)  # noqa: E731
    with tempfile.TemporaryDirectory() as d, patched((QE, "graph_numbers", toy_numbers)):
        cards = []
        for seed in (0, 1):
            g = gains[seed]
            xs = dict(XS, snapped=(lambda f, g=g: 0.6 + g["w"] + noise(f)) if seed == 0 else 0.6 + g["w"], primary=0.6 + g["p"], stack=0.6 + g["k"], e312_snapped=0.6)
            r, frames = run_toy_seed(d, xs, ds=dict(stack=0.25, c2_plain=0.25))
            cards.append(J.make_card(seed, f"e324_s_s{seed}", "val", frames, r["sc"], 30, setup, r["variants"], r["control"], True))
        for c in cards:
            v = c["compare"]["vs_e312"]
            assert v["w"]["guards"]["holds"] and v["p"]["guards"]["holds"] and not v["k"]["guards"]["holds"] and not v["k"]["guards"]["conditions"]["dup_frac"], "arm k rises 0.05 in dup_frac against E312 (its own plain graph rises 0): a miss against E312 only"
        inp = J.reading_inputs(cards[0])
        p0 = cards[0]["compare"]["vs_e312"]["w"]["paired"]["topo_F1"]
        assert inp["lo"] < inp["hi"] and (inp["lo"], inp["hi"]) == (p0["lo"], p0["hi"]), "lo and hi come out in their own places"
        assert abs(J.reading_inputs(cards[1])["lo"] - J.reading_inputs(cards[1])["hi"]) < 1e-9 and abs(J.reading_inputs(cards[1])["D"] - 0.008) < 1e-12
        txt = J.results_md(cards, "val", setup)
        assert "**NOT CONTRADICTED**" in txt, "seed 1 has D = 0.008 < 0.010: not full retention, but D > 0, lo > -0.005 and the guards hold on both seeds"
        row = lambda arm, seed, table: [ln for ln in txt.splitlines() if ln.startswith(f"| {arm} ") and f" | {seed} | " in ln and table in ln][0]  # noqa: E731
        for arm, label in (("w", "snapped"), ("p", "primary"), ("k", "stack")):
            for seed in (0, 1):
                ln = row(f"{arm} ({label})", seed, "| 0.6")
                want = cards[seed]["compare"]["vs_e312"][arm]["difference"]["topo_F1"]
                assert abs(want - gains[seed][arm]) < (0.003 if (arm, seed) == ("w", 0) else 1e-12), (arm, seed, want)
                assert f"| {want:+.4f}" in ln and f"| {0.6 + want:.4f} |" in ln, (arm, seed, ln)
                assert ("MISS" in ln) == (arm == "k"), ln
        assert any(ln.startswith("| 1 | +0.008000 |") for ln in txt.splitlines()), "the reading table shows D with 6 digits for seed 1"
        assert any(ln.startswith("| 0 | ") and "MISS" not in ln for ln in txt.splitlines() if "hold" in ln), "the reading table has seed 0 too"


def test_intervals_only_when_the_counts_reproduced_on_every_seed():
    import qgraph_eval as QE
    setup = J.SETUPS["e324_s_s"]
    with tempfile.TemporaryDirectory() as d, patched((QE, "graph_numbers", toy_numbers)):
        ok, frames = run_toy_seed(d, XS)
        bad, _ = run_toy_seed(d, dict(XS, e312_snapped=0.6 + 1e-7))
        assert J.counts_reproduced([ok, ok]) and not J.counts_reproduced([ok, bad]) and not J.counts_reproduced([bad, ok]) and not J.counts_reproduced([bad, bad]), "one seed that does not reproduce: no interval on either"
        c_no = J.make_card(0, "e324_s_s0", "val", frames, ok["sc"], 20, setup, ok["variants"], ok["control"], False)
        c_yes = J.make_card(0, "e324_s_s0", "val", frames, ok["sc"], 20, setup, ok["variants"], ok["control"], True)
        assert "paired" not in c_no["compare"]["vs_e312"]["w"] and "paired" in c_yes["compare"]["vs_e312"]["w"], "use_counts decides, not only the presence of the counts"


def test_inputs_per_split():
    """paths only: which dumps and which control a split reads (the TEST command names B's dump and E312's card; E312's pair dump only by flag; B's dump is never made on TEST)"""
    ns = lambda **k: argparse.Namespace(**dict(dict(split="val", max=0, run_prefix="e324_s_s", b_dump=None, control_ends_dump=None, control_card=None), **k))  # noqa: E731
    setup = J.SETUPS["e324_s_s"]
    r = lambda p: os.path.relpath(p, J.ROOT).replace(os.sep, "/")  # noqa: E731
    i = J.seed_inputs(ns(), setup, 1, True, False)
    assert i["b_made"] and r(i["b"]) == "temp/e343/val/predfull_e273_p_s1.pkl" and r(i["ends"]) == "temp/e343/val/ends_e324_s_s1.pkl" and r(i["c2"]) == "temp/e343/val/predfull_e292_c2_s1.pkl"
    assert r(i["ctl_ends"]) == "runs/e312_p_s1/ends_val.pkl" and r(i["ctl_b"]) == "temp/e276/predfull_e273_p_s1.pkl" and r(i["card"]) == "runs/e312_p_s1/ends_card_snap_thr0.1.json", "VAL: the control is E312's stored dumps and card"
    t = J.seed_inputs(ns(split="test", b_dump="temp/e321/test/predfull_e273_p_s{seed}.pkl", control_card="runs/eval/e312_test/card_s{seed}.json"), setup, 0, True, False)
    assert not t["b_made"] and t["b"] == "temp/e321/test/predfull_e273_p_s0.pkl" and t["ctl_ends"] is None and t["ctl_b"] == t["b"] and t["card"].endswith("runs/eval/e312_test/card_s0.json"), "TEST: no re-score of E312 without the flag"
    assert r(t["ends"]) == "temp/e343/test/ends_e324_s_s0.pkl" and r(t["c2"]) == "temp/e343/test/predfull_e292_c2_s0.pkl", "the new dumps go to temp/e343/test"
    t2 = J.seed_inputs(ns(split="test", control_ends_dump="temp/e321/test/ends_e312_p_s{seed}.pkl"), setup, 1, False, True)
    assert not t2["b_made"] and t2["ctl_ends"] == "temp/e321/test/ends_e312_p_s1.pkl" and t2["ctl_b"] == t2["b"] and t2["c2"] is None and r(t2["card"]) == "runs/eval/e312_test/card_s1.json", "TEST by flag: E312's graph is the same B dump"
    v2 = J.seed_inputs(ns(control_ends_dump="x/ends_{seed}.pkl"), setup, 0, True, True)
    assert v2["ctl_ends"] == "x/ends_0.pkl" and v2["ctl_b"] == v2["b"], "a control pair dump named by a flag is scored on the winner's B dump, on VAL too"
    assert not J.seed_inputs(ns(split="test"), setup, 0, True, False)["b_made"], "B is never dumped on TEST: E321's dump is reused"
    assert r(J.seed_inputs(ns(max=8), setup, 0, True, False)["b"]) == "temp/e343/val_max8/predfull_e273_p_s0.pkl", "a smoke run has its own folder"
    assert r(J.out_dir(setup, "val", 8)) != r(J.out_dir(J.SETUPS["e312_p_s"], "val", 8)) and r(J.out_dir(setup, "val")) == "runs/eval/e324s_val", "smoke cards of two systems do not meet; a full run writes to runs/eval/e324s_<split>"


# ------------------------------------------------------------------------------------------------------------------------------ cmd_all end to end
def toy_cmd_all(d, a, counts1=True, check=None, codes=None):
    """`cmd_all` with the dumps, the checks of the inputs and the scorer replaced -> (rc or the SystemExit, text, files of the output folder, calls of guard_e321, calls of score_seed)"""
    import qgraph_eval as QE
    setup = J.SETUPS["e324_s_s"]
    frames = [Frame(r["row"], r["row"] % 3) for r in toy_dump()]
    scored, guards, buf = [], [], io.StringIO()
    code_iter = iter(codes or [{"a.py": "1"}, {"a.py": "1"}])

    def score(a_, st, seed, fr, stack, pre):
        scored.append(seed)
        variants = J.variants_of(setup, True, True)
        sc = toy_sc(variants, dict(XS, snapped=0.60 + (0.016 if seed == 0 else 0.014), e312_snapped=0.6), fr)
        ok = seed == 0 or counts1
        control = dict(card="control.json", numbers=toy_numbers([dict(x=0.6)]), plain_check=dict(max_abs_diff=0.0, where="", tolerance="-", ok=True), repro=dict(max_abs_diff=0.0, where="", ok=ok), counts=sc["counts"]["e312_snapped"] if ok else None)
        return dict(seed=seed, run=f"e324_s_s{seed}", sc=sc, variants=variants, control=control)
    pats = [(J, "EXTRA_WRITABLE", (d,)), (J, "frames_of", lambda split: frames), (J, "precheck", lambda *x: (None, None)), (J, "score_seed", score), (J, "out_dir", lambda st, split, n=0: d), (J, "guard_e321", guards.append), (QE, "graph_numbers", toy_numbers),
            (J, "code_sha256", lambda entry=None: next(code_iter))]
    if check:
        pats.append((J, "check_stored", check))
    with patched(*pats):
        old, sys.stdout = sys.stdout, buf
        try:
            rc = J.cmd_all(a)
        except SystemExit as e:
            rc = e
        finally:
            sys.stdout = old
    return rc, buf.getvalue(), sorted(os.listdir(d)), guards, scored


def test_cmd_all_end_to_end_on_toy_data():
    """`cmd_all`: both seeds scored, the intervals dropped on BOTH seeds when one seed's E312 counts did not reproduce, cards and results.md written with the device and the code hashes, E321's hashes checked before and after, the label printed;
    a changed code file is named, the rehearsal branch compares with the stored cards (exit 1 on a miss, the hint when only arm k fails), existing results are not overwritten, a failed precheck scores nothing"""
    a = argparse.Namespace(split="val", max=873, run_prefix="e324_s_s", seeds=[0, 1], boot=20, no_stack=False, device="cpu", b_dump=None, control_ends_dump=None, control_card=None, test_id="", expect_code_digest=None, expect_code_card=None)
    with tempfile.TemporaryDirectory() as d:
        rc, text, files, guards, scored = toy_cmd_all(d, a)
        card0, md = json.load(open(os.path.join(d, "card_s0.json"))), open(os.path.join(d, "results.md")).read()
        assert rc == 0 and files == ["card_s0.json", "card_s1.json", "results.md"] and guards == ["before the run", "after the run"] and "paired" in card0["compare"]["vs_e312"]["w"] and scored == [0, 1]
        assert "**FULL RETENTION**" in md and "READING: FULL RETENTION" in text and "CODE DRIFT" not in text, text[-300:]
        assert f"code digest: {J.code_digest({'a.py': '1'})} (1 files)" in text.splitlines()[0], "the digest is printed at the start of every run"
        assert card0["device"] == "cpu" and card0["code_sha256"] == {"a.py": "1"} and card0["code_digest"] == J.code_digest({"a.py": "1"}) and card0["code_drift"] == []
        assert "Device of the dumps and the scoring: cpu" in md and "no change of those files during the run" in md
    with tempfile.TemporaryDirectory() as d:                                                           # seed 1 did not reproduce: no interval on seed 0 either
        rc, text, files, guards, scored = toy_cmd_all(d, a, counts1=False)
        card0, md = json.load(open(os.path.join(d, "card_s0.json"))), open(os.path.join(d, "results.md")).read()
        assert "paired" not in card0["compare"]["vs_e312"]["w"] and "**FULL RETENTION (without interval)**" in md and "NO interval against E312 on any seed" in text
    with tempfile.TemporaryDirectory() as d:                                                           # a lane code file changed during the run: named, not fatal
        rc, text, files, guards, scored = toy_cmd_all(d, a, codes=[{"a.py": "1", "b.py": "1"}, {"a.py": "2", "b.py": "1"}])
        card0, md = json.load(open(os.path.join(d, "card_s0.json"))), open(os.path.join(d, "results.md")).read()
        assert rc == 0 and "CODE DRIFT: a.py" in text and card0["code_drift"] == ["a.py"] and "CODE DRIFT during the run: a.py" in md
    rehearsal = argparse.Namespace(**dict(vars(a), max=0))
    for verdict, rc_want, shown, hint in (((False, ["CHECK x"], ["snapped"]), 1, "REHEARSAL CHECK: FAIL", False), ((False, [], ["stack"]), 1, "REHEARSAL CHECK: FAIL", True), ((True, [], []), 0, "REHEARSAL CHECK: PASS", False)):
        with tempfile.TemporaryDirectory() as d:
            rc, text, files, guards, scored = toy_cmd_all(d, rehearsal, check=lambda *x, **k: verdict)
            card0 = json.load(open(os.path.join(d, "card_s0.json")))
            assert rc == rc_want and shown in text and ("only arm k fails" in text) == hint and card0["rehearsal_check"]["ok"] == verdict[0] and files == ["card_s0.json", "card_s1.json", "results.md"], (verdict, rc, text[-200:])
        with tempfile.TemporaryDirectory() as d:                                                       # the rehearsal rule is not applied to a smoke run
            rc, text, *_ = toy_cmd_all(d, a, check=lambda *x, **k: (False, [], ["snapped"]))
            assert rc == 0 and "REHEARSAL CHECK" not in text
    with tempfile.TemporaryDirectory() as d:                                                           # results are written once: nothing scored, the existing file untouched
        open(os.path.join(d, "card_s0.json"), "w").write("keep")
        rc, text, files, guards, scored = toy_cmd_all(d, rehearsal, check=lambda *x, **k: (True, [], []))
        assert isinstance(rc, SystemExit) and scored == [] and guards == [] and open(os.path.join(d, "card_s0.json")).read() == "keep", "refused before any work"
        rc, text, files, guards, scored = toy_cmd_all(d, a)                                           # a smoke run may overwrite
        assert rc == 0 and json.load(open(os.path.join(d, "card_s0.json")))["seed"] == 0
    with tempfile.TemporaryDirectory() as d:                                                           # a failed precheck: refused before the frames are read and before anything is scored
        with patched((J, "frames_of", boom), (J, "score_seed", boom), (J, "guard_e321", lambda w: None), (J, "out_dir", lambda st, split, n=0: d), (J, "precheck", lambda *x: sys.exit("precheck refused"))):
            assert raises(SystemExit, J.cmd_all, a)


# ------------------------------------------------------------------------------------------------------------------------------ the stored cards
def test_expect_code_digest_refuses_other_code_before_any_work():
    """`--expect-code-digest` (the digest of the rehearsal's cards, or its first 16 characters): other code is refused before the E321 hashes, the precheck and any dump; `--expect-code-card` names the changed files"""
    base = argparse.Namespace(split="val", max=873, run_prefix="e324_s_s", seeds=[0, 1], boot=20, no_stack=False, device="cpu", b_dump=None, control_ends_dump=None, control_card=None, test_id="", expect_code_digest=None, expect_code_card=None)
    files = {"a.py": "1", "b.py": "1"}
    digest = J.code_digest(files)
    with_ = lambda **k: argparse.Namespace(**dict(vars(base), **k))  # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        rc, text, _, guards, scored = toy_cmd_all(d, with_(expect_code_digest=digest), codes=[files, files])
        assert rc == 0 and scored == [0, 1] and text.startswith(f"code digest: {digest} (2 files)"), text[:200]
        rc, text, _, guards, scored = toy_cmd_all(d, with_(expect_code_digest=digest[:16]), codes=[files, files])
        assert rc == 0, "the first 16 characters (as results.md prints them) are enough"
    for bad, card_files, names in ((digest[:12], None, None), ("0" * 64, None, None), (digest[:16] + "0", None, None), ("0" * 64, {"a.py": "1", "b.py": "0", "c.py": "5"}, ["b.py", "c.py"])):
        with tempfile.TemporaryDirectory() as d:
            kw = dict(expect_code_digest=bad)
            if card_files:
                json.dump(dict(code_sha256=card_files), open(os.path.join(d, "rehearsal_card.json"), "w"))
                kw["expect_code_card"] = os.path.join(d, "rehearsal_card.json")
            rc, text, out_files, guards, scored = toy_cmd_all(d, with_(**kw), codes=[files, files])
            assert isinstance(rc, SystemExit) and scored == [] and guards == [], "refused before the E321 check, the precheck and any dump"
            msg = str(rc)
            assert (f"changed files against {kw['expect_code_card']}: " + ", ".join(names) in msg) if names else ("--expect-code-card" in msg), msg
    with tempfile.TemporaryDirectory() as d:                                                           # a card whose file hashes are the current ones: the digest given is not that card's
        json.dump(dict(code_sha256=files), open(os.path.join(d, "same.json"), "w"))
        rc, *_ = toy_cmd_all(d, with_(expect_code_digest="0" * 64, expect_code_card=os.path.join(d, "same.json")), codes=[files, files])
        assert isinstance(rc, SystemExit) and "equal the current ones" in str(rc), str(rc)


def test_variants_are_the_stored_readouts():
    """the options of every variant equal the options of the stored VAL card it reproduces (`score --snap`, `--snap --snap-nogeom`, `--attach --snap --snap-nogeom`): the stored cards' spec fields are checked in `verify` when the file has them"""
    v = {x["name"]: x for x in J.variants_of(J.SETUPS["e324_s_s"], True, True)}
    assert (v["snapped"]["snap"], v["snapped"]["nogeom"], v["snapped"]["attach"]) == (True, False, False)
    assert (v["snap_nogeom"]["snap"], v["snap_nogeom"]["nogeom"], v["snap_nogeom"]["attach"]) == (True, True, False)
    assert (v["primary"]["snap"], v["primary"]["nogeom"], v["primary"]["attach"]) == (True, True, True)
    assert (v["stack"]["graph"], v["stack"]["snap"], v["stack"]["base"]) == ("c2", True, "c2_plain") and v["e312_snapped"]["ends"] == "ctl" and v["e312_snapped"]["graph"] == "ctl"
    e = {x["name"]: x for x in J.variants_of(J.SETUPS["e312_p_s"])}
    assert [x["man"] for x in e.values()] == ["plain", "p0.1", "p0.1s"] and (e["unsnapped"]["snap"], e["snapped"]["snap"]) == (False, True) and not any(x["attach"] or x["nogeom"] for x in e.values()), "E312's variants stay as they were"
    assert J.SETUPS["e324_s_s"]["stored_counts"] == ("snapped", "snap_nogeom", "primary", "stack") and J.SETUPS["e312_p_s"]["stored_counts"] == ("snapped",), "E324 s's four stored cards all have a counts file; E312's unsnapped card has none"


def toy_stored(d, name, old):
    os.makedirs(os.path.join(d, "runs", "r0"), exist_ok=True)
    json.dump(old, open(os.path.join(d, "runs", "r0", name), "w"))


def test_check_stored_reads_the_stored_cards():
    """`check_stored` on a toy stored card in a temporary run folder: equal passes, a number off by 1e-6 fails EXACT but passes the rehearsal tolerance, curves must be equal under EXACT, a stored spec field that differs from the variant's fails"""
    setup = dict(J.SETUPS["e312_p_s"], stored={"snapped": "ends_card_snap_thr0.1.json"}, stored_counts=(), stored_man=None)
    num = lambda x: {k: x for k in J.KEYS}  # noqa: E731
    new = dict(snapped=dict(base="plain", curves=100, wired_both_ends=60, numbers=dict(base=num(0.5), head=num(0.6)), difference=num(0.1)))
    card = dict(variants=new, maneuver={"plain": dict(recall=0.5)})
    old = dict(curves=100, wired_both_ends=60, numbers=dict(base=num(0.5), head=num(0.6)), difference=num(0.1), thr=0.1, spec="close:1.5@0.5", snap=True, attach=False, attach_r=None, snap_options=[], wire=None)
    check = lambda tol, frames=(): J.check_stored(setup, "r0", 0, card, {"counts": {}}, tol, list(frames))  # noqa: E731
    with tempfile.TemporaryDirectory() as d, patched((J, "ROOT", d)):
        write = lambda o: toy_stored(d, "ends_card_snap_thr0.1.json", o)  # noqa: E731
        write(old)
        assert check(J.EXACT)[0] and check(J.EXACT)[2] == []
        write(dict(old, numbers=dict(base=num(0.5), head=num(0.6 + 1e-6))))
        assert not check(J.EXACT)[0] and check(J.EXACT)[2] == ["snapped"] and check(J.REHEARSAL)[0]
        write(dict(old, curves=11378, wired_both_ends=5968))
        card["variants"]["snapped"].update(curves=11379, wired_both_ends=5969)                                       # E321's fresh CPU dump: 11379 against 11378, 5969 against 5968 (relative 9e-5, 2e-4)
        assert not check(J.EXACT)[0] and check(J.REHEARSAL)[0], "integer counts: equal under EXACT, relative 5e-4 in the rehearsal"
        write(dict(old, curves=11300, wired_both_ends=5968))
        assert not check(J.REHEARSAL)[0], "11379 against 11300 curves is outside the relative 5e-4"
        card["variants"]["snapped"].update(curves=100, wired_both_ends=60)
        write(dict(old, snap=False))
        assert not check(J.REHEARSAL)[0], "the stored spec says unsnapped, the variant is snapped"


def test_check_stored_per_frame_counts():
    """the per-frame counts of the stored `.counts.pkl`: equal passes; one frame differing in the head or in the plain graph fails; other scenes fail; a card of `stored_counts` without its file fails, one that has no file by design does not"""
    setup = dict(J.SETUPS["e324_s_s"], stored={"snapped": "ends_card_snap_thr0.1.json"}, stored_counts=("snapped",), stored_man=None)
    num = lambda x: {k: x for k in J.KEYS}  # noqa: E731
    cnt = lambda v: dict(rec=dict(a=float(v)), dp=np.array([v, 2.0 * v]))  # noqa: E731
    head, base = [cnt(i + 1) for i in range(4)], [cnt(10 + i) for i in range(4)]
    frames = [Frame(i, i // 2) for i in range(4)]
    card = dict(variants=dict(snapped=dict(base="plain", curves=100, wired_both_ends=60, numbers=dict(base=num(0.5), head=num(0.6)), difference=num(0.1))), maneuver={"plain": dict(recall=0.5)})
    old = dict(curves=100, wired_both_ends=60, numbers=dict(base=num(0.5), head=num(0.6)), difference=num(0.1), thr=0.1, spec="close:1.5@0.5", snap=True, attach=False, attach_r=None, snap_options=[], wire=None, full="temp/e276/predfull_e273_p_s0.pkl")
    check = lambda h=head, b=base, fr=frames, st=setup: J.check_stored(st, "r0", 0, card, {"counts": {"snapped": h, "plain": b}}, J.EXACT, fr)  # noqa: E731
    with tempfile.TemporaryDirectory() as d, patched((J, "ROOT", d)):
        toy_stored(d, "ends_card_snap_thr0.1.json", old)
        cpath = os.path.join(d, "runs", "r0", "ends_card_snap_thr0.1.counts.pkl")
        pickle.dump(dict(scenes=[0, 0, 1, 1], base=base, head=head), open(cpath, "wb"))
        ok, lines, failed = check()
        assert ok and failed == [] and "scenes equal" in lines[0], lines
        assert not check(h=head[:2] + [cnt(99)] + head[3:])[0] and check(h=head[:2] + [cnt(99)] + head[3:])[2] == ["snapped"], "one frame of the head differs"
        assert not check(b=base[:3] + [cnt(0)])[0], "one frame of the plain graph differs"
        assert not check(h=head[:3])[0], "fewer frames"
        assert not check(fr=[Frame(i, i // 2 + (i == 3)) for i in range(4)])[0], "the scenes differ"
        os.remove(cpath)
        ok, lines, failed = check()
        assert not ok and failed == ["snapped"] and "MISSING" in lines[0], "a stored card of the setup's list without its counts file fails"
        assert check(st=dict(setup, stored_counts=()))[0], "a card without a counts file by design (E312's unsnapped) passes"


def test_no_stack_checks_w_and_p_alone():
    """without `--no-stack` the stack's own check failing makes the exit non-zero and names arm k only; with it the card has no stack and w and p are checked alone"""
    setup = dict(J.SETUPS["e324_s_s"], stored={"snapped": "ends_card_snap_thr0.1.json", "primary": "ends_card_snap_nogeom_attach_thr0.1.json", "stack": "ends_card_e338_snap_thr0.1.json"}, stored_counts=(), stored_man=None)
    num = lambda x: {k: x for k in J.KEYS}  # noqa: E731
    var = lambda head: dict(base="plain", curves=100, wired_both_ends=60, numbers=dict(base=num(0.5), head=num(head)), difference=num(head - 0.5))  # noqa: E731
    old = lambda head, **k: dict(var(head), thr=0.1, spec="close:1.5@0.5", attach_r=None, wire=None, **k)  # noqa: E731
    with tempfile.TemporaryDirectory() as d, patched((J, "ROOT", d)):
        toy_stored(d, "ends_card_snap_thr0.1.json", old(0.6, snap=True, attach=False, snap_options=[]))
        toy_stored(d, "ends_card_snap_nogeom_attach_thr0.1.json", old(0.6, snap=True, attach=True, snap_options=["snap_nogeom"]))
        toy_stored(d, "ends_card_e338_snap_thr0.1.json", old(0.7, snap=True, attach=False, snap_options=[]))
        both = lambda w=0.6, p=0.6, k=0.6: dict(variants=dict(snapped=var(w), primary=var(p), stack=var(k)), maneuver={"plain": dict(recall=0.5)})  # noqa: E731
        run = lambda card: J.check_stored(setup, "r0", 0, card, {"counts": {}}, J.REHEARSAL)  # noqa: E731
        ok, lines, failed = run(both(k=0.6))
        assert not ok and failed == ["stack"] and J.arm_k_only(failed), "only arm k fails: the card's rule applies, run again with --no-stack"
        no_stack = both(k=0.6)
        del no_stack["variants"]["stack"]
        ok, lines, failed = run(no_stack)
        assert ok and failed == [] and not any("stack" in ln for ln in lines), "--no-stack: w and p are checked alone, the stack's stored card is not looked at"
        no_stack["variants"]["snapped"] = var(0.65)
        ok, lines, failed = run(no_stack)
        assert not ok and failed == ["snapped"] and not J.arm_k_only(failed), "w failing is not a k-only failure"
        ok, lines, failed = run(both(w=0.65, k=0.6))
        assert failed == ["snapped", "stack"] and not J.arm_k_only(failed) and not J.arm_k_only([])
        assert run(both(k=0.7))[0], "arm k reproducing passes"


def main():
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"test_jtest_eval: {n} tests passed")


if __name__ == "__main__":
    main()
