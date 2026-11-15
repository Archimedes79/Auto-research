"""scratch (E343 code step, round 3): tests for the re-review's findings"""
import os

P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "tests", "test_jtest_eval.py")
s = open(P, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''import io
import json
import os
import pickle
import sys
import tempfile
''', '''import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
''')

# ---------------------------------------------------------------------------------------------------- the allow-list
rep('''        J.save_json(os.path.join(d, "temp", "e343", "test", "new.json"), {"a": 1}, overwrite=False)
''', '''        J.save_json(os.path.join(d, "temp", "e343", "test", "new.json"), {"a": 1}, overwrite=False)
        # the allow-list: only temp/e343 and runs/eval/e324s_val, e324s_test, e312_val are written (under the root); an NTFS short name of E321's folders, another folder or a place outside the root is refused, the deny-list or not
        for rel in ("runs/eval/E312_T~1/never.json", "temp/E321~1/test/never.pkl", "runs/eval/e312_test2/never.json", "runs/eval/other/never.json", "temp/other/never.txt", "runs/eval/e324s_train/never.json", "never.txt"):
            assert raises(PermissionError, J.save_text, os.path.join(d, rel), "x"), rel
        assert raises(PermissionError, J.save_text, os.path.join(os.path.dirname(d), "outside_the_root.txt"), "x"), "a path outside the root"
        assert not os.path.exists(os.path.join(d, "runs", "eval", "E312_T~1")) and not os.path.exists(os.path.join(d, "runs", "eval", "other")) and not os.path.exists(os.path.join(os.path.dirname(d), "outside_the_root.txt"))
        for rel in ("runs/eval/e324s_test/card_s0.json", "runs/eval/e324s_val/results.md", "runs/eval/e312_val/card_s0.json", "temp/e343/val_max6/cards_e312/card_s0.json"):
            J.save_text(os.path.join(d, rel), "x")
            assert os.path.exists(os.path.join(d, rel)), rel
        with patched((J, "EXTRA_WRITABLE", (os.path.join(os.path.dirname(d), "extra"),))):
            J.save_text(os.path.join(os.path.dirname(d), "extra", "ok.txt"), "x")                      # the tests' own temporary folders
            assert raises(PermissionError, J.save_text, os.path.join(os.path.dirname(d), "extra2", "no.txt"), "x")
''')

# ---------------------------------------------------------------------------------------------------- provenance test: writes to a temporary folder
rep('''    with tempfile.TemporaryDirectory() as d:
        p, calls = os.path.join(d, "ends_r.pkl"), []''', '''    with tempfile.TemporaryDirectory() as d, patched((J, "EXTRA_WRITABLE", (d,))):
        p, calls = os.path.join(d, "ends_r.pkl"), []''')

# ---------------------------------------------------------------------------------------------------- the code digest test: the resolver of the process, subfolders, lazy imports, the scratch folders
a = s.index("def test_code_sha256_follows_the_imports():")
b = s.index("# ------------------------------------------------------------------------------------------------------------------------------ the reading rule")
s = s[:a] + '''def test_code_sha256_follows_the_imports():
    """the closure is looked up in every folder of the project's search path (what bflc_env.sh puts on PYTHONPATH: subfolders such as evaluation/sparse_lanes, training/models), also through imports inside functions of the lane's code (not of the shared library autoresearch/, whose
    lazy imports run only when a function is called); a name in two folders counts for both; the scratch folders, the gate's `evaluate`, the GT package and the library are not followed; a file that changes changes the digest"""
    files = {"proj/evaluation/entry.py": "import helper\\nimport qgraph_eval as QE\\nimport dup\\nimport scratch_mod\\nfrom Evaluation import stats as ST\\nimport numpy\\n\\n\\ndef f():\\n    import lazy\\n    import evaluate\\n    from GtGenerator import x\\n",
             "proj/evaluation/qgraph_eval.py": "import topo_metrics as TM\\n\\n\\ndef g():\\n    import inner_lazy\\n", "proj/evaluation/sparse_lanes/topo_metrics.py": "", "proj/training/models/inner_lazy.py": "",
             "proj/training/grid/helper.py": "from . import sub\\nimport pkg.mod\\n", "proj/training/grid/sub.py": "", "proj/training/grid/lazy.py": "def (:\\n", "proj/training/grid/pkg/__init__.py": "", "proj/training/grid/pkg/mod.py": "",
             "proj/training/grid/dup.py": "", "proj/evaluation/lane_task/dup.py": "", "proj/temp/scratch_mod.py": "", "proj/training/grid/_session_scratch_/scratch_mod.py": "",
             "proj/evaluation/evaluate.py": "import never\\n", "proj/training/grid/never.py": "", "autoresearch/Evaluation/__init__.py": "def __getattr__(name):\\n    from . import lazy_api\\n", "autoresearch/Evaluation/lazy_api.py": "", "autoresearch/Evaluation/stats.py": "", "GtGenerator/x.py": "", "proj/training/grid/other.py": ""}
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
            open(os.path.join(t, "proj/training/grid/sub.py"), "w").write("x = 1\\n")
            again = J.code_sha256(entry=entry)
            assert [k for k in got if got[k] != again[k]] == ["proj/training/grid/sub.py"] and J.code_digest(got) != J.code_digest(again)
            open(os.path.join(t, "proj/evaluation/sparse_lanes/topo_metrics.py"), "w").write("x = 2\\n")      # the scorer's own helper module, imported by `import topo_metrics as TM` in qgraph_eval: the digest follows it
            third = J.code_sha256(entry=entry)
            assert third["proj/evaluation/sparse_lanes/topo_metrics.py"] != again["proj/evaluation/sparse_lanes/topo_metrics.py"] and J.code_digest(third) != J.code_digest(again)


def test_the_code_digest_covers_every_module_a_run_loads():
    """a subprocess imports what this module imports (the lazy imports of a VAL run included, CUDA hidden) and lists the repository modules it loaded: every one is in the digest of the real tree, and the digest names the helper modules of the scorer
    (evaluation/sparse_lanes/topo_metrics.py ...) that a fixed folder list would have missed; library-only helpers of the suite that nothing imports are not in it"""
    script = (
        "import ast, importlib, json, os, sys\\n"
        "import jtest_eval as J\\n"
        "tree = ast.parse(open(J.__file__).read())\\n"
        "names = sorted({a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and not n.level})\\n"
        "for name in names:\\n"
        "    if name.split('.')[0] not in J.CODE_SKIP:\\n"
        "        importlib.import_module(name)\\n"
        "repo = os.path.realpath(os.path.join(J.ROOT, '..'))\\n"
        "rel = lambda p: os.path.relpath(os.path.realpath(p), repo).replace(os.sep, '/')\\n"
        "print(json.dumps(sorted({rel(m.__file__) for m in list(sys.modules.values()) if getattr(m, '__file__', None) and os.path.realpath(m.__file__).startswith(repo + os.sep) and '/temp/' not in rel(m.__file__)})))\\n")
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), cwd=os.path.join(J.ROOT, "evaluation"))
    assert r.returncode == 0, r.stderr[-800:]
    loaded = json.loads(r.stdout.strip().splitlines()[-1])
    digest = J.code_sha256(entry=os.path.join(J.ROOT, "evaluation", "jtest_eval.py"))
    assert loaded and not [x for x in loaded if x not in digest], [x for x in loaded if x not in digest]
    assert any(x.endswith("evaluation/sparse_lanes/topo_metrics.py") for x in loaded) and any(x.endswith("evaluation/sparse_lanes/topo_metrics.py") for x in digest), "the scorer's helper modules are loaded and in the digest"
    assert not [x for x in digest if x.endswith(("autoresearch/Evaluation/render/video.py", "autoresearch/Evaluation/api.py", "autoresearch/suitecli.py", "evaluation/evaluate.py"))], "files that nothing imports at import time (and the gate's evaluator) are not in it"


''' + s[b:]

# ---------------------------------------------------------------------------------------------------- precheck: flagged dumps with an old-style head, provenance refused before any dump
rep('''def test_precheck_refuses_before_any_dump_is_made():
    ns = lambda **k: argparse.Namespace(**dict(dict(split="val", max=0, run_prefix="e324_s_s", b_dump=None, control_ends_dump=None, control_card=None), **k))  # noqa: E731
    setup = J.SETUPS["e324_s_s"]
    with tempfile.TemporaryDirectory() as d, patched((J, "pyramid_same", lambda *x: True)):''', '''def test_precheck_refuses_before_any_dump_is_made():
    ns = lambda **k: argparse.Namespace(**dict(dict(split="val", max=0, device="cpu", run_prefix="e324_s_s", b_dump=None, control_ends_dump=None, control_card=None), **k))  # noqa: E731
    setup = J.SETUPS["e324_s_s"]
    with tempfile.TemporaryDirectory() as d, patched((J, "pyramid_same", lambda *x: True), (J, "dump_dir", lambda split, n=0: d), (J, "provenance", PROV)):''')
rep('''def test_card_arms_and_results_md():''', '''OLD_HEAD = dict(ent_recall=0.7, ent_precision=0.7, ext_recall=0.7, ext_precision=0.7, pair_recall=0.4, pair_precision=0.7, counts={}, thr=0.5, frames=833, run="e312_p_s0")      # the `_head.json` that E321's old dump_ends wrote: no provenance key


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
        assert J.precheck(a, setup, 0, False)[0]["ctl_ends"] == p, "the R1 line of the TEST command: E321's pair dump is not refused"
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
    with tempfile.TemporaryDirectory() as d, patched((J, "provenance", lambda run, device: seen.append((run, device)) or PROV(run, device)), (J, "frames_of", stop)):
        assert raises(Stop, J.dump_b, "val", "e273_p_s0", os.path.join(d, "b.pkl"), "cpu") and seen == [("e273_p_s0", "cpu")], seen
        seen.clear()
        assert raises(Stop, J.dump_ends, "val", "e324_s_s0", os.path.join(d, "e.pkl"), "cpu") and seen == [("e324_s_s0", "cpu")], seen


def test_card_arms_and_results_md():''')

# ---------------------------------------------------------------------------------------------------- cmd_all: toy namespace, writable folder, --expect-code-digest
rep('''a = argparse.Namespace(split="val", max=873, run_prefix="e324_s_s", seeds=[0, 1], boot=20, no_stack=False, device="cpu", b_dump=None, control_ends_dump=None, control_card=None, test_id="")
    with tempfile.TemporaryDirectory() as d:
        rc, text, files, guards, scored = toy_cmd_all(d, a)''', '''a = argparse.Namespace(split="val", max=873, run_prefix="e324_s_s", seeds=[0, 1], boot=20, no_stack=False, device="cpu", b_dump=None, control_ends_dump=None, control_card=None, test_id="", expect_code_digest=None, expect_code_card=None)
    with tempfile.TemporaryDirectory() as d:
        rc, text, files, guards, scored = toy_cmd_all(d, a)''')
rep('''        assert "**FULL RETENTION**" in md and "READING: FULL RETENTION" in text and "CODE DRIFT" not in text, text[-300:]''', '''        assert "**FULL RETENTION**" in md and "READING: FULL RETENTION" in text and "CODE DRIFT" not in text, text[-300:]
        assert f"code digest: {J.code_digest({'a.py': '1'})} (1 files)" in text.splitlines()[0], "the digest is printed at the start of every run"''')
rep('''    pats = [(J, "frames_of", lambda split: frames),''', '''    pats = [(J, "EXTRA_WRITABLE", (d,)), (J, "frames_of", lambda split: frames),''')
rep('''a = argparse.Namespace(split="test", max=0, run_prefix="e324_s_s", seeds=[0, 1], boot=20, no_stack=False, device="cpu", b_dump=None, control_ends_dump=None, control_card=None, test_id="")
    old_env''', '''a = argparse.Namespace(split="test", max=0, run_prefix="e324_s_s", seeds=[0, 1], boot=20, no_stack=False, device="cpu", b_dump=None, control_ends_dump=None, control_card=None, test_id="", expect_code_digest=None, expect_code_card=None)
    old_env''')
rep('''def test_variants_are_the_stored_readouts():''', '''def test_expect_code_digest_refuses_other_code_before_any_work():
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


def test_variants_are_the_stored_readouts():''')

open(P, "w", encoding="utf-8").write(s)
print("ok")
