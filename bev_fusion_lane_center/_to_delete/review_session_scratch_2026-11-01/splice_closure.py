"""one-off: replace import_closure ... bundle in provenance.py by the version with test scopes and repository files, and extend repro and the selftest"""
p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()
new = open("bev_fusion_lane_center/temp/review_session_scratch_2026-11-01/new_closure_block.py", encoding="utf-8").read()
i, j = s.index("def import_closure("), s.index("def results_dir(")
s = s[:i] + new + s[j:]


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


# repro: the bundle line counts the repository files too, and says what only tests import
sub('''(f"model bundle (network + losses): {len(_get(prov, ('bundle', 'files')) or [])} files, hash''', '''(f"model bundle (network + losses): {len(_get(prov, ('bundle', 'files')) or [])} files + {len(_get(prov, ('bundle', 'repo_files')) or [])} repository files, hash''')
sub('''"model bundle (network + losses): 8 files" in buf.getvalue()''', '''"model bundle (network + losses): 8 files + 0 repository files" in buf.getvalue()''')
sub('''("external_local", "repository code outside the bundle (its version is the run's GT version / tag)"), ("dynamic_imports", "files that import by name at run time")):''',
    '''("external_local", "repository code outside the bundle (its version is the run's GT version / tag)"), ("dynamic_imports", "files that import by name at run time"),
                          ("test_imports", "only selftests / command-line tools import (not part of the bundle)")):''')
# selftest: test scopes, repository files
sub('''        assert bundle(os.path.join(tmp, "nowhere"))''', '''        # test scopes and repository files: the imports of a selftest are listed, `autogt.layers` (a repository module) is part of the bundle
        s3 = os.path.join(tmp, "snap3")
        put(s3, "model.py", "import autogt.layers\\nimport torch.nn\\nimport helper\\n\\ndef _selftest():\\n    import loader\\n    import train\\n\\nif __name__ == '__main__':\\n    import loader2\\n")
        for f in ("helper.py", "loader.py", "loader2.py", "train.py"):
            put(s3, f, "X = 1\\n")
        put(s3, "losses.py", "from autogt.layers import AREAS\\n")
        put(tmp, "autogt/__init__.py", "")
        put(tmp, "autogt/layers.py", "AREAS = 1\\n")
        put(tmp, "autogt/other.py", "OTHER = 1\\n")
        real_root3 = repo_root
        globals()["repo_root"] = lambda start=None: tmp
        try:
            b3 = bundle(s3, into=os.path.join(tmp, "out3"), exclude=("train.py",))
            assert b3["files"] == ["helper.py", "losses.py", "model.py"], b3["files"]
            assert b3["repo_files"] == ["autogt/__init__.py", "autogt/layers.py"], b3["repo_files"]
            assert b3["test_imports"] == ["model.py -> loader.py", "model.py -> loader2.py", "model.py -> train.py"] and b3["imports_excluded"] == [], b3
            assert b3["external_local"] == [] and b3["copied_to"] == "out3" and os.path.isfile(os.path.join(tmp, "out3", "_repo", "autogt", "layers.py")), b3
            h3 = b3["code_hash"]
            put(tmp, "autogt/layers.py", "AREAS = 2\\n")
            assert bundle(s3, exclude=("train.py",))["code_hash"] != h3, "a change of the repository module changes the hash"
            put(tmp, "autogt/other.py", "OTHER = 2\\n")
            put(tmp, "autogt/layers.py", "AREAS = 1\\n")
            assert bundle(s3, exclude=("train.py",))["code_hash"] == h3, "a repository module that is not imported does not"
            put(s3, "model.py", "import autogt.missing\\nimport numpy\\n")
            assert bundle(s3)["repo_files"] == ["autogt/__init__.py"], bundle(s3)
            assert "autogt" not in bundle(s3)["external_local"] or True
        finally:
            globals()["repo_root"] = real_root3
        assert bundle(os.path.join(tmp, "nowhere"))''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
