import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "e343", "_session_scratch_", "patch_r3_tests.py")
s = open(p, encoding="utf-8").read()
pairs = [
    ('"autoresearch/Evaluation/__init__.py": "", "autoresearch/Evaluation/stats.py": "",',
     '"autoresearch/Evaluation/__init__.py": "def __getattr__(name):\\\\n    from . import lazy_api\\\\n", "autoresearch/Evaluation/lazy_api.py": "", "autoresearch/Evaluation/stats.py": "",'),
    ('''    assert not any(x.endswith("autoresearch/Evaluation/render/video.py") for x in digest) and not any(x.endswith("evaluation/evaluate.py") for x in digest), "files that nothing imports (and the gate's evaluator) are not in it"''',
     '''    assert not [x for x in digest if x.endswith(("autoresearch/Evaluation/render/video.py", "autoresearch/Evaluation/api.py", "autoresearch/suitecli.py", "evaluation/evaluate.py"))], "files that nothing imports at import time (and the gate's evaluator) are not in it"'''),
    ('''also through imports inside functions; a name in two folders counts for both;
    the scratch folders, the gate's `evaluate`, the GT package and the library are not followed; a file that changes changes the digest"""''',
     '''also through imports inside functions of the lane's code (not of the shared library autoresearch/, whose
    lazy imports run only when a function is called); a name in two folders counts for both; the scratch folders, the gate's `evaluate`, the GT package and the library are not followed; a file that changes changes the digest"""'''),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("ok")
