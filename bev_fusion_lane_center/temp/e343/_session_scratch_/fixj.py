import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "e343", "_session_scratch_", "patch_r3j.py")
s = open(p, encoding="utf-8").read()
a = s.index("    ('''            assert isinstance(rc, SystemExit) and scored == [] and guards == [], \"refused before the E321 check")
b = s.index("])\nprint(\"ok\")")
new = """    ('''            assert (f"changed files against {kw['expect_code_card']}: " + ", ".join(names) in msg) if names else ("--expect-code-card" in msg), msg
''', '''            assert (f"changed files against {kw['expect_code_card']}: " + ", ".join(names) in msg) if names else ("--expect-code-card" in msg), msg
    with tempfile.TemporaryDirectory() as d:                                                           # a card whose file hashes are the current ones: the digest given is not that card's
        json.dump(dict(code_sha256=files), open(os.path.join(d, "same.json"), "w"))
        rc, *_ = toy_cmd_all(d, with_(expect_code_digest="0" * 64, expect_code_card=os.path.join(d, "same.json")), codes=[files, files])
        assert isinstance(rc, SystemExit) and "equal the current ones" in str(rc), str(rc)
'''),
"""
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
