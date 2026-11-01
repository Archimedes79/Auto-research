p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()
old = '''            put(s3, "model.py", "import autogt.missing\nimport numpy\n")
            assert bundle(s3)["repo_files"] == ["autogt/__init__.py"], bundle(s3)
            assert "autogt" not in bundle(s3)["external_local"] or True
'''
new = '''            put(s3, "model.py", "import autogt.missing\nimport numpy\n")
            put(s3, "losses.py", "Y = 1\n")
            b4 = bundle(s3)
            assert b4["repo_files"] == ["autogt/__init__.py"] and b4["external_local"] == [], b4   # a name that is no module still finds its package; numpy is not repository code
'''
assert s.count(old) == 1, s.count(old)
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("ok")
