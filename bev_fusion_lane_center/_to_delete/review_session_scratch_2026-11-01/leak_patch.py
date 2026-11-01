p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()
old = '''            leaked = [d for d in os.listdir(tempfile.gettempdir()) if d.startswith("tmp") and os.path.isdir(os.path.join(tempfile.gettempdir(), d, "autogt"))]
            assert not leaked, "the temp folder of the tag's files is removed"'''
new = '''            before = set(os.listdir(tempfile.gettempdir()))
            bundle(s3, into=os.path.join(tmp, "out5"), repo_ref="tg/layers")
            assert set(os.listdir(tempfile.gettempdir())) - before == set(), "the temp folder of the tag's files is removed"'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("ok")
