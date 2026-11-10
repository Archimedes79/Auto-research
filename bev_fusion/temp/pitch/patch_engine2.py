def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/quantitative.py", [
    ('''        desc = subj.describe()
        run_name = name or''', '''        desc = subj.describe() if subj is not None else dict(results.manifest["identity"].get("network", {}))     # a stored result set: the network as it was described then
        run_name = name or'''),
])
patch("evalsuite/api.py", [
    ('''results=rs, reference=ref_of, progress=progress, ignore={"source": ("evaluator_hash",)} if False else None)''', '''results=rs, reference=ref_of, progress=progress)'''),
    ("import hashlib\nimport importlib\nimport inspect\nimport json", "import hashlib\nimport importlib\nimport json"),
])
print("ok")
