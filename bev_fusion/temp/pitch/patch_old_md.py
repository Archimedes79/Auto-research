def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/EXTENSIONS.md", [
    ("**Design history** (the plan, the data flow, the proof plan, the migration of the old scripts): `../history/evalsuite_architecture_2026-10-30.md`.",
     "**Design history** (the plan, the data flow, the proof plan, the migration of the old scripts): git history, `git log -- autoresearch/history/evalsuite_architecture_2026-10-30.md` (the file was removed on 2026-11-10: the README and this page are the description)."),
])
patch("evalsuite/README.md", [
    (" The design notes of the first build are in `../history/evalsuite_architecture_2026-10-30.md`.", " The design notes of the first build are in git history (`autoresearch/history/evalsuite_architecture_2026-10-30.md`, removed 2026-11-10)."),
])
print("ok")
