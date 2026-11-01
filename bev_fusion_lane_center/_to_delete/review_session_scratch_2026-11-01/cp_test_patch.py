p = "autoresearch/tools/commit_paths.py"
s = open(p, encoding="utf-8").read()
for old, new in (
    ('(["pic.png"], "forbidden")):', '(["big.txt"], "forbidden")):'),
    ('            w("pic.png", "p\n") if bad == ["pic.png"] else None', '            w("big.txt", "x" * 1_300_000) if bad == ["big.txt"] else None'),
    ("scratch, a binary, a moved HEAD", "scratch, a file over 1 MB, a moved HEAD"),
):
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
