p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()
old = '''        have = (git(dest, "rev-parse", "HEAD") or "").strip()'''
new = '''        have = (git(dest, "rev-parse", "HEAD") or "").strip() if os.path.exists(os.path.join(dest, ".git")) else ""   # a folder inside the repository without its own .git would answer with the repository's HEAD'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("ok")
