p = "autoresearch/tools/commit_paths.py"
s = open(p, encoding="utf-8").read()
def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:50], s.count(old))
    s = s.replace(old, new)
sub("    saved = dict(os.environ)\n    try:\n        for v in", "    saved, cwd0 = dict(os.environ), os.getcwd()\n    try:\n        for v in")
sub("        top = os.path.join(tmp, \"r\")\n        os.makedirs(top)\n\n        def g", "        top = os.path.join(tmp, \"r\")\n        os.makedirs(top)\n        os.chdir(top)                                              # paths are relative to the folder you run it in\n\n        def g")
sub("    finally:\n        os.environ.clear()\n        os.environ.update(saved)\n        shutil.rmtree(tmp, ignore_errors=True)", "    finally:\n        os.chdir(cwd0)\n        os.environ.clear()\n        os.environ.update(saved)\n        shutil.rmtree(tmp, ignore_errors=True)")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
