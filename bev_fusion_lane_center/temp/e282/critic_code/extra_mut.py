import os, shutil, subprocess, sys
sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, "work", "training", "grid")
ORIG = os.path.join(HERE, "orig")
src = open(os.path.join(HERE, "mut.py")).read()
ns = {}
exec(src.split("only = sys.argv[1:]")[0], dict(__file__=os.path.join(HERE, "mut.py")), ns)
M = {m[0][:3]: m for m in ns["M"]}
shutil.copy(os.path.join(HERE, "extra_test.py"), os.path.join(G, "extra_test.py"))
for key in (None, "M09", "M13", "M14", "M15", "M12", "M10", "M05"):
    for fn in ("qzones.py", "qgraph.py"):
        shutil.copy(os.path.join(ORIG, fn), os.path.join(G, fn))
    name = "original"
    if key:
        name, f, old, new = M[key]
        p = os.path.join(G, f); s = open(p).read(); assert old in s; open(p, "w").write(s.replace(old, new, 1))
    r = subprocess.run([sys.executable, "extra_test.py"], cwd=G, capture_output=True, text=True, timeout=300)
    out = (r.stdout + r.stderr).strip().splitlines()
    print(f"{name:45s} {'PASS' if r.returncode == 0 else 'FAIL'}  {out[-1][:140] if out else ''}", flush=True)
for fn in ("qzones.py", "qgraph.py"):
    shutil.copy(os.path.join(ORIG, fn), os.path.join(G, fn))
os.remove(os.path.join(G, "extra_test.py"))
