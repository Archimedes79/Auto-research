import os, sys
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtest_eval as J
files = J.code_sha256()
print(len(files))
for k in files:
    if any(t in k for t in ("jwire", "qends", "jends", "jtest", "jmaps", "jwiring")):
        print(k)
