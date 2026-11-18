import re, os, sys
ROOT = os.environ["BFLC_ROOT"]
src = open(os.path.join(ROOT, "evaluation", "jtoken_eval.py"), encoding="utf-8").read()
names = re.findall(r"^(?:def )?([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|=)", src, re.M)
names = [n for n in dict.fromkeys(names) if n not in ("if", "for", "import", "from")]
files = []
for d in ("evaluation", "training", "tools"):
    for dp, dn, fn in os.walk(os.path.join(ROOT, d)):
        if "__pycache__" in dp: continue
        for f in fn:
            if f.endswith(".py"): files.append(os.path.join(dp, f))
texts = {f: open(f, encoding="utf-8", errors="ignore").read() for f in files}
selfpath = os.path.join(ROOT, "evaluation", "jtoken_eval.py")
res = []
for n in names:
    pat = re.compile(r"\b" + re.escape(n) + r"\b")
    own = len(pat.findall(src))
    ext = sum(len(pat.findall(t)) for f, t in texts.items() if f != selfpath and "tests" + os.sep + "test_jtoken_eval" not in f)
    tst = sum(len(pat.findall(t)) for f, t in texts.items() if "test_jtoken_eval" in f)
    if own <= 1 and ext == 0:
        res.append((n, own, ext, tst))
print("defined, never used elsewhere (own refs <=1 = the definition only), outside tests:", [(n, own, ext, tst) for n, own, ext, tst in res])
