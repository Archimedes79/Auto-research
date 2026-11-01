"""top-level functions / classes of live code whose name occurs only once (the def) in the whole repository (tracked + untracked text, without history/ and _to_delete/)"""
import ast, os, re, subprocess, collections
R = os.environ["BFLC_ROOT"]; REPO = os.path.dirname(R)
files = subprocess.run(["git", "-c", "safe.directory=*", "-C", REPO, "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True).stdout.split("\n")
files = [f for f in files if f and not re.search(r"/(history|_to_delete)/", f) and f.endswith((".py", ".sh", ".md", ".json", ".ps1")) and os.path.isfile(os.path.join(REPO, f))]
cnt = collections.Counter()
texts = {}
for f in files:
    try:
        t = open(os.path.join(REPO, f), encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    texts[f] = t
    cnt.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", t))
live = [f for f in files if f.startswith("bev_fusion_lane_center/") and f.endswith(".py") and re.match(r"bev_fusion_lane_center/(training|evaluation|tools|reports/build)/", f) and "/legacy/" not in f and "/loops/" not in f]
for f in sorted(live):
    try:
        tree = ast.parse(texts[f])
    except Exception as e:
        print("PARSE", f, e); continue
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and cnt[node.name] <= 1 and not node.name.startswith("test_") and node.name != "main":
            print(f"{f}:{node.lineno} {node.name}")
