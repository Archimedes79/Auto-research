import re, os, subprocess
def added(path):
    d = subprocess.run(["git", "diff", "-U0", "--", path], capture_output=True, text=True, encoding="utf-8").stdout
    return "\n".join(l[1:] for l in d.splitlines() if l.startswith("+") and not l.startswith("+++"))
for path in ("README.md", "bev_fusion/README.md"):
    base = os.path.dirname(path) or "."
    bad = []
    for tok in set(re.findall(r"`([^`\s]+)`", added(path))) | set(re.findall(r"\]\(([^)]+)\)", added(path))):
        t = tok.split("#")[0].rstrip(",.;:)")
        if "/" not in t or t.startswith(("http", "-", "$")) or any(c in t for c in "<>*|") or t.startswith("python"):
            continue
        if not (os.path.exists(os.path.join(base, t)) or os.path.exists(t)):
            bad.append(t)
    print(path, "missing:", sorted(bad))
