"""list the CLI commands / options of my area and where each is mentioned outside its own definition (scratch)"""
import re, subprocess, os, collections
files = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
text = {f: open(f, encoding="utf-8", errors="replace").read() for f in files if os.path.isfile(f) and re.search(r"\.(py|md|sh|json)$", f) and "/temp/" not in f and "_to_delete" not in f and "/runs/" not in f and not f.startswith("literature") and "/history/" not in f}
def n_uses(word, exclude):
    c = collections.Counter()
    for f, t in text.items():
        k = len(re.findall(re.escape(word), t))
        if f == exclude: k -= 1
        if k > 0: c[f] = k
    return c
for path in ("bev_fusion/evaluation/bev_suite.py", "bev_fusion/evaluation/evaluate.py", "autoresearch/evalsuite/cli.py", "bev_fusion/evaluation/bev_sets.py", "bev_fusion/evaluation/front_compare.py", "bev_fusion/evaluation/bev_gtcheck.py", "bev_fusion/evaluation/bev_response.py", "bev_fusion/evaluation/board.py", "bev_fusion/evaluation/bev_errors.py"):
    t = text[path]
    flags = sorted(set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', t)))
    subs = sorted(set(re.findall(r'(?:add_parser|sub\.add_parser)\(\s*"([a-z0-9-]+)"', t)))
    print("==", path, "subcommands:", subs)
    for fl in flags:
        c = n_uses(fl, path)
        # uses in other files (docs, tests, shell, scripts)
        own = len(re.findall(re.escape(fl), t))
        print(f"   {fl:28s} own={own} elsewhere={sum(c.values())} {','.join(os.path.basename(x) for x in list(c)[:4])}")
