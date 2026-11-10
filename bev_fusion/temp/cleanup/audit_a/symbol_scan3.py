import sys, re, os
sys.argv = [sys.argv[0], "999"]
exec(open("/mnt/c/Auto-research/bev_fusion/temp/cleanup/audit_a/symbol_scan.py").read().split("maxo = int")[0])
core = [f"bev_fusion/training/{m}.py" for m in SCOPE]
def cls(f):
    if f.startswith("bev_fusion/training/legacy/"): return "legacy"
    if f.startswith("bev_fusion/studies/"): return "studies"
    if f in core: return "core"
    return "live"
for path in core:
    lines = text[path].split("\n")
    for name, l0, l1, qual, kind in symbols(text[path]):
        if name in SKIP or (name.startswith("__") and name.endswith("__")):
            continue
        n_own = counts[path][name]
        seg = Counter(tok.findall("\n".join(lines[l0 - 1:l1])))
        own_other = n_own - seg[name]
        by = {"core": [], "legacy": [], "studies": [], "live": [], "test": [], "text": []}
        for f, c in counts.items():
            if f == path or not c[name]: continue
            k = kind_of(f)
            if k == "test": by["test"].append(f)
            elif k == "text": by["text"].append(f)
            else: by[cls(f)].append(f)
        if not by["live"]:
            b = lambda k: [os.path.basename(x) for x in by[k]][:5]
            print(f"{os.path.basename(path)}:{l0}-{l1} {qual} [{kind}] own={own_other} core={b('core')} legacy={b('legacy')} studies={b('studies')} tests={len(by['test'])}:{b('test')[:3]} text={len(by['text'])}")
