import glob, json, sys, collections
root = sys.argv[1]
pat = sys.argv[2]
lower = set(json.load(open(root + "/project.json"))["lower_is_better"]) | {"ms", "ms_per_frame"}
lasts = collections.Counter(); examples = {}
better_seen = {}
files = glob.glob(root + "/" + pat, recursive=True)
print("files", len(files))
for f in files:
    try:
        d = json.load(open(f))
    except Exception as e:
        print("unreadable", f, e); continue
    m = d.get("metrics") if isinstance(d, dict) else None
    if not isinstance(m, dict):
        continue
    for k, v in d.get("better", {}).items() if isinstance(d.get("better"), dict) else []:
        better_seen.setdefault(k, set()).add(v)
    for k in m:
        last = k.rsplit("/", 1)[-1]
        lasts[last] += 1
        examples.setdefault(last, k)
print("distinct last parts:", len(lasts))
for l, n in sorted(lasts.items()):
    print(("LOWER " if l in lower else "      ") + f"{l:40s} {n:6d}  e.g. {examples[l]}")
# conflicts between better maps and named fallback
bad = []
for k, vs in better_seen.items():
    last = k.rsplit("/", 1)[-1]
    nl = k in lower or last in lower
    for v in vs:
        if (v == "lower") != nl:
            bad.append((k, v, nl))
print("better-map keys:", len(better_seen), "conflicts with named fallback:", len(bad))
for b in bad[:40]:
    print("  ", b)
