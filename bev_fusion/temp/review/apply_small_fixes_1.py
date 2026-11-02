"""Small verified fixes of the structure review 2026-10-30 (batch 1). Each replacement asserts that the old text is present once."""
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def edit(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == 1, (path, old[:50], s.count(old))
    open(path, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    print("edited", path)


# 1 autogt/clean.py: --all-caches deleted EVERY sub-folder (the isdir alternative defeated the index.json guard)
edit("autogt/clean.py",
     '''items += [(os.path.join(all_caches, d), "cache") for d in sorted(os.listdir(all_caches))
                  if os.path.exists(os.path.join(all_caches, d, "index.json")) or os.path.isdir(os.path.join(all_caches, d))]''',
     '''items += [(os.path.join(all_caches, d), "cache") for d in sorted(os.listdir(all_caches))
                  if os.path.exists(os.path.join(all_caches, d, "index.json"))]          # only what autogt marked as its cache''')

# 2 autogt/cli.py: the cache folder is keyed by the signature, which omits res and ss: refuse to mix grids
edit("autogt/cli.py",
     '''    sig_dir = os.path.join(out, gt.signature())
    os.makedirs(sig_dir, exist_ok=True)
    names = scenes or gt.scenes()''',
     '''    sig_dir = os.path.join(out, gt.signature())
    idx_path = os.path.join(sig_dir, "index.json")
    if os.path.exists(idx_path):               # the signature does not contain res / ss: existing files would be reused silently
        held = json.load(open(idx_path))
        if abs(held.get("res", res) - res) > 1e-9 or held.get("ss", ss) != ss:
            raise SystemExit(f"{sig_dir} holds res {held.get('res')} / ss {held.get('ss')} (asked: res {res} / ss {ss}): "
                             "build into another output folder")
    os.makedirs(sig_dir, exist_ok=True)
    names = scenes or gt.scenes()''')

# 3 training/gt_bev.py: a pinned run snapshot (runs/<name>/code) has no autogt next to it: search upwards for the folder
edit("training/gt_bev.py",
     '''_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
''',
     '''_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
while not os.path.isdir(os.path.join(_ROOT, "autogt")) and os.path.dirname(_ROOT) != _ROOT:
    _ROOT = os.path.dirname(_ROOT)      # a run snapshot (runs/<name>/code/) sits three levels below the bev_fusion folder
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
''')
