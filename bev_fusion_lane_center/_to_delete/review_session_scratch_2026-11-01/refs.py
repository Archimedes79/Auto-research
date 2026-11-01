"""For each module under the lane project's code dirs, list who references it (import or file name)."""
import os, re, sys, collections
ROOT = sys.argv[1]  # repo root
P = os.path.join(ROOT, 'bev_fusion_lane_center')
code_dirs = ['training', 'evaluation', 'tools', 'reports/build']
mods = {}
for d in code_dirs:
    for dp, dn, fn in os.walk(os.path.join(P, d)):
        if '__pycache__' in dp:
            continue
        for f in fn:
            if f.endswith(('.py', '.sh', '.ps1')):
                mods.setdefault(os.path.splitext(f)[0], []).append(os.path.relpath(os.path.join(dp, f), P))
# corpus: all text files in repo except big stuff
texts = {}
skip = ('literature/papers', '/.git', '__pycache__', '/wsl', 'datasets', 'node_modules', 'literature/code', 'literature/deep')
for top in ['bev_fusion_lane_center', 'autoresearch', 'bev_fusion']:
    for dp, dn, fn in os.walk(os.path.join(ROOT, top)):
        rel = os.path.relpath(dp, ROOT).replace('\\', '/')
        if any(s in '/' + rel for s in skip):
            continue
        if rel.startswith('bev_fusion_lane_center/runs') and 'cards' not in rel:
            continue
        for f in fn:
            if f.endswith(('.py', '.sh', '.ps1', '.md', '.json', '.csv', '.txt', '.yml')):
                fp = os.path.join(dp, f)
                try:
                    if os.path.getsize(fp) > 3_000_000:
                        continue
                    texts[os.path.relpath(fp, ROOT).replace('\\', '/')] = open(fp, encoding='utf-8', errors='ignore').read()
                except Exception:
                    pass
def cat(path):
    p = path
    if p.startswith('bev_fusion_lane_center/'):
        q = p[len('bev_fusion_lane_center/'):]
        for d in ['training', 'evaluation', 'tools', 'reports/build']:
            if q.startswith(d + '/'):
                return 'LIVE'
        if q.startswith('temp/'):
            return 'temp'
        if q.startswith('_to_delete'):
            return 'trash'
        if '/history/' in q or q.startswith('experiment_execution/history'):
            return 'hist'
        if q.startswith('experiments/'):
            return 'exp'
        return 'doc:' + q.split('/')[0]
    return 'other:' + p.split('/')[0]
out = []
for m, paths in sorted(mods.items()):
    pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(m) + r'(?![A-Za-z0-9_])')
    hits = collections.defaultdict(list)
    for t, s in texts.items():
        if t.endswith(tuple(x.replace('\\', '/') for x in [])):
            pass
        own = any(t.endswith('/' + p.replace('\\', '/')) for p in paths)
        if own:
            continue
        if pat.search(s):
            hits[cat(t)].append(t)
    live = hits.get('LIVE', [])
    summary = {k: len(v) for k, v in hits.items()}
    out.append((len(live), m, paths, summary, live[:6]))
for n, m, paths, summ, live in sorted(out, key=lambda x: (x[0], x[1])):
    print(n, m, paths, summ, [l.replace('bev_fusion_lane_center/', '') for l in live])
