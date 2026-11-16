import os, shutil, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, "work", "training", "grid")
ORIG = os.path.join(HERE, "orig")
SCORE = 'score = (e_logit.detach().float() + bias * gt).reshape(-1)[valid]'
TIES = '''    cand = torch.where(score >= mx[seg], idx, torch.full_like(idx, big))
    win = torch.full((nseg,), big, device=dev, dtype=torch.long).scatter_reduce(0, seg, cand, "amin", include_self=True)
    win = win[win < big]'''
TIES_HI = '''    cand = torch.where(score >= mx[seg], idx, torch.full_like(idx, -1))
    win = torch.full((nseg,), -1, device=dev, dtype=torch.long).scatter_reduce(0, seg, cand, "amax", include_self=True)
    win = win[win >= 0]'''
M = [
 ("M01 argmax->argmin", "qgraph.py", SCORE, 'score = (-(e_logit.detach().float() + bias * gt)).reshape(-1)[valid]'),
 ("M02 ties->highest index", "qgraph.py", TIES, TIES_HI),
 ("M03 bias ignored", "qgraph.py", SCORE, 'score = (e_logit.detach().float() + 0.0 * gt).reshape(-1)[valid]'),
 ("M04 bias as *", "qgraph.py", SCORE, 'score = (e_logit.detach().float() * (1.0 + bias * gt)).reshape(-1)[valid]'),
 ("M05 winner always GT cell", "qgraph.py", SCORE, 'score = (e_logit.detach().float() * 0 + gt).reshape(-1)[valid]'),
 ("M06 offsets from sparse map", "qgraph.py", 'off_key, dir_key = "offset_dense", "dirv_dense"', 'off_key, dir_key = "offset", "dirv"'),
 ("M07 anchors not fixed", "qzones.py", "            if gid != s and s in anchors:\n                continue\n", "            pass\n"),
 ("M08 later node on ties", "qzones.py", "key=lambda q: (abs(gid - q), q)", "key=lambda q: (abs(gid - q), -q)"),
 ("M09 npos = dense node count", "qgraph.py", "    npos = ex.sum().clamp_min(1.0)\n    info = {}", "    npos = (batch['exists_dense'].to(dev).sum() if zinfo is not None else ex.sum()).clamp_min(1.0)\n    info = {}"),
 ("M10 exists_m also marks GT cell", "qgraph.py", "    ex[win] = 1.0\n", "    ex[win] = 1.0\n    ex[gt.reshape(-1) > 0.5] = 1.0\n"),
 ("M11a E273 combination 2.0->1.5 l_u", "qgraph.py", "+ 0.5 * l_s + 2.0 * l_u, info", "+ 0.5 * l_s + 1.5 * l_u, info"),
 ("M11b E273 flags pos_weight 6->5", "qgraph.py", "pos_weight=torch.tensor(6.0, device=dev)) * m[:, None]", "pos_weight=torch.tensor(5.0, device=dev)) * m[:, None]"),
 ("M11c E273 focal alpha at call", "qgraph.py", "l_e = focal_bce(no[\"e\"].float(), ex, norm=npos, pos_w=wmap, neg_w=negw)", "l_e = focal_bce(no[\"e\"].float(), ex, alpha=0.4, norm=npos, pos_w=wmap, neg_w=negw)"),
 ("M12 batch offset missing", "qgraph.py", "seg = (torch.arange(B, device=dev)[:, None, None] * zmax + zone).reshape(-1)[valid]", "seg = zone.reshape(-1)[valid]"),
 ("M13 win_off_gt over all zones", "qgraph.py", 'reshape(-1)[win] == 1', 'reshape(-1)[win] >= 0'),
 ("M14 mask m = sparse GT cells in zone mode", "qgraph.py", "    m = ex > 0.5\n    npos", "    m = (batch['exists'].to(dev) > 0.5) if zinfo is not None else (ex > 0.5)\n    npos"),
 ("M15 mask m = all zone cells", "qgraph.py", "    m = ex > 0.5\n    npos", "    m = (batch['zone'].to(dev) >= 0) if zinfo is not None else (ex > 0.5)\n    npos"),
 ("M16 kind swapped", "qzones.py", "kind.append(0 if s in anchors else 1)", "kind.append(1 if s in anchors else 0)"),
 ("M17 first/last not anchors", "qzones.py", "        anchors.add(int(gd[ids[0]]))\n        anchors.add(int(gd[ids[-1]]))\n", "        pass\n"),
 ("M18 nearest by Euclid not index", "qzones.py", "key=lambda q: (abs(gid - q), q)", "key=lambda q: (float(np.linalg.norm(nd['pts'][gid] - nd['pts'][q])), q)"),
 ("M19 win_off_gt inverted", "qgraph.py", "off_gt = (~on_gt[free]).float().mean()", "off_gt = (on_gt[free]).float().mean()"),
 ("M20 tier-switch assert removed", "qgraph.py", 'assert CFG["tier_w"] is None and CFG["vis_w"] == 1.0 and CFG["neg_w"] is None, "the zone loss runs without the E276 tier switches"', "pass"),
 ("M21 zone cell swapped (j,i)", "qzones.py", "zone[cell[m][0], cell[m][1]] = z", "zone[cell[m][1], cell[m][0]] = z"),
 ("M22 flags from winner-masked dense? uout uses sparse -> use dense exists as ex", "qgraph.py", "ex, zinfo = zone_targets(no[\"e\"], batch, CFG[\"zone_bias\"])", "ex, zinfo = zone_targets(no[\"e\"], batch, CFG[\"zone_bias\"]); ex = torch.maximum(ex, (batch['zone_kind'].to(dev) == 0).float())"),
 ("M23 bias on free zones only -> bias to fixed also (no-op); bias 0.05->0.5 default in CFG", "qgraph.py", 'zone_bias=0.05)', 'zone_bias=0.5)'),
 ("M24 scatter -inf init without self -> include_self False", "qgraph.py", 'scatter_reduce(0, seg, score, "amax", include_self=True)', 'scatter_reduce(0, seg, score, "amax", include_self=False)'),
]
only = sys.argv[1:]
res = []
for name, f, old, new in M:
    if only and not any(name.startswith(o) for o in only):
        continue
    for fn in ("qzones.py", "qgraph.py"):
        shutil.copy(os.path.join(ORIG, fn), os.path.join(G, fn))
    p = os.path.join(G, f)
    s = open(p).read()
    assert old in s, ("pattern not found", name)
    open(p, "w").write(s.replace(old, new, 1))
    t0 = time.time()
    r = subprocess.run([sys.executable, "test_zones.py"], cwd=G, capture_output=True, text=True, timeout=900)
    tail = (r.stdout + r.stderr).strip().splitlines()[-1][:160] if (r.stdout + r.stderr).strip() else ""
    verdict = "SURVIVES" if r.returncode == 0 else "killed"
    # which test killed it
    out = r.stdout + r.stderr
    which = ""
    if r.returncode != 0:
        import re
        m = re.findall(r'File ".*test_zones.py", line (\d+), in (\w+)', out)
        which = m[-1][1] + ":" + m[-1][0] if m else ""
    line = f"{name:60s} {verdict:9s} {which:40s} {time.time() - t0:5.0f}s  {tail if verdict == 'SURVIVES' else ''}"
    print(line, flush=True)
    res.append(line)
for fn in ("qzones.py", "qgraph.py"):
    shutil.copy(os.path.join(ORIG, fn), os.path.join(G, fn))
