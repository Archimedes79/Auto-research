p = "training/grid/jgt.py"
s = open(p, encoding="utf-8").read()
old = "        tier = np.asarray(pc.get(\"tier\", np.full(len(q), 3)))\n"
new = "        tier = np.asarray(pc.get(\"tier\", np.full(len(q), 3)))\n        vis = np.asarray(pc.get(\"vis\", np.ones(len(q), bool))).astype(bool)\n"
assert old in s
s = s.replace(old, new, 1)
old = "runs.append(dict(pts=q[k: j + 1], a=(q[k], tan[k]), b=(q[j], tan[j]), driven=bool((tier[k: j + 1] == 1).mean() > 0.5), piece=pc[\"piece\"]))"
new = "runs.append(dict(pts=q[k: j + 1], a=(q[k], tan[k]), b=(q[j], tan[j]), driven=bool((tier[k: j + 1] == 1).mean() > 0.5), piece=pc[\"piece\"], vis_a=bool(vis[k]), vis_b=bool(vis[j])))"
assert old in s
s = s.replace(old, new, 1)
old = "    zones = _zones(runs)\n"
new = "    ent_vis, ext_vis = np.zeros(len(ENT), bool), np.zeros(len(EXT), bool)\n    for r, a_, b_ in zip(runs, eid, xid):                                              # a GT end counts as VISIBLE when a camera sees its point and no box hides it (the l4 label `vis`) for at least one of its runs\n        ent_vis[a_] |= r[\"vis_a\"]\n        ext_vis[b_] |= r[\"vis_b\"]\n    zones = _zones(runs)\n"
assert old in s
s = s.replace(old, new, 1)
old = "                core=core, order=dict("
new = "                ent_vis=ent_vis, ext_vis=ext_vis, core=core, order=dict("
assert old in s
s = s.replace(old, new, 1)
s = s.replace('    jg["order"]                   dict(', '    jg["ent_vis"], jg["ext_vis"]  (E,) / (X,) bool: the end is seen by a camera and not hidden by a box (the l4 label `vis` at the start / end of the connector run; True without the label)\n    jg["order"]                   dict(', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
