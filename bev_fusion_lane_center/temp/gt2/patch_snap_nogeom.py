p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)


sub("snap_guard=False, snap_dedup=False):", "snap_guard=False, snap_dedup=False, snap_nogeom=False):")
sub("snap_dedup = a second curve between the same two snapped pieces is not added\"\"\"", "snap_dedup = a second curve between the same two snapped pieces is not added, snap_nogeom = a control: wire only the nearest piece per end but draw the curve from the keypoints (separates the wiring rule from the geometry)\"\"\"")
sub('''            if w_in is not None:
                pa, ta = ends[w_in][0], ends[w_in][1]
            if w_out is not None:
                pb, tb = ends[w_out][2], ends[w_out][3]
            if snap_guard:''', '''            if w_in is not None and not snap_nogeom:
                pa, ta = ends[w_in][0], ends[w_in][1]
            if w_out is not None and not snap_nogeom:
                pb, tb = ends[w_out][2], ends[w_out][3]
            if snap_guard:''')
sub('opts = [k for k in ("snap_all", "snap_guard", "snap_dedup") if getattr(a, k, False)]', 'opts = [k for k in ("snap_all", "snap_guard", "snap_dedup", "snap_nogeom") if getattr(a, k, False)]')
sub('snap_dedup=bool(getattr(a, "snap_dedup", False)))', 'snap_dedup=bool(getattr(a, "snap_dedup", False)), snap_nogeom=bool(getattr(a, "snap_nogeom", False)))')
sub('("--snap-dedup", "with --snap: no second curve between the same two snapped pieces")):', '("--snap-dedup", "with --snap: no second curve between the same two snapped pieces"), ("--snap-nogeom", "with --snap: control, nearest piece per end for the wiring but the curve from the keypoints")):')
open(p, "w", encoding="utf-8").write(s)
p = "evaluation/jends_eval.py"
s = open(p, encoding="utf-8").read()
s = s.replace('snap_dedup=bool(getattr(a, "snap_dedup", False))))', 'snap_dedup=bool(getattr(a, "snap_dedup", False)), snap_nogeom=bool(getattr(a, "snap_nogeom", False))))', 1)
s = s.replace('for k_ in ("--snap-all", "--snap-guard", "--snap-dedup"):', 'for k_ in ("--snap-all", "--snap-guard", "--snap-dedup", "--snap-nogeom"):', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
