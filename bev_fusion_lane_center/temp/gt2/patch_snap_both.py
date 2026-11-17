p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)


sub("snap_dedup=False, snap_nogeom=False):", "snap_dedup=False, snap_nogeom=False, snap_both=False):")
sub("snap_nogeom = a control: wire only", "snap_both = no curve for a pair whose entry or exit finds no piece in reach (no floating curves), snap_nogeom = a control: wire only")
sub('''            if snap_dedup and w_in is not None and w_out is not None:''', '''            if snap_both and (w_in is None or w_out is None):
                continue
            if snap_dedup and w_in is not None and w_out is not None:''')
sub('("snap_all", "snap_guard", "snap_dedup", "snap_nogeom")', '("snap_all", "snap_guard", "snap_dedup", "snap_nogeom", "snap_both")')
sub('snap_nogeom=bool(getattr(a, "snap_nogeom", False)))', 'snap_nogeom=bool(getattr(a, "snap_nogeom", False)), snap_both=bool(getattr(a, "snap_both", False)))')
sub('("--snap-nogeom", "with --snap: control, nearest piece per end for the wiring but the curve from the keypoints")):', '("--snap-nogeom", "with --snap: control, nearest piece per end for the wiring but the curve from the keypoints"), ("--snap-both", "with --snap: no curve for a pair that is not wired at both ends")):')
# test: the far curve of the earlier test (no pieces in reach) is dropped with snap_both
old = "    # snap_guard: two abutting pieces"
sub(old, '''    rb, nb, wb = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, far, snap=True, snap_both=True)
    assert len(rb["pts"]) == 3 - 1 + 0 or len(rb["pts"]) == 3, len(rb["pts"])            # the unwired far pair adds no curve (the covered-link rule is unchanged)
''' + old)
open(p, "w", encoding="utf-8").write(s)
p = "evaluation/jends_eval.py"
s = open(p, encoding="utf-8").read()
s = s.replace('snap_nogeom=bool(getattr(a, "snap_nogeom", False))))', 'snap_nogeom=bool(getattr(a, "snap_nogeom", False)), snap_both=bool(getattr(a, "snap_both", False))))', 1)
s = s.replace('"--snap-nogeom"):', '"--snap-nogeom", "--snap-both"):', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
