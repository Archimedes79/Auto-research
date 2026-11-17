p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
old_start = s.index("def compare_dumps(ref, new):")
old_end = s.index("def compare_frames(ref, new):")
new = '''SCORED = ("idx", "e", "pos", "dirv", "flags", "s", "section", "link_src", "link_dst", "link_p", "link_long", "link_a", "link_b")           # the keys `qends_eval.cmd_score` reads from a B dump (the assignment rows are not read by the scorer)


def compare_dumps(ref, new, keys=SCORED):
    """B' dumps frame by frame on the keys the scorer reads: -> dict(frames, shape_mismatch (frames whose node or link arrays differ in length), max_abs (per float key, over the frames of equal shape), equal_int (the integer / bool keys are equal),
    assign_columns_differ (frames whose assignment rows have another number of map pieces: the stored dump of 2026-11-08 predates the GT merge of 2026-11-11 (v9 -> v13), the map layers of the loader changed; the scorer does not read them))"""
    mx, bad, ints, cols = {}, [], True, 0
    for i, (a, b) in enumerate(zip(ref, new)):
        assert a["row"] == b["row"]
        cols += int("assign_P" in a and a["assign_P"].shape[1:] != b["assign_P"].shape[1:])
        if len(a["e"]) != len(b["e"]) or len(a["link_p"]) != len(b["link_p"]):
            bad.append(i)
            continue
        for k in keys:
            v = a[k]
            if v is None or not isinstance(v, np.ndarray):
                continue
            if v.dtype.kind in "iub":
                ints &= bool(np.array_equal(v, b[k]))
            else:
                mx[k] = max(mx.get(k, 0.0), float(np.abs(v.astype(float) - b[k].astype(float)).max()) if v.size else 0.0)
    return dict(frames=len(ref), shape_mismatch=bad, max_abs=mx, equal_int=ints, assign_columns_differ=cols)


'''
s = s[:old_start] + new + s[old_end:]
open(p, "w", encoding="utf-8").write(s)
