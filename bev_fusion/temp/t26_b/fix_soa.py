p = "evaluation/soa_maps.py"
s = open(p, encoding="utf-8").read()

# 1. clean vectors()/keys in rows_of
a = s.index("    keys = [f\"{d}|{li}|{ti}\" for d in")
b = s.index("    def reduce(counts, keys=keys):")
new = '''    cols = [(d, li, ti) for d in (0, 1) for li in range(len(LAYERS)) for ti in range(len(TOLERANCES))]        # the keys: d1 (threshold 0.5) then d2 (each system's best), per layer and tolerance
    keys = [f"d{d + 1}|{li}|{ti}" for d, li, ti in cols]
    where = {k: i for i, k in enumerate(keys)}

    def vectors(s):                                                   # (frames, keys, 4): the counts each key reads from a frame
        c = scores.cnt[(region, s)]
        return np.stack([c[:, t05 if d == 0 else best[s][li, ti], li, ti] for d, li, ti in cols], 1)

'''
s = s[:a] + new + s[b:]
s = s.replace('"n_gt": int(n_gt[li, ti] if n_gt.ndim == 2 else n_gt[li])}', '"n_gt": int(n_gt[li])}')

# 2. mutant orientation: no silly globals lookup
s = s.replace('    keep = reference_index.__globals__["reference_index"]\n\n    def bad(offset=None):\n        row, col = keep(offset)', '    keep = reference_index\n\n    def bad(offset=None):\n        row, col = keep(offset)')

# 3. selftest: defaults (so that the OFFSET mutant bites), right planted cells
s = s.replace('''    m = reference_maps(L, 0.5)
    for layer, cells in (("lane_divider", [(20, 10)]), ("drivable_area", [(49, 51), (59, 31)]), ("road_divider", [(98, y) for y in range(NY) if 151 - y == 30])):
        got = {tuple(c) for c in np.argwhere(m[layer] > 0.5)}
        assert got == set(cells) | ({(99, y) for _, y in cells} if layer == "road_divider" else set()), (layer, sorted(got)[:4])      # ix 99 = the clip artefact of the old region
    assert not (m["lane_divider"] > 0.5).sum() - 1 and not (m["walkway"] > 0.5).any() and not (m["carpark_area"] > 0.5).any() and not (m["ped_crossing"] > 0.5).any()''',
'''    m = reference_maps(L)
    for layer, cells in (("lane_divider", {(20, 10)}), ("drivable_area", {(49, 51), (59, 31)}), ("road_divider", {(98, 51), (99, 51)}), ("walkway", set()), ("carpark_area", set()), ("ped_crossing", set())):
        assert {tuple(c) for c in np.argwhere(m[layer] > 0.5)} == cells, (layer, sorted(map(tuple, np.argwhere(m[layer] > 0.5)))[:4])       # (99, 51): the clip artefact of the old region''')
s = s.replace('L[GC_FIXED["road_divider"], 30, 199] = 10.0                     # the last column: ix 98 (the last row of the claim region)',
              'L[GC_FIXED["road_divider"], 100, 199] = 10.0                    # the last column: iy 51, ix 98 (the last row of the claim region); ix 99 reads it too (clipped)')
s = s.replace('assert claim_rows(0.5, 50.0) == 99 and claim_rows(0.0, 50.0) == 100 and claim_rows(0.5, 20.0) == 40', 'assert claim_rows(x_max=50.0) == 99 and claim_rows(0.0, 50.0) == 100 and claim_rows(0.5, 20.0) == 40')
s = s.replace('frame_counts({"ours": ours, "ref": reference_maps(_reference_logits(gt), 0.5)}, gt, fov, claim_rows(0.5, 50.0))', 'frame_counts({"ours": ours, "ref": reference_maps(_reference_logits(gt))}, gt, fov, claim_rows(x_max=50.0))')
s = s.replace('    nr = claim_rows(0.5, 50.0)\n', '    nr = claim_rows(x_max=50.0)\n')

# 4. absolute paths before the chdir in collect
s = s.replace('''    t0 = time.time()
    run = os.path.relpath(os.path.abspath(a.run), ROOT).replace("\\\\", "/") if a.run else None
''', '''    t0 = time.time()
    run = os.path.relpath(os.path.abspath(a.run), ROOT).replace("\\\\", "/") if a.run else None
    os.chdir(ROOT)
''')
s = s.replace('''    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    scores, meta = collect(a)''', '''    for k in ("out", "frames_from", "check", "logits", "tokens"):                   # absolute, the run's folder is entered
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    scores, meta = collect(a)''')
open(p, "w", encoding="utf-8").write(s)
