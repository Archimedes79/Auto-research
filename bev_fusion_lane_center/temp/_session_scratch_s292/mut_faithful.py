"""E358: mutation evidence for the new `faithful` lines of jwire_data.py: each mutant = the source with ONE replacement, executed as a fresh module, only `_test_faithful()` is run (the tree file is never touched).
usage (problem folder): wsl bash tools/env/_py14.sh temp/_session_scratch_s292/mut_faithful.py"""
import os, sys, types
ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, sub))
path = ROOT + "/evaluation/jwire_data.py"
src = open(path, encoding="utf-8").read()
M = [
 ("B loader bs = the pair's", 'bs=DUMP_B["bs"], workers=workers, shuffle=False, map_mode=DUMP_B["map_mode"]', 'bs=DUMP_PAIR["bs"], workers=workers, shuffle=False, map_mode=DUMP_B["map_mode"]'),
 ("B loader map mode = the pair's", 'workers=workers, shuffle=False, map_mode=DUMP_B["map_mode"], pixels=False, strict=True, ends=True)', 'workers=workers, shuffle=False, map_mode=DUMP_PAIR["map_mode"], pixels=False, strict=True, ends=True)'),
 ("pair loader bs = B's", 'bs=DUMP_PAIR["bs"], workers=DUMP_PAIR["workers"]', 'bs=DUMP_B["bs"], workers=DUMP_PAIR["workers"]'),
 ("pair loader map mode = B's", 'map_mode=DUMP_PAIR["map_mode"], pixels=False)', 'map_mode=DUMP_B["map_mode"], pixels=False)'),
 ("pair loader workers", 'workers=DUMP_PAIR["workers"]', 'workers=workers'),
 ("junction flag not set", "LD.QData.DEFAULT_JUNCTION = True  ", "LD.QData.DEFAULT_JUNCTION = False  "),
 ("junction flag not restored", "LD.QData.DEFAULT_JUNCTION = old_junction", "pass"),
 ("pair loader: no missing assertion", "assert ds2.missing == 0", "assert True"),
 ("ends flag not restored", "    LD.QData.DEFAULT_ENDS = old_ends\n", "    pass\n"),
 ("join takes item 0 of the batch", "by_row[int(b[\"row\"][i])] = frame_dict(o, i, ends)", "by_row[int(b[\"row\"][i])] = frame_dict(o, 0, ends)"),
 ("join by position, not row", "by_row[int(b[\"row\"][i])] = frame_dict(o, i, ends)", "by_row[rows[len(by_row)]] = frame_dict(o, i, ends)"),
 ("GT of another row", "        pr, g = b_side[r]\n", "        pr, g = b_side[r]\n        g = b_side[rows[0]][1]\n"),
 ("prediction of another row", "        pr, g = b_side[r]\n", "        pr, g = b_side[rows[0]][0], b_side[r][1]\n"),
 ("B pass does not stop at nmax", "        if len(b_side) >= nmax:\n            break\n    rows", "    rows"),
 ("rows not cut at nmax", "rows = list(b_side)[:nmax]", "rows = list(b_side)"),
 ("no pair-loader completeness assertion", 'assert all(r in by_row for r in rows), f"the pair loader did not reach', 'assert True or all(r in by_row for r in rows), f"the pair loader did not reach'),
 ("guard: train allowed", 'assert not faithful or (split == "val_clean" and 0 < nmax <= FAITHFUL_MAX)', 'assert not faithful or (0 < nmax <= FAITHFUL_MAX)'),
 ("guard: nmax 0 allowed", 'assert not faithful or (split == "val_clean" and 0 < nmax <= FAITHFUL_MAX)', 'assert not faithful or (split == "val_clean" and nmax <= FAITHFUL_MAX)'),
 ("guard: FAITHFUL_MAX not applied", 'assert not faithful or (split == "val_clean" and 0 < nmax <= FAITHFUL_MAX)', 'assert not faithful or (split == "val_clean" and 0 < nmax)'),
 ("faithful ignored", "    if faithful:\n        return _live_records_faithful(thr, nmax, workers, dev, mb, mp, ends)\n", ""),
 ("constant DUMP_B bs", 'DUMP_B = dict(bs=4, map_mode="exact")', 'DUMP_B = dict(bs=5, map_mode="exact")'),
 ("constant DUMP_PAIR bs", 'DUMP_PAIR = dict(bs=8, map_mode="error", workers=2)', 'DUMP_PAIR = dict(bs=4, map_mode="error", workers=2)'),
]
killed = 0
for name, a, b in M:
    if src.count(a) != 1:
        print(f"NOT APPLIED  {name}: {src.count(a)} matches"); continue
    mod = types.ModuleType("jwire_data_mut")
    mod.__file__ = path
    try:
        exec(compile(src.replace(a, b), path, "exec"), mod.__dict__)
        mod._test_faithful()
        print(f"SURVIVED     {name}")
    except Exception as e:
        killed += 1
        print(f"killed       {name}: {type(e).__name__}: {str(e)[:70]}")
print(f"{killed} of {len(M)} mutants killed")
