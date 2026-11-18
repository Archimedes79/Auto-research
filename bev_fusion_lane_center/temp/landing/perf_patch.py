"""The reviewer's perf item (2026-11-17 23:30, measured bit-identical on 125 VAL frames: 11.1 -> 0.21 ms): `ylong = (y > 0) & (norm(pos[None] - pos[:, None]) > SHORT)` is a dense K x K chord = 92 % of a PTokData item and 43 % of the S4 label step; the same values from the pair list in ONE helper `lane_ptok_data.chord_long(y, pos)` that
the three places call (lane_ptok_data.PTokData, lane_live_tok.label_batch; lane_ptok_gpu.py is the report chat's file and its torch version stays theirs). The text mutants of test_lane_tokcache / test_lane_live move to the helper's text. usage: python perf_patch.py <project folder> [--apply]   (rehearse on a copy first)"""
import os
import sys

root = os.path.abspath(sys.argv[1])
apply_ = "--apply" in sys.argv
changed = {}


def read(rel):
    if rel not in changed:
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            changed[rel] = fh.read()
    return changed[rel]


def rep(rel, old, new):
    s = read(rel)
    assert s.count(old) == 1, f"{rel}: anchor found {s.count(old)} times: {old[:90]!r}"
    changed[rel] = s.replace(old, new)


LP = "training/data/lane_ptok_data.py"
rep(LP, "        yl = (y > 0) & (np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT)", "        yl = chord_long(y, pos)")
helper = '''def chord_long(y, pos):
    """the long class of the positive pairs of `y` (K, K): the chord of the two TOKENS above QT.SHORT (8 m; the class of the pair, not of the GT link) -> (K, K) bool. The same values as the dense `(y > 0) & (norm(pos[None] - pos[:, None]) > SHORT)` from the pair list only
    (the dense K x K chord was 92 % of a `PTokData` item and 43 % of the S4 label step; measured identical on 125 VAL frames)"""
    yl = np.zeros(y.shape, bool)
    i, j = np.nonzero(y > 0)
    yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT)
    return yl


'''
rep(LP, "def selftest():\n    K = 6\n", helper + "def selftest():\n    K = 6\n")
rep(LP, '    toy_cache_test()\n    print("lane_ptok_data selftest ok")', '''    pos6 = np.array([[0.0, 0.0], [8.0, 0.0], [8.0001, 0.0], [20.0, 0.0], [3.0, 4.0], [0.0, 9.0]])
    y6 = np.zeros((6, 6), np.uint8)
    y6[0, 1] = y6[0, 2] = y6[0, 3] = y6[0, 4] = y6[5, 0] = y6[1, 3] = 1
    want = np.zeros((6, 6), bool)
    want[0, 2] = want[0, 3] = want[5, 0] = want[1, 3] = True                       # a chord of exactly 8 m is short, 8.0001 m long, 5 m short; the pair (1, 3) is 12 m; a pair that is no positive is never long
    assert np.array_equal(chord_long(y6, pos6), want), chord_long(y6, pos6)
    assert not chord_long(np.zeros((6, 6), np.uint8), pos6).any()
    dense = (y6 > 0) & (np.linalg.norm(pos6[None, :, :] - pos6[:, None, :], axis=-1) > QT.SHORT)
    assert np.array_equal(chord_long(y6, pos6), dense), "the pair-list form equals the dense form"
    toy_cache_test()
    print("lane_ptok_data selftest ok")''')
rep("training/data/lane_live_tok.py", "ylong=(y > 0) & (np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT), yscale=ys,", "ylong=LP.chord_long(y, pos), yscale=ys,")
TC = "training/grid/test_lane_tokcache.py"
rep(TC, '''    (PT, "ylong marks every pair", "yl = (y > 0) & (np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1) > QT.SHORT)", "yl = (y > 0)"),
    (PT, "ylong from the wrong token distance", "> QT.SHORT)", "> 2 * QT.SHORT)"),''', '''    (PT, "ylong marks every pair", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT)", "yl[i, j] = True"),
    (PT, "ylong from the wrong token distance", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT)", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > 2 * QT.SHORT)"),
    (PT, "a chord of exactly 8 m is long", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT)", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) >= QT.SHORT)"),
    (PT, "ylong uses the chord of the wrong pair", "yl[i, j] = (np.linalg.norm(pos[j] - pos[i], axis=-1) > QT.SHORT)", "yl[i, j] = (np.linalg.norm(pos[i] - pos[0], axis=-1) > QT.SHORT)"),''')
TL = "training/grid/test_lane_live.py"
rep(TL, '''    ("ylong is classed by twice the chord", "axis=-1) > QT.SHORT)", "axis=-1) > 2 * QT.SHORT)"),
    ("a chord of exactly 8 m is long", "axis=-1) > QT.SHORT)", "axis=-1) >= QT.SHORT)"),''', '''    ("ylong is classed by twice the chord", "ylong=LP.chord_long(y, pos)", "ylong=LP.chord_long(y, 2 * pos)"),
    ("ylong marks every positive pair", "ylong=LP.chord_long(y, pos)", "ylong=(y > 0)"),''')
if apply_:
    for rel, s in changed.items():
        with open(os.path.join(root, rel), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(s)
    os.makedirs(os.path.join(root, "temp", "landing"), exist_ok=True)
    with open(os.path.join(root, "temp", "landing", "perf_changed.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(changed)) + "\n")
    print(f"applied: {len(changed)} files")
else:
    print(f"checked: every anchor found, {len(changed)} files would change (nothing written)")
