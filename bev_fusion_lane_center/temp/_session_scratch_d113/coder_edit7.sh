set -e
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
p = "training/data/lane_past_data.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
rep('''[-30.0, -30.0], [0.0, 2.0]])
    dirv = np.tile([0.0, 1.0], (10, 1))
    dirv[5] = [0.0, -1.0]  ''', '''[-30.0, -30.0], [0.0, 2.0], [-20.0, -20.0]])
    dirv = np.tile([0.0, 1.0], (11, 1))
    dirv[5] = [0.0, -1.0]  ''')
rep('''    dirv[7] = h3
    valid = np.array([True] * 9 + [False])''', '''    dirv[7] = h3
    dirv[10] = [0.0, -1.0]                                                                              # a far weak token against a past token 0.5 m away with the opposite heading: not supported
    valid = np.array([True] * 9 + [False, True])''')
rep('''0.35, 0.5, 0.2])  ''', '''0.35, 0.5, 0.2, 0.4])  ''')
rep('''[20.0, 19.5], [3.0, 2.0]]), np.tile([0.0, 1.0], (3, 1))), identity''', '''[20.0, 19.5], [3.0, 2.0], [-20.0, -19.5]]), np.tile([0.0, 1.0], (4, 1))), identity''')
rep('''fc["novel_n"] == 4 and fc["dup_n"] == 1 and fc["false_n"] == 2 and fc["novel_pose"] == 3''', '''fc["novel_n"] == 4 and fc["dup_n"] == 1 and fc["false_n"] == 3 and fc["novel_pose"] == 3''')
rep('''fc["tw"] == 5 and fc["tw_s"] == 3 and fc["fw"] == 2 and fc["fw_s"] == 1''', '''fc["tw"] == 5 and fc["tw_s"] == 3 and fc["fw"] == 3 and fc["fw_s"] == 1''')
rep('''    assert fc["rows"].shape == (7, 4) and fc["rows"][:, 0].tolist() == ep[1:8].tolist() and fc["rows"][:, 1].tolist() == [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    assert fc["rows"][:, 2].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0] and fc["rows"][:, 3].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0], fc["rows"]''', '''    assert fc["rows"].shape == (8, 4) and fc["rows"][:, 0].tolist() == ep[[1, 2, 3, 4, 5, 6, 7, 10]].tolist() and fc["rows"][:, 1].tolist() == [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    assert fc["rows"][:, 2].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0] and fc["rows"][:, 3].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], fc["rows"]''')
rep('''none["fw"] == 7 and none["fd"] == 2''', '''none["fw"] == 8 and none["fd"] == 2''')
rep('''    assert [b["n"] for b in tb] == [0, 2, 2, 2, 1] and [b["true_share"] for b in tb] == [0.0, 0.0, 1.0, 1.0, 1.0], tb
    assert tb[1]["supported_if_false"] == 0.5 and tb[2]["supported_if_true"] == 0.5 and tb[3]["supported_if_true"] == 1.0 and tb[4]["supported_if_true"] == 0.0 and np.isnan(tb[1]["supported_if_true"]) and np.isnan(tb[0]["supported_if_false"])
    assert [b["novel_share_supported"] for b in tb] == [0.0, 0.0, 1.0, 1.0, 0.0] and [b["novel_share_unsupported"] for b in tb] == [0.0, 0.0, 0.0, 0.0, 1.0], tb''', '''    assert [b["n"] for b in tb] == [0, 2, 2, 2, 2] and [b["true_share"] for b in tb] == [0.0, 0.0, 1.0, 1.0, 0.5], tb
    assert tb[1]["supported_if_false"] == 0.5 and tb[2]["supported_if_true"] == 0.5 and tb[3]["supported_if_true"] == 1.0 and tb[4]["supported_if_true"] == 0.0 and tb[4]["supported_if_false"] == 0.0 and np.isnan(tb[1]["supported_if_true"]) and np.isnan(tb[0]["supported_if_false"])
    assert [b["novel_share_supported"] for b in tb] == [0.0, 0.0, 1.0, 1.0, 0.0] and [b["novel_share_unsupported"] for b in tb] == [0.0, 0.0, 0.0, 0.0, 0.5], tb''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
EOF
