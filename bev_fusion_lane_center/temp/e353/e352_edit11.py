p = "evaluation/tests/test_jprune_b.py"
s = open(p, encoding="utf-8").read()
a = s.index("def test_empty_set_reproduces_the_stored_dumps():")
b = s.index("# ---------------------------------------------------------------- an edited pass, the point folder")
new = '''def test_empty_set_reproduces_the_stored_dumps():
    with tempfile.TemporaryDirectory() as tmp:
        r = PB.check(0, 8, "cpu", out=tmp)
    print(json.dumps(r, default=float))
    b, j = r["b"], r["junction"]
    assert b["frames"] == 8 and b["nodes_matched"] > 3000 and b["nodes_only_in_ref"] + b["nodes_only_in_new"] <= 0.005 * b["nodes_matched"] and b["e_max_unmatched"] < 0.4, b            # the node sets (e >= 0.2) are the stored ones but for the threshold
    assert b["nodes"]["e"] < 2e-3 and b["nodes"]["pos"] < 1e-2 and b["nodes"]["dirv"] < 2e-2 and b["nodes"]["flags"] < 1e-2 and b["long_equal"], b["nodes"]                           # float32 on the CPU against the stored dump of the GPU
    assert b["links_matched"] > 20000 and b["links_only_in_ref"] + b["links_only_in_new"] <= 0.003 * b["links_matched"] and b["links"]["p"] < 2e-2 and b["links"]["a"] < 1e-2 and b["links"]["b"] < 1e-2, b
    assert j["keypoints_matched"] > 100 and j["keypoints_only_in_ref"] + j["keypoints_only_in_new"] <= 0.01 * j["keypoints_matched"] and j["allowed_equal"], j
    assert j["max_abs"]["kp_pos"] < 5e-3 and j["max_abs"]["kp_p"] < 5e-3 and j["max_abs"].get("logit", 0.0) < 5e-2, j["max_abs"]


def test_dump_comparison_catches_a_changed_dump():
    """the comparison is not blind: a permuted dump is equal, a moved node / a changed link probability / a dropped keypoint is reported"""
    import copy
    ref = pickle.load(open(os.path.join(ROOT, "temp", "e276", "predfull_e273_p_s0.pkl"), "rb"))[:2]
    perm = copy.deepcopy(ref)
    for d in perm:                                                                                       # the same graph with its nodes in another order (links and section starts renumbered through the filtered list)
        n = len(d["e"])
        o = np.random.RandomState(0).permutation(n)
        inv = np.empty(n, int)
        inv[o] = np.arange(n)
        for k in ("e", "pos", "dirv", "flags", "s"):
            d[k] = d[k][o]
        d["link_src"], d["link_dst"] = inv[d["link_src"]], inv[d["link_dst"]]
        where = -np.ones(int(max(d["idx"].max(), d["section"].max())) + 1, int)
        where[d["idx"]] = np.arange(n)
        d["section"] = np.where(where[d["section"][o]] >= 0, d["idx"][o][inv[where[d["section"][o]]]] if False else d["section"][o], -1)
        d["idx"] = d["idx"][o]
    same = PB.compare_dumps(ref, perm)
    assert same["nodes_only_in_ref"] == same["nodes_only_in_new"] == same["links_only_in_ref"] == same["links_only_in_new"] == 0 and max(same["nodes"].values()) == 0.0 and same["links"]["p"] == 0.0, same
    bad = copy.deepcopy(ref)
    bad[0]["pos"][3] += 3.0
    bad[1]["link_p"][5] += 0.1
    r = PB.compare_dumps(ref, bad)
    assert r["nodes_only_in_ref"] == 1 and r["nodes_only_in_new"] == 1 and r["links"]["p"] >= 0.1 - 1e-9, r
    jf = pickle.load(open(os.path.join(ROOT, "runs", "e345_p000_e324_s_s0", "ends_val.pkl"), "rb"))[:2]
    assert PB.compare_frames(jf, copy.deepcopy(jf))["max_abs"]["kp_pos"] == 0.0
    drop = copy.deepcopy(jf)
    drop[0]["valid"][0][np.flatnonzero(drop[0]["valid"][0])[0]] = False
    assert PB.compare_frames(jf, drop)["keypoints_only_in_ref"] == 1


'''
s = s[:a] + new + s[b:]
s = s.replace("test_empty_set_reproduces_the_stored_dumps, test_edited_pass", "test_empty_set_reproduces_the_stored_dumps, test_dump_comparison_catches_a_changed_dump, test_edited_pass")
open(p, "w", encoding="utf-8").write(s)
