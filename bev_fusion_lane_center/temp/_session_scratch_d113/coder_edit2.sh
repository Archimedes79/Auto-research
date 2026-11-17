set -e
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
p = "training/data/lane_past_data.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
rep('''    P0 = np.stack([np.zeros(20), np.arange(20.0)], 1)
    Tr = np.tile([np.sin(np.radians(30.0)), np.cos(np.radians(30.0))], (20, 1))''', '''    P0 = np.stack([np.zeros(20), 2.0 * np.arange(20.0)], 1)                                              # points 2 m apart: the tangents must be normalised
    Tr = np.tile([np.sin(np.radians(30.0)), np.cos(np.radians(30.0))], (20, 1))''')
rep('''    assert len(aa) == 20 and np.allclose(aa, 30.0, atol=1e-6) and np.allclose(dd, 0.0)''', '''    assert len(aa) == 20 and np.allclose(aa, 30.0, atol=1e-6) and np.allclose(dd, 0.0)
    s_ = stats(np.arange(101.0))
    assert s_ == dict(n=101, median=50.0, p90=90.0, p99=99.0), s_''')
rep('''        assert it["frames"][2]["row"] == 103 and it["frames"][0]["token"] == "t101"''', '''        assert it["frames"][2]["row"] == 103 and it["frames"][0]["token"] == "t101" and np.allclose(it["rel_t"], ext["rel_t"][[1, 2, 3]]) and np.allclose(it["rel_R"], ext["rel_R"][[1, 2, 3]])''')
rep('''and b["idx"].tolist() == [[1, 2, 3], [5, 6, 7]]''', '''and b["idx"].tolist() == [[1, 2, 3], [5, 6, 7]] and np.allclose(b["rel_t"][1].numpy(), ext["rel_t"][[5, 6, 7]]) and np.allclose(b["rel_R"][0].numpy(), ext["rel_R"][[1, 2, 3]])''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
EOF
