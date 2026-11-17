set -e
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
p = "training/data/lane_past_data.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
rep('''    try:
        link(bad, rows)
        raise AssertionError("a previous keyframe outside the cache was accepted")
    except AssertionError as e:
        assert "not in the cache" in str(e), e
    so = dict(pose)
    check_stream(so, rows, prev, None)
''', '''    refuses(lambda: link(bad, rows), "not in the cache")
    so = dict(pose)
    check_stream(so, rows, prev, None)
''')
rep('''    for wrong in (np.array([-1, 0, 0, -1, 3], np.int32), np.array([-1, 0, 1, -1, -1], np.int32)):
        try:
            check_stream(so, rows, wrong, None)
            raise AssertionError("a wrong stream was accepted")
        except AssertionError as e:
            assert "stream" in str(e) or "scene starts" in str(e) or "row right before" in str(e), e
''', '''    refuses(lambda: check_stream(so, rows, np.array([-1, 0, 0, -1, 3], np.int32), None), "row right before")
    refuses(lambda: check_stream(so, rows, np.array([-1, 0, 1, -1, -1], np.int32), None), "scene starts of the stream order")
    refuses(lambda: check_stream(so, [4, 3, 5, 8, 9], prev, None), "not the stream order")
''')
rep('''    for wrong in ([39, 40], [39, 40, 41, 900], [40, 39, 41], [39, 40, 900]):
        try:
            check_rows("val", wrong, sv, None)
            raise AssertionError(f"rows {wrong} were accepted as VAL")
        except AssertionError as e:
            assert "splits.json" in str(e), e
    try:
        check_rows("train", [1, 2, 2], None, None)
        raise AssertionError("a row twice")
    except AssertionError:
        pass
''', '''    for wrong in ([39, 40], [39, 40, 41, 900], [40, 39, 41], [39, 40, 900]):
        refuses(lambda: check_rows("val", wrong, sv, None), "splits.json")
    refuses(lambda: check_rows("train", [1, 2, 2], None, None), "a row twice")
''')
rep('''        try:
            PastData("train", dir=d)
            raise AssertionError("a changed cache was accepted")
        except AssertionError as e:
            assert "another cache" in str(e), e
        os.remove(os.path.join(d, "past.json"))
        try:
            PastData("train", dir=d)
            raise AssertionError("a cache without the extension was accepted")
        except AssertionError as e:
            assert "no past.json" in str(e), e
''', '''        refuses(lambda: PastData("train", dir=d), "another cache")
        os.remove(os.path.join(d, "past.json"))
        refuses(lambda: PastData("train", dir=d), "no past.json")
''')
rep('''def toy_world():''', '''def refuses(fn, text):
    """fn must fail with an AssertionError whose message holds `text` (a call that is accepted raises a RuntimeError, which no test catches)"""
    try:
        fn()
    except AssertionError as e:
        assert text in str(e), (text, str(e))
        return
    raise RuntimeError(f"accepted although it must be refused ({text})")


def toy_world():''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
EOF
