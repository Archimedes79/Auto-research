p = "tests/test_evaluation_api.py"
s = open(p, encoding="utf-8").read()
old = '''        assert len(os.listdir(os.path.join(rs_dir, "frames"))) == 1, "only the frame asked for is stored (the scene ran whole up to it)"'''
new = '''        assert len(os.listdir(os.path.join(rs_dir, "frames"))) == 1, "only the frame asked for is stored (the scene ran whole up to it)"
        from evalsuite.results import ResultSet
        rs0 = ResultSet.open(rs_dir)
        stored0 = rs0.load(rs0.refs()[0])
        assert "loss" not in stored0.info and not any(k.startswith("group.") for k in stored0.info) and stored0.info.get("device"), "the Runner read no ground truth: no validation loss in what it stored"
        assert rs0.manifest["identity"]["inputs"].get("dataset") and "gt_signature" not in rs0.manifest["identity"]["inputs"], rs0.manifest["identity"]["inputs"]'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
