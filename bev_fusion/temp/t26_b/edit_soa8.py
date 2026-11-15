p = "evaluation/soa_maps.py"
s = open(p, encoding="utf-8").read()
old = '''    return "soa_maps selftest ok:'''
new = '''    # 9. the identity of the GT: another one and an UNKNOWN one are refused unless told (no run, a folder that does not exist, no file, no key); the same GT (also through a version that makes it) is not
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "empty"))
        os.makedirs(os.path.join(t, "nokey"))
        os.makedirs(os.path.join(t, "v8"))
        json.dump({}, open(os.path.join(t, "nokey", "gt.json"), "w"))
        json.dump({"gt_signature": "0.5_4_6a7aa7864a"}, open(os.path.join(t, "v8", "gt.json"), "w"))
        assert trained_gt(None) is None and trained_gt(os.path.join(t, "nowhere")) is None and trained_gt(os.path.join(t, "empty")) is None and trained_gt(os.path.join(t, "nokey")) is None
        assert trained_gt(os.path.join(t, "v8")) == "0.5_4_6a7aa7864a"

        def refused(fn):
            try:
                fn()
            except SystemExit as e:
                return str(e)
            return ""
        for run in (None, os.path.join(t, "nowhere"), os.path.join(t, "empty"), os.path.join(t, "nokey")):
            msg = refused(lambda: gt_identity(run, "6a7aa7864a"))
            assert "unknown" in msg and "--allow-gt-mismatch" in msg, (run, msg)             # no TypeError for run None, no pass for an unknown GT
        assert "trained on 0.5_4_6a7aa7864a" in refused(lambda: gt_identity(os.path.join(t, "v8"), "a107e0b4e8"))
        assert check_gt("a107e0b4e8", "0.5_4_6a7aa7864a", True) is False and check_gt("6a7aa7864a", None, True) is None
        assert check_gt("c643a7539d", "0.5_4_6a7aa7864a") is True and check_gt("6a7aa7864a", "0.5_4_6a7aa7864a") is True, "the same GT through the version that makes it (v9 = v8)"
    # 10. frames of a stored list that the run's val set does not have are reported, not dropped silently
    uni = [{"i": k, "token": f"t{k}", "scene": f"s{k // 2}"} for k in range(6)]
    sel, missing = select_frames(uni, None, ["t3", "tx", "t1", "ty"])
    assert [u["token"] for u in sel] == ["t3", "t1"] and missing == ["tx", "ty"], (sel, missing)
    assert select_frames(uni, 3, None)[1] == [] and len(select_frames(uni, 3, None)[0]) == 3
    # 11. the best-threshold comparison says it is in-sample, the plain one does not
    assert r["d2"]["in_sample_thresholds"] is True and "in_sample_thresholds" not in r["d1"]
    return "soa_maps selftest ok:'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('old region artefact"', 'old region artefact, GT identity (unknown = refused), missing frames reported, in-sample flag"')

mut_anchor = '''def mutants():'''
mut_new = '''@mutant("an unknown training GT passes the identity check")
def _m_unknown():
    keep = check_gt
    return {"check_gt": lambda sig, trained, allow_mismatch=False: None if trained is None else keep(sig, trained, allow_mismatch)}


@mutant("no run folder is a TypeError instead of an unknown GT")
def _m_run_none():
    return {"trained_gt": lambda run: json.load(open(os.path.join(run, "gt.json"))).get("gt_signature")}


@mutant("frames missing from the run's val set are dropped silently")
def _m_silent():
    keep = select_frames
    return {"select_frames": lambda universe, n=None, only=None: (keep(universe, n, only)[0], [])}


'''
s = s.replace(mut_anchor, mut_new + mut_anchor, 1)
open(p, "w", encoding="utf-8").write(s)
