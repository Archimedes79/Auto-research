R = "/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/tests/"
p = R + "test_jprune_probe.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''sw.__dict__.update(run="e312_p_s0", dev="cpu", out=tmp,''', '''sw.__dict__.update(run="e312_p_s0", dev="cpu", deterministic=False, out=tmp,''')
rep('''sw.rec = dict(run="e312_p_s0", attach=False, rows=sw.rows,''', '''sw.rec = dict(run="e312_p_s0", attach=False, deterministic=False, rows=sw.rows,''')
rep("        sw.nulls(False)\n", "        sw.nulls()\n")
rep('''assert sw.rec["passes"]["p001"]["seconds"] == sec and sw.rec["passes"]["p001"]["saved_macs"]''', '''assert sw.rec["passes"]["p001"]["gpu_mb"] is None, "no GPU memory on the CPU"
        assert sw.rec["passes"]["p001"]["seconds"] == sec and sw.rec["passes"]["p001"]["saved_macs"]''')
rep('''        before = os.path.getmtime(os.path.join(d, "ends_val.pkl"))
        sw.dump_point(7, spec)                                                 # the same set again (a resumed job): nothing happens
        assert os.path.getmtime(os.path.join(d, "ends_val.pkl")) == before
''', '''        assert not [n for n in os.listdir(d) if n.endswith(".tmp")] and os.path.getmtime(os.path.join(d, "point.json")) >= os.path.getmtime(os.path.join(d, "ends_val.pkl")), "no temporary file is left, point.json is written last"
        before = os.path.getmtime(os.path.join(d, "ends_val.pkl"))
        sw.dump_point(7, spec)                                                 # the same set again (a resumed job): nothing happens
        assert os.path.getmtime(os.path.join(d, "ends_val.pkl")) == before
''')
rep('''        assert sorted(os.listdir(os.path.join(tmp, "runs", "e312_p_s0"))) == ["model.pt"], "the stored run's folder gets no file"
''', '''        assert sorted(os.listdir(os.path.join(tmp, "runs", "e312_p_s0"))) == ["model.pt"], "the stored run's folder gets no file"
        # crash safety: a crash between the pickle and point.json leaves an incomplete folder, which is redone and which the scorer refuses; a pickle that does not match its point.json is refused, never accepted by an existence check
        os.remove(os.path.join(d, "point.json"))
        old_write = PP.atomic_write
        try:
            def crash(path, data):
                if path.endswith("point.json"):
                    raise KeyboardInterrupt("crash before point.json")
                old_write(path, data)
            PP.atomic_write = crash
            os.remove(os.path.join(d, "ends_val.pkl"))
            try:
                sw.dump_point(7, spec)
                raise RuntimeError("the crash did not happen")
            except KeyboardInterrupt:
                pass
        finally:
            PP.atomic_write = old_write
        assert os.path.exists(os.path.join(d, "ends_val.pkl")) and not os.path.exists(os.path.join(d, "point.json"))
        try:
            PT.score_point("e312_p_s0", "e312", 7, os.path.join(tmp, "runs"))
            raise RuntimeError("an unfinished dump must not be scored")
        except AssertionError as ex:
            assert "no point.json" in str(ex)
        sw.dump_point(7, spec)                                                 # the resumed job redoes it
        assert os.path.exists(os.path.join(d, "point.json")) and len(pickle.load(open(os.path.join(d, "ends_val.pkl"), "rb"))) == 16
        blob = open(os.path.join(d, "ends_val.pkl"), "rb").read()
        open(os.path.join(d, "ends_val.pkl"), "wb").write(blob[:len(blob) // 2])      # a truncated pickle next to a complete point.json
        try:
            sw.dump_point(7, spec)
            raise RuntimeError("a truncated pickle was accepted")
        except AssertionError as ex:
            assert "does not match" in str(ex)
        PP.save_calib(f["calib"], os.path.join(tmp, "calib.pt"))
        assert not os.path.exists(os.path.join(tmp, "calib.pt.tmp")) and PP.load_calib(os.path.join(tmp, "calib.pt"), "cpu").frames == f["calib"].frames
''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
