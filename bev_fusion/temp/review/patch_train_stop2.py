p = "training/train.py"
s = open(p, encoding="utf-8").read()
a = s.index("        # 3b stop request (autoresearch/trainsuite)")
b = s.index("        # 4 validation under the EMA weights restores the training weights afterwards")
new = '''        # 3b stop request (autoresearch/trainsuite): CONTROL / STOPPED of an earlier stop are moved aside at the start (no clocks compared); a
        # request that arrives while training stops the run at a step boundary with a checkpoint, a STOPPED marker and exit code 75 (not 0:
        # no `.done`); the run resumes from that checkpoint; the log rows carry the time `t`
        stop_cfg = dataclasses.replace(cfg, name="stoprun", steps=1, save_every=0, eval_every=0)
        sdir = os.path.join(tmp, "stoprun")
        os.makedirs(sdir)
        for n in ("CONTROL", "STOPPED"):
            with open(os.path.join(sdir, n), "w") as fh:
                json.dump({"cmd": "stop", "reason": "old"}, fh)
        train(stop_cfg, datasets=(base, base))
        assert not os.path.exists(os.path.join(sdir, "STOPPED")) and not os.path.exists(os.path.join(sdir, "CONTROL")), "old files not moved"
        assert os.path.exists(os.path.join(sdir, "CONTROL.old")) and os.path.exists(os.path.join(sdir, "STOPPED.old"))
        assert all("t" in json.loads(ln) for ln in open(os.path.join(sdir, "log.jsonl")) if '"val"' not in ln), "log rows carry t"
        real_stop, calls = globals()["_stop_requested"], [0]

        def stop_third(run_dir):                              # the stop request arrives while the run trains: seen at the third step
            calls[0] += 1
            return "selftest" if calls[0] > 2 else None
        stop_cfg2 = dataclasses.replace(stop_cfg, name="stoprun2", steps=4)
        globals()["_stop_requested"] = stop_third
        try:
            train(stop_cfg2, datasets=(base, base))
            raise AssertionError("the stop request must end the run")
        except SystemExit as e:
            assert e.code == EXIT_STOPPED, e.code
        finally:
            globals()["_stop_requested"] = real_stop
        d2 = os.path.join(tmp, "stoprun2")
        st = json.load(open(os.path.join(d2, "STOPPED")))
        assert st["step"] == 2 and st["reason"] == "selftest" and os.path.exists(os.path.join(d2, "ckpt_00000002.pt")), st
        assert not os.path.exists(os.path.join(d2, ".done")), "a stopped run is never done"
        train(stop_cfg2, datasets=(base, base))               # resume: from step 2 to 4, the old STOPPED is moved aside, no second stop
        assert os.path.exists(os.path.join(d2, "ckpt_00000004.pt")) and os.path.exists(os.path.join(d2, "STOPPED.old"))
        assert not os.path.exists(os.path.join(d2, "STOPPED")), "STOPPED of the earlier stop must not stay"
'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
