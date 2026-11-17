import io
p = "evaluation/jwire_check.py"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)
rep('def live_check(seed, pair_run, nmax=24, workers=2, device=None, ends="kp"):', 'def live_check(seed, pair_run, nmax=24, workers=2, device=None, ends="kp", bs=4):')
rep('live = JD.live_records(seed, "val_clean", 0.1, nmax, workers, device=dev, pair_run=pair_run, ends=ends)', 'live = JD.live_records(seed, "val_clean", 0.1, nmax, workers, bs=bs, device=dev, pair_run=pair_run, ends=ends)')
rep('mock.patch.object(JD, "live_records", lambda seed, split, thr, nmax, workers, device=None, pair_run=None, ends="kp": seen.append(("live", split, nmax, pair_run, ends)) or recs):', 'mock.patch.object(JD, "live_records", lambda seed, split, thr, nmax, workers, bs=4, device=None, pair_run=None, ends="kp": seen.append(("live", split, nmax, pair_run, ends, bs)) or recs):')
rep('assert live_check(0, "e324_s_s0", 24, 2, "cpu", "att")[0] and live_check(0, "e324_s_s0", 24, 2, "cpu")[0]', 'assert live_check(0, "e324_s_s0", 24, 2, "cpu", "att")[0] and live_check(0, "e324_s_s0", 24, 2, "cpu", bs=8)[0]')
rep('("live", "val_clean", 24, "e324_s_s0", "att"), ("path", "val", 0, "e324_s_s0", "kp"), ("load", 0, "e324_s_s0", "kp"), ("live", "val_clean", 24, "e324_s_s0", "kp")], seen', '("live", "val_clean", 24, "e324_s_s0", "att", 4), ("path", "val", 0, "e324_s_s0", "kp"), ("load", 0, "e324_s_s0", "kp"), ("live", "val_clean", 24, "e324_s_s0", "kp", 8)], seen                  # the batch size of the live pass reaches the loader (default 4, E358: 8 = the dump\'s)')
rep('mock.patch.object(sys.modules[__name__], "live_check", lambda *x: calls.append(x) or (True, 24, {}))', 'mock.patch.object(sys.modules[__name__], "live_check", lambda *x, **k: calls.append(x + (k.get("bs"),)) or (True, 24, {}))')
rep('assert [c[0:2] + c[-1:] for c in calls] == [(0, "e324_s_s0", "att"), (0, "e324_s_s0", "att"), (0, "e312_p_s0", "kp")], calls', 'assert [c[0:2] + c[5:6] for c in calls] == [(0, "e324_s_s0", "att"), (0, "e324_s_s0", "att"), (0, "e312_p_s0", "kp")], calls\n            assert [c[-1] for c in calls] == [4, 4, 4]                                                       # --bs defaults to 4 (E325 / E350), E358 passes --bs 8 for the dump\'s batch size')
rep('v.add_argument("--workers", type=int, default=2)', 'v.add_argument("--workers", type=int, default=2)\n    v.add_argument("--bs", type=int, default=4, help="the batch size of the live pass (default 4 as E325; the stored dumps were made with 8: E358 compares at --bs 8)")')
rep("ok, n, diff = live_check(a.seed, a.pair_run, a.max, a.workers, a.device, ends)", "ok, n, diff = live_check(a.seed, a.pair_run, a.max, a.workers, a.device, ends, bs=a.bs)")
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
