import io
p = "evaluation/jwire_check.py"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:90], s.count(a))
    s = s.replace(a, b)
rep('def live_check(seed, pair_run, nmax=24, workers=2, device=None, ends="kp", bs=4):', 'def live_check(seed, pair_run, nmax=24, workers=2, device=None, ends="kp", bs=4, faithful=False):')
rep('live = JD.live_records(seed, "val_clean", 0.1, nmax, workers, bs=bs, device=dev, pair_run=pair_run, ends=ends)', 'live = JD.live_records(seed, "val_clean", 0.1, nmax, workers, bs=bs, device=dev, pair_run=pair_run, ends=ends, faithful=faithful)')
rep('lambda seed, split, thr, nmax, workers, bs=4, device=None, pair_run=None, ends="kp": seen.append(("live", split, nmax, pair_run, ends, bs)) or recs):', 'lambda seed, split, thr, nmax, workers, bs=4, device=None, pair_run=None, ends="kp", faithful=False: seen.append(("live", split, nmax, pair_run, ends, bs, faithful)) or recs):')
rep('assert live_check(0, "e324_s_s0", 24, 2, "cpu", "att")[0] and live_check(0, "e324_s_s0", 24, 2, "cpu", bs=8)[0]', 'assert live_check(0, "e324_s_s0", 24, 2, "cpu", "att")[0] and live_check(0, "e324_s_s0", 24, 2, "cpu", bs=8)[0] and live_check(0, "e324_s_s0", 24, 2, "cpu", "att", faithful=True)[0]')
rep('("live", "val_clean", 24, "e324_s_s0", "att", 4), ("path", "val", 0, "e324_s_s0", "kp"), ("load", 0, "e324_s_s0", "kp"), ("live", "val_clean", 24, "e324_s_s0", "kp", 8)], seen', '("live", "val_clean", 24, "e324_s_s0", "att", 4, False), ("path", "val", 0, "e324_s_s0", "kp"), ("load", 0, "e324_s_s0", "kp"), ("live", "val_clean", 24, "e324_s_s0", "kp", 8, False), ("path", "val", 0, "e324_s_s0", "att"), ("load", 0, "e324_s_s0", "att"), ("live", "val_clean", 24, "e324_s_s0", "att", 4, True)], seen')
rep('calls.append(x + (k.get("bs"),)) or (True, 24, {})), contextlib.redirect_stdout(io.StringIO()):\n                assert main(["live", "--seed", "0", "--pair-run", "e324_s_s0", "--read-out", "primary"]) == 0', 'calls.append(x + (k.get("bs"), k.get("faithful"))) or (True, 24, {})), contextlib.redirect_stdout(io.StringIO()):\n                assert main(["live", "--seed", "0", "--pair-run", "e324_s_s0", "--read-out", "primary"]) == 0')
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
